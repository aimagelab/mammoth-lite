from typing import List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn


if TYPE_CHECKING:
    pass


class BaseSampleSelection:
    """
    Base class for sample selection strategies.
    """

    def __init__(self, buffer_size: int, device):
        """
        Initialize the sample selection strategy.

        Args:
            buffer_size: the maximum buffer size
            device: the device to store the buffer on
        """
        self.buffer_size = buffer_size
        self.device = device

    def __call__(self, num_seen_examples: int) -> int:
        """
        Selects the index of the sample to replace.

        Args:
            num_seen_examples: the number of seen examples

        Returns:
            the index of the sample to replace
        """

        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        (optional) Update the state of the sample selection strategy.
        """
        pass


class ReservoirSampling(BaseSampleSelection):
    def __call__(self, num_seen_examples: int) -> int:
        """
        Reservoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < self.buffer_size:
            return rand
        else:
            return -1

class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    buffer_size: int  # the maximum size of the buffer
    device: str|torch.device  # the device to store the buffer on
    num_seen_examples: int  # the total number of examples seen, used for reservoir
    attributes: List[str]  # the attributes stored in the buffer

    examples: torch.Tensor  # (mandatory) buffer attribute: the tensor of images
    labels: torch.Tensor  # (optional) buffer attribute: the tensor of labels
    logits: torch.Tensor  # (optional) buffer attribute: the tensor of logits
    task_labels: torch.Tensor  # (optional) buffer attribute: the tensor of task labels
    true_labels: torch.Tensor  # (optional) buffer attribute: the tensor of true labels

    def __init__(self, buffer_size: int, device: str | torch.device = "cpu"):
        """
        Initialize a reservoir-based Buffer object.

        Supports storing images, labels, logits, and task_labels. This can be extended by adding more attributes to the `attributes` list and updating the `init_tensors` method accordingly.

        Args:
            buffer_size (int): The maximum size of the buffer.
            device (str | torch.device, optional): The device to store the buffer on. Defaults to "cpu".

        Note:
            If during the `get_data` the transform is PIL, data will be moved to cpu and then back to the device. This is why the device is set to cpu by default.
        """
        self._dl_transform = None
        self._it_index = 0
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

        self.sample_selection_fn = ReservoirSampling(buffer_size, device)

    def serialize(self, out_device='cpu'):
        """
        Serialize the buffer.

        Returns:
            A dictionary containing the buffer attributes.
        """
        return {attr_str: getattr(self, attr_str).to(out_device) for attr_str in self.attributes if hasattr(self, attr_str)}

    def to(self, device):
        """
        Move the buffer and its attributes to the specified device.

        Args:
            device: The device to move the buffer and its attributes to.

        Returns:
            The buffer instance with the updated device and attributes.
        """
        self.device = device
        self.sample_selection_fn.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        """
        Returns the number items in the buffer.
        """
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor,
                     true_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            true_labels: tensor containing the true labels (used only for logging)
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):  # create tensor if not already present
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
            elif hasattr(self, attr_str):  # if tensor already exists, update it and possibly resize it according to the buffer_size
                if self.num_seen_examples < self.buffer_size:  # if the buffer is full, extend the tensor
                    old_tensor = getattr(self, attr_str)
                    pad = torch.zeros((self.buffer_size - old_tensor.shape[0], *attr.shape[1:]), dtype=old_tensor.dtype, device=self.device)
                    setattr(self, attr_str, torch.cat([old_tensor, pad], dim=0))

    @property
    def used_attributes(self):
        """
        Returns a list of attributes that are currently being used by the object.
        """
        return [attr_str for attr_str in self.attributes if hasattr(self, attr_str)]

    def is_full(self):
        return self.num_seen_examples >= self.buffer_size

    def add_data(self, examples, labels=None, logits=None, task_labels=None, true_labels=None, sample_selection_scores=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            true_labels: if setting is noisy, the true labels associated with the examples. **Used only for logging.**
            sample_selection_scores: tensor containing the scores used for the sample selection strategy. NOTE: this is only used if the sample selection strategy defines the `update` method.

        Note:
            Only the examples are required. The other tensors are initialized only if they are provided.
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, true_labels)

        for i in range(examples.shape[0]):
            index: int = self.sample_selection_fn(self.num_seen_examples)
            
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if sample_selection_scores is not None:
                    self.sample_selection_fn.update(index, sample_selection_scores[i])
                if true_labels is not None:
                    self.true_labels[index] = true_labels[i].to(self.device)

    def get_data(self, size: int,
                 transform: Optional[nn.Module] = None,
                 device: str|torch.device|None=None) -> Sequence[torch.Tensor]:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            device: the device to store the data on. If None, uses the device of the buffer.

        Returns:
            a tuple containing the requested items. If return_index is True, the tuple contains the indexes as first element.
        """
        target_device: str|torch.device = self.device if device is None else device
        num_avail_samples: int = min(self.num_seen_examples, self.examples.shape[0])

        # if the buffer has not enough samples, change the size accordingly
        if size > min(num_avail_samples, self.examples.shape[0]):
            size: int = min(num_avail_samples, self.examples.shape[0])

        choice: np.ndarray = np.random.choice(num_avail_samples, size=size, replace=False)
        if transform is None:
            def transform(x): return x

        selected_samples: torch.Tensor = self.examples[choice]

        ret_tuple: List[torch.Tensor] = [torch.stack([transform(sample) for sample in selected_samples.cpu()], dim=0).to(target_device)]
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr: torch.Tensor = getattr(self, attr_str)
                ret_tuple += [attr[choice].to(target_device)]
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def reset(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0