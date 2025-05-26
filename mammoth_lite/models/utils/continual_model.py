"""
This is the base class for all models. It provides some useful methods and defines the interface of the models.

The `observe` method is the most important one: it is called at each training iteration and it is responsible for computing the loss and updating the model's parameters.

The `begin_task` and `end_task` methods are called before and after each task, respectively.

The `get_parser` method returns the parser of the model. Additional model-specific hyper-parameters can be added by overriding this method.

The `get_optimizer` method returns the optimizer to be used for training. Default: SGD.

The `load_buffer` method is called when a buffer is loaded. Default: do nothing.
"""

from abc import abstractmethod
import logging
from argparse import ArgumentParser, Namespace
from typing import List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim

from utils.buffer import Buffer
from utils.conf import get_device
from torchvision import transforms  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset
    from backbone import MammothBackbone


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str] = ['class-il', 'task-il']

    args: Namespace  # The command line arguments
    device: torch.device  # The device to be used for training
    net: 'MammothBackbone'  # The backbone of the model (defined by the `dataset`)
    loss: nn.Module  # The loss function to be used (defined by the `dataset`)
    opt: optim.Optimizer  # The optimizer to be used for training
    transform: transforms.Compose # The transformation to be applied to the input data. The model will try to convert it to a kornia transform to be applicable to a batch of samples at once
    dataset: 'ContinualDataset'  # The instance of the dataset. Used to update the number of classes in the current task
    num_classes: int  # The total number of classes
    n_tasks: int  # The number of tasks

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        """
        Defines model-specific hyper-parameters, which will be added to the command line arguments. 
        Additional model-specific hyper-parameters can be added by overriding this method.
        """
        return parser

    def __init__(self, backbone: 'MammothBackbone', loss: nn.Module,
                 args: Namespace, transform: nn.Module, dataset: Optional['ContinualDataset'] = None) -> None:
        super(ContinualModel, self).__init__()
        print("Loading model: ", self.NAME)
        print(f"- Using {backbone.__class__.__name__} as backbone")
        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.dataset = dataset
        
        self.num_classes = self.dataset.N_CLASSES
        self.n_tasks = self.dataset.N_TASKS

        self.normalization_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), self.dataset.get_normalization_transform()])

        self.opt = self.get_optimizer()
        self.device = get_device()

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

    def to(self, device):
        """
        Captures the device to be used for training.
        """
        self.device = device
        return super().to(device)

    def load_buffer(self, buffer):
        """
        Default way to handle load buffer.
        """

        if isinstance(buffer, Buffer):
            assert buffer.examples.shape[0] == self.args.buffer_size, "Buffer size mismatch. Expected {} got {}".format(
                self.args.buffer_size, buffer.examples.shape[0])
            self.buffer = buffer
        elif isinstance(buffer, dict):  # serialized buffer
            assert 'examples' in buffer, "Buffer does not contain examples"
            assert self.buffer.buffer_size == buffer['examples'].shape[0], "Buffer size mismatch. Expected {} got {}".format(
                self.buffer.buffer_size, buffer['examples'].shape[0])
            for k, v in buffer.items():
                setattr(self.buffer, k, v)
            self.buffer.attributes = list(buffer.keys())
            self.buffer.num_seen_examples = buffer['examples'].shape[0]
        else:
            raise ValueError("Buffer type not recognized")

    def get_optimizer(self, params: Optional[List[torch.Tensor]] = None, lr: Optional[float]=None) -> optim.Optimizer:
        """
        Create and return the optimizer to be used for training.

        Args:
            params: the parameters to be optimized. If None, the default specified by `get_parameters` is used.
            lr: the learning rate. If None, the default specified by the command line arguments is used.

        Returns:
            the optimizer
        """

        params: list[torch.Tensor] = params if params is not None else list(self.net.parameters())
        lr: float = lr if lr is not None else self.args.lr

        # check if optimizer is in torch.optim
        if self.args.optimizer.lower() == 'sgd':
            opt = optim.SGD(params, lr=lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom, nesterov=self.args.optim_nesterov)
        elif self.args.optimizer.lower() == 'adam':
            opt = optim.Adam(params, lr=lr, weight_decay=self.args.optim_wd)
        elif self.args.optimizer.lower() == 'adamw':
            opt = optim.AdamW(params, lr=lr, weight_decay=self.args.optim_wd)

        return opt

    def begin_task(self, dataset: 'ContinualDataset') -> None:
        """
        Prepares the model for the current task.
        Executed before each task.
        """
        pass

    def end_task(self, dataset: 'ContinualDataset') -> None:
        """
        Prepares the model for the next task.
        Executed after each task.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.

        Args:
            x: batch of inputs
            task_label: some models require the task label

        Returns:
            the result of the computation
        """
        return self.net(x)

    @abstractmethod
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: Optional[int] = None) -> float:
        """
        Compute a training step over a given batch of examples.

        Args:
            inputs: batch of examples
            labels: ground-truth labels
            kwargs: some methods could require additional parameters

        Returns:
            the value of the loss function
        """
        raise NotImplementedError
