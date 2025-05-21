from argparse import Namespace
import os
from typing import Callable, Optional, Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class MammothDataset(Dataset):
    data: np.ndarray
    targets: np.ndarray


class ContinualDataset(object):
    """
    A base class for defining continual learning datasets.

    Data is divided into tasks and loaded only when the `get_data_loaders` method is called.

    Attributes:
        NAME (str): the name of the dataset
        SETTING (str): the setting of the dataset
        N_CLASSES_PER_TASK (int): the number of classes per task
        N_TASKS (int): the number of tasks
        N_CLASSES (int): the number of classes
        SIZE (Tuple[int]): the size of the dataset
        AVAIL_SCHEDS (List[str]): the available schedulers
        class_names (List[str]): list of the class names of the dataset (should be populated by `get_class_names`)
        train_loader (DataLoader): the training loader
        test_loaders (List[DataLoader]): the test loaders
        i (int): the current task
        c_task (int): the current task
        args (Namespace): the arguments which contains the hyperparameters
        eval_fn (Callable): the function used to evaluate the model on the dataset
    """

    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    N_CLASSES: int
    SIZE: Tuple[int]

    log_fn: Callable
    train_loader: torch.utils.data.DataLoader
    test_loaders: list[torch.utils.data.DataLoader] = []

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.

        Args:
            args: the arguments which contains the hyperparameters
        """
        self.args = args
        self.c_task = -1

        if 'class-il' in self.SETTING:
            self.N_CLASSES = self.N_CLASSES if hasattr(self, 'N_CLASSES') else \
                (self.N_CLASSES_PER_TASK * self.N_TASKS) if isinstance(self.N_CLASSES_PER_TASK, int) else sum(self.N_CLASSES_PER_TASK)
        else:
            self.N_CLASSES = self.N_CLASSES_PER_TASK

        if args.joint:
            if self.SETTING in ['class-il', 'task-il']:
                # just set the number of classes per task to the total number of classes
                self.N_CLASSES_PER_TASK = self.N_CLASSES
                self.N_TASKS = 1
            else:
                # bit more tricky, not supported for now
                raise NotImplementedError('Joint training is only supported for class-il and task-il.'
                                          'For other settings, please use the `joint` model with `--model=joint` and `--joint=0`')

    def get_offsets(self, c_task: Optional[int] = None) -> Tuple[int, int]:
        if c_task is None:
            c_task = self.c_task
        if 'class-il' in self.SETTING or 'task-il' in self.SETTING:
            start_c = self.N_CLASSES_PER_TASK * c_task
            end_c = self.N_CLASSES_PER_TASK * (c_task + 1)
        else:
            start_c = 0
            end_c = self.N_CLASSES
        return start_c, end_c

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.

        Returns:
            the current training and test loaders
        """
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> str:
        """Returns the name of the backbone to be used for the current dataset. This can be changes using the `--backbone` argument or by setting it in the `dataset_config`."""
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """Returns the transform to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """Returns the loss to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """Returns the transform used for normalizing the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """Returns the transform used for denormalizing the current dataset."""
        raise NotImplementedError


def store_masked_loaders(train_dataset: MammothDataset, test_dataset: MammothDataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.

    Attributes:
        train_dataset (Dataset): the training dataset
        test_dataset (Dataset): the test dataset
        setting (ContinualDataset): the setting of the dataset

    Returns:
        the training and test loaders
    """
    # Initializations
    if 'class-il' in setting.SETTING or 'task-il' in setting.SETTING:
        setting.c_task += 1

    assert hasattr(train_dataset, 'targets'), 'Targets missing from train dataset.'
    assert hasattr(train_dataset, 'data'), 'Data missing from train dataset.'

    assert hasattr(test_dataset, 'targets'), 'Targets missing from test dataset.'
    assert hasattr(test_dataset, 'data'), 'Data missing from test dataset.'

    if not isinstance(train_dataset.targets, np.ndarray):
        train_dataset.targets = np.array(train_dataset.targets)
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)

    # Split the dataset into tasks
    if 'class-il' in setting.SETTING or 'task-il' in setting.SETTING:
        start_c, end_c = setting.N_CLASSES_PER_TASK * setting.c_task, setting.N_CLASSES_PER_TASK * (setting.c_task + 1)

        train_mask = np.logical_and(train_dataset.targets >= start_c,
                                    train_dataset.targets < end_c)

        test_mask = np.logical_and(test_dataset.targets >= start_c,
                                       test_dataset.targets < end_c)
        
        test_dataset.data = test_dataset.data[test_mask]
        test_dataset.targets = test_dataset.targets[test_mask]

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = train_dataset.targets[train_mask]

    n_cpus = 4 if not hasattr(os, 'sched_getaffinity') else len(os.sched_getaffinity(0))
    num_workers = min(8, n_cpus) if setting.args.num_workers is None else setting.args.num_workers  # limit to 8 cpus if not specified

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=num_workers)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader
