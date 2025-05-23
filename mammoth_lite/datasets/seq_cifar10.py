from typing import Tuple

import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from utils.conf import base_path
from datasets import register_dataset
from datasets.utils.continual_dataset import (ContinualDataset, MammothDataset)


class MyCIFAR10(MammothDataset, CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None) -> None:
        # not self._check_integrity() -> trick to avoid printing debug messages
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, not_aug_img


@register_dataset(name='seq-cifar10')
class SequentialCIFAR10(ContinualDataset):
    """Sequential CIFAR10 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self):
        """Class method that returns the train and test loaders."""
        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=self.TRANSFORM)
        test_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=False,
                                download=True, transform=self.TEST_TRANSFORM)

        return train_dataset, test_dataset

    @staticmethod
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR10.MEAN, SequentialCIFAR10.STD)
        return transform
