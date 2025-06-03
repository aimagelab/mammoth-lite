"""
Implements the Sequential CUB200 Dataset, as used in `Transfer without Forgetting <https://arxiv.org/abs/2206.00388>`_ (Version with ResNet50 as backbone).
"""
import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from typing import Tuple
from torch.utils.data import Dataset
from PIL import Image

from datasets import register_dataset
from datasets.utils.continual_dataset import ContinualDataset

from utils.conf import base_path


class MyCUB200(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 224
    N_CLASSES = 200

    def __init__(self, root, train=True, transform=None,
                 target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset not found at {root}. Please download the 'CUB_200_2011' dataset and place it in the 'data' directory.")

        f_images = pd.read_csv(os.path.join(self.root, 'images.txt'), delim_whitespace=True, names=['id', 'path'],
                               header=None)
        f_targets = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'), delim_whitespace=True,
                                names=['id', 'class'], header=None)

        self.data = np.array([os.path.join(self.root, 'images', path) for path in f_images['path'].to_list()])
        self.targets = np.array(f_targets['class'].to_list()) - 1  # convert to zero-based index

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
        img = Image.open(img).convert("RGB")

        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, not_aug_img, self.logits[index]] if hasattr(self, 'logits') else [
            img, target, not_aug_img]

        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)

@register_dataset('seq-cub200')
class SequentialCUB200(ContinualDataset):
    """
    Sequential CUB200 Dataset. Version with ResNet50 (as in `Transfer without Forgetting`)
    """
    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(MyCUB200.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyCUB200(base_path() + 'CUB_200_2011', train=True, transform=SequentialCUB200.TRANSFORM)
        test_dataset = MyCUB200(base_path() + 'CUB_200_2011', train=False, transform=SequentialCUB200.TEST_TRANSFORM)

        return train_dataset, test_dataset

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return "resnet18_7x7_pt"

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)
