"""
Implements the Sequential CUB200 Dataset, as used in `Transfer without Forgetting <https://arxiv.org/abs/2206.00388>`_ (Version with ResNet50 as backbone).
"""
import logging
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple
from torch.utils.data import Dataset
from PIL import Image

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import store_masked_loaders, ContinualDataset
from utils.conf import base_path


class MyCUB200(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 224
    N_CLASSES = 200

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(MyCUB200.IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                logging.info('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download
                ln = '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21110&authkey=AIEfi5nlRyY1yaE" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
                logging.info('Downloading dataset')
                download(ln, filename=os.path.join(root, 'cub_200_2011.zip'), unzip=True, unzip_path=root, clean=True)

        data_file = np.load(os.path.join(root, 'train_data.npz' if train else 'test_data.npz'), allow_pickle=True)

        self.data = data_file['data']
        self.targets = torch.from_numpy(data_file['targets']).long()
        self.classes = data_file['classes']
        self.segs = data_file['segs']
        self._return_segmask = False

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

        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, not_aug_img, self.logits[index]] if hasattr(self, 'logits') else [
            img, target, not_aug_img]

        if self._return_segmask:
            # TODO: add to the return tuple
            raise "Unsupported segmentation output in training set!"

        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)


class CUB200(MyCUB200):
    """Base CUB200 dataset."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    def __getitem__(self, index: int, ret_segmask=False) -> Tuple[Image.Image, int, Image.Image]:
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

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, self.logits[index]] if hasattr(self, 'logits') else [img, target]

        if ret_segmask or self._return_segmask:
            # TODO: does not work with the current implementation
            seg = self.segs[index]
            seg = Image.fromarray(seg, mode='L')
            seg = transforms.ToTensor()(transforms.CenterCrop((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE))(seg))[0]
            ret_tuple.append((seg > 0).int())

        return ret_tuple

class SequentialCUB200RS(ContinualDataset):
    """
    Sequential CUB200 Dataset. Version with ResNet50 (as in `Transfer without Forgetting`)
    """
    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    TRANSFORM = transforms.Compose([
        transforms.Resize(MyCUB200.IMG_SIZE),
        transforms.RandomCrop(MyCUB200.IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = MyCUB200.TEST_TRANSFORM

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyCUB200(base_path() + 'CUB200', train=True,
                                   download=True, transform=SequentialCUB200RS.TRANSFORM)
        test_dataset = CUB200(base_path() + 'CUB200', train=False,
                              download=True, transform=SequentialCUB200RS.TEST_TRANSFORM)

        train, test = store_masked_loaders(
            train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200RS.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet50_pt"

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200RS.MEAN, SequentialCUB200RS.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCUB200RS.MEAN, SequentialCUB200RS.STD)
        return transform

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 16

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 30
