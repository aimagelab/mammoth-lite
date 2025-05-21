from typing import Union
from PIL.Image import Image
import numpy as np
import torch


class DeNormalize(object):
    def __init__(self, mean, std):
        """
        Initializes a DeNormalize object.

        Args:
            mean (list): List of mean values for each channel.
            std (list): List of standard deviation values for each channel.
        """
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean)
        elif isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std)

        self.mean = mean
        self.std = std

    def __call__(self, tensor: Union[torch.Tensor, Image]):
        """
        Applies denormalization to the input tensor.

        Args:
            tensor (Tensor): Tensor of images of size ([B,] C, H, W) to be denormalized.

        Returns:
            Tensor: Denormalized tensor.
        """
        if isinstance(tensor, Image):
            tensor = torch.tensor(np.array(tensor).transpose(2, 0, 1)).float()

        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.device != self.mean.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)

        return (tensor * self.std[:, None, None]) + self.mean[:, None, None]
