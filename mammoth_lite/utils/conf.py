"""
This module contains utility functions for configuration settings.
"""

import logging
import os
import torch


def get_device() -> torch.device:
    """
    Returns the least used GPU device if available else MPS or CPU.
    """
    def _get_device() -> torch.device:
        # get least used gpu by used memory
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device('cuda')
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("WARNING: MSP support is still experimental. Use at your own risk!")
                return torch.device("mps")
        except BaseException:
            print("ERROR: Something went wrong with MPS. Using CPU.")

        print("WARNING: No GPU available. Using CPU.")
        return torch.device("cpu")

    # Permanently store the chosen device
    if not hasattr(get_device, 'device'):
        get_device.device = _get_device()  # type: ignore
        print(f'Using device {get_device.device}')  # type: ignore

    return get_device.device  # type: ignore


def base_path() -> str:
    """
    Returns the base path where to save data.

    Returns:
        the base path (default: `./data/`)
    """
    if os.getenv('MAMMOTH_NOTEBOOK'):
        return './data/'
    else:
        return '../data/'