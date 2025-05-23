"""
Datasets can be included either by registering them using the `register_dataset` decorator or by following the old naming convention:
- A single dataset is defined in a file named `<dataset_name>.py` in the `datasets` folder.
- The dataset class must inherit from `ContinualDataset`.
"""

import os
from typing import Callable, Tuple, Type
import importlib
from argparse import Namespace

from utils import register_dynamic_module_fn
from datasets.utils.continual_dataset import ContinualDataset

REGISTERED_DATASETS: dict[str, dict] = dict()  # dictionary containing the registered datasets. Template: {name: {'class': class, 'parsable_args': parsable_args}}


def register_dataset(name: str) -> Callable:
    """
    Decorator to register a ContinualDatasety. The decorator may be used on a class that inherits from `ContinualDataset` or on a function that returns a `ContinualDataset` instance.
    The registered dataset can be accessed using the `get_dataset` function and can include additional keyword arguments to be set during parsing.

    The arguments can be inferred by the *signature* of the dataset's class.
    The value of the argument is the default value. If the default is set to `Parameter.empty`, the argument is required. If the default is set to `None`, the argument is optional. The type of the argument is inferred from the default value (default is `str`).

    Args:
        name: the name of the dataset
    """
    if hasattr(get_dataset_names, 'names'):  # reset the cache of the dataset names
        del get_dataset_names.names

    return register_dynamic_module_fn(name, REGISTERED_DATASETS)

def get_dataset_names(names_only=False):
    """
    Return the names of the available continual dataset.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        names_only (bool): whether to return only the names of the available datasets

    Exceptions:
        AssertError: if the dataset is not available
        Exception: if an error is detected in the dataset

    Returns:
        the named of the available continual datasets
    """

    def _dataset_names():
        names = {}  # key: dataset name, value: {'class': dataset class, 'parsable_args': parsable_args}
        for dataset, dataset_conf in REGISTERED_DATASETS.items():
            names[dataset.replace('_', '-')] = {'class': dataset_conf['class'], 'parsable_args': dataset_conf['parsable_args']}
        return names

    if not hasattr(get_dataset_names, 'names'):
        setattr(get_dataset_names, 'names', _dataset_names())
    names = getattr(get_dataset_names, 'names')
    if names_only:
        return list(names.keys())
    return names

def get_dataset_class(args: Namespace) -> Tuple[Type[ContinualDataset], dict]:
    """
    Return the class of the selected continual dataset among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the `--dataset` attribute
        return_args (bool): whether to return the parsable arguments of the dataset

    Exceptions:
        AssertError: if the dataset is not available
        Exception: if an error is detected in the dataset

    Returns:
        the continual dataset class
    """
    names = get_dataset_names()
    assert args.dataset in names
    if isinstance(names[args.dataset], Exception):
        raise names[args.dataset]
    return names[args.dataset]['class'], names[args.dataset]['parsable_args']  # type: ignore


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the hyperparameters

    Exceptions:
        AssertError: if the dataset is not available
        Exception: if an error is detected in the dataset

    Returns:
        the continual dataset instance
    """
    dataset_class, dataset_args = get_dataset_class(args)
    missing_args = [arg for arg in dataset_args.keys() if arg not in vars(args) and arg != 'args']
    assert len(missing_args) == 0, "Missing arguments for the dataset: " + ', '.join(missing_args)

    parsed_args = {arg: getattr(args, arg) for arg in dataset_args.keys() if arg != 'args'}

    return dataset_class(args, **parsed_args)


# import all files in the `datasets` folder to register the datasets
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file != '__init__.py':
        importlib.import_module(f'datasets.{file[:-3]}')
