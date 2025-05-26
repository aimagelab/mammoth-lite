import importlib
import os
from argparse import Namespace
from typing import Callable, TYPE_CHECKING, Union
from torch import nn

from utils import register_dynamic_module_fn

from models.utils.continual_model import ContinualModel

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset
    from torchvision.transforms import Compose

REGISTERED_MODELS = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a ContinualModel. The decorator may be used on a class that inherits from `ContinualModel` or on a function that returns a `ContinualModel` instance.
    The registered model can be accessed using the `get_model` function and can include additional keyword arguments to be set during parsing.

    The arguments can be inferred by the *signature* of the model's class.
    The value of the argument is the default value. If the default is set to `Parameter.empty`, the argument is required. If the default is set to `None`, the argument is optional. The type of the argument is inferred from the default value (default is `str`).

    Args:
        name: the name of the model
    """
    if hasattr(get_model_names, 'names'):  # reset the cache of the model names
        del get_model_names.names

    return register_dynamic_module_fn(name, REGISTERED_MODELS)


def get_all_models() -> dict[str, str]:
    """
    Get a dictionary of all available models.
    """
    basepath: str = os.getenv("MAMMOTH_BASE_PATH", '.')

    return {model.split('.')[0].replace('_', '-'): model.split('.')[0]
            for model in os.listdir(os.path.join(basepath, 'models'))
            if not model.find('__') > -1 and not os.path.isdir(os.path.join(basepath, 'models', model))}


def get_model_names(names_only=False):
    """
    Return the names of the available continual models.
    If an error was detected while loading the available models, it raises the appropriate error message.

    Args:
        names_only (bool): whether to return only the names of the available models

    Exceptions:
        AssertError: if the model is not available
        Exception: if an error is detected in the model

    Returns:
        the named of the available continual models
    """

    def _model_names():
        names = {}  # key: model name, value: {'class': model class, 'parsable_args': parsable_args}
        for model, model_conf in REGISTERED_MODELS.items():
            names[model.replace('_', '-')] = {'class': model_conf['class'], 'parsable_args': model_conf['parsable_args']}
        return names

    if not hasattr(get_model_names, 'names'):
        setattr(get_model_names, 'names', _model_names())
    names = getattr(get_model_names, 'names')
    if names_only:
        return list(names.keys())
    return names


def get_model(args: Namespace, backbone: nn.Module, loss: nn.Module, transform: Union['Compose', nn.Module], dataset: 'ContinualDataset') -> 'ContinualModel':
    """
    Return the class of the selected continual model among those that are available.
    If an error was detected while loading the available models, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the `--model` attribute
        backbone (nn.Module): the backbone of the model
        loss: the loss function
        transform: the transform function
        dataset: the instance of the dataset

    Exceptions:
        AssertError: if the model is not available
        Exception: if an error is detected in the model

    Returns:
        the continual model instance
    """
    model_name = args.model.replace('_', '-')
    names = get_model_names()
    assert model_name in names
    return get_model_class(args)(backbone, loss, args, transform, dataset)

def get_model_class(args: Namespace) -> 'ContinualModel':
    """
    Return the class of the selected continual model among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

    Args:
        args (Namespace): the arguments which contains the `--model` attribute

    Exceptions:
        AssertError: if the model is not available
        Exception: if an error is detected in the model

    Returns:
        the continual model class
    """
    names = get_model_names()
    model_name = args.model.replace('_', '-')
    assert model_name in names
    if isinstance(names[model_name], Exception):
        raise names[model_name]
    return names[model_name]['class']

# import all files in the `models` folder to register the models
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file != '__init__.py':
        importlib.import_module(f'models.{file[:-3]}')
