import os
from argparse import Namespace
from typing import Dict
from torch import nn
import importlib
import inspect

from models.utils.continual_model import ContinualModel


def get_all_models() -> dict:
    """
    Get a dictionary of all available models.
    """
    basepath: str = os.getenv("MAMMOTH_BASE_PATH", '.')

    return {model.split('.')[0].replace('_', '-'): model.split('.')[0] for model in os.listdir(os.path.join(basepath, 'models'))
            if not model.find('__') > -1 and not os.path.isdir(os.path.join(basepath, 'models', model))}


def get_model(args: Namespace, backbone: nn.Module, loss, transform, dataset) -> ContinualModel:
    """
    Return the class of the selected continual model among those that are available.
    If an error was detected while loading the available datasets, it raises the appropriate error message.

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


def get_model_class(args: Namespace) -> ContinualModel:
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
    return names[model_name]


def get_model_names() -> Dict[str, ContinualModel]:
    """
    Return the available continual model names and classes.

    Returns:
        A dictionary containing the names of the available continual models and their classes.
    """

    def _get_names():
        names: Dict[str, ContinualModel] = {}
        for model_name, model in get_all_models().items():
            try:
                mod = importlib.import_module('models.' + model)
                model_classe_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x)))
                                     and 'ContinualModel' in str(inspect.getmro(getattr(mod, x))[1:])][-1]
                c = getattr(mod, model_classe_name)
                names[c.NAME.replace('_', '-')] = c
            except Exception as e:
                names[model.replace('_', '-')] = e
        return names

    if not hasattr(get_model_names, 'names'):
        setattr(get_model_names, 'names', _get_names())
    return getattr(get_model_names, 'names')
