import inspect
import os
from typing import Any, Callable, Optional, Type


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.

    Args:
        path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def binary_to_boolean_type(value: str) -> bool:
    """
    Converts a binary string to a boolean type.

    Args:
        value: the binary string

    Returns:
        the boolean type
    """
    if not isinstance(value, str):
        value = str(value)

    value = value.lower()
    true_values = ['true', '1', 't', 'y', 'yes']
    false_values = ['false', '0', 'f', 'n', 'no']

    assert value in true_values + false_values

    return value in true_values


def infer_args_from_signature(signature: inspect.Signature, excluded_signature: Optional[inspect.Signature] = None) -> dict:
    """
    Load the arguments of a function from its signature.

    Args:
        signature: the signature of the function

    Returns:
        the inferred arguments
    """
    excluded_args = {} if excluded_signature is None else list(excluded_signature.parameters.keys())
    parsable_args = {}

    for arg_name, value in list(signature.parameters.items()):
        if arg_name in excluded_args:
            continue
        if arg_name != 'self' and not arg_name.startswith('_'):
            default = value.default
            tp: Type[Any] = str
            if value.annotation is not inspect._empty:
                tp = value.annotation
            elif default is not inspect.Parameter.empty:
                tp = type(default)
            if default is inspect.Parameter.empty and arg_name != 'num_classes':
                parsable_args[arg_name] = {
                    'type': tp,
                    'required': True
                }
            else:
                parsable_args[arg_name] = {
                    'type': tp,
                    'required': False,
                    'default': default if default is not inspect.Parameter.empty else None
                }
    return parsable_args


def register_dynamic_module_fn(name: str, register: dict):
    """
    Register a dynamic module in the specified dictionary.

    Args:
        name: the name of the module
        register: the dictionary where the module will be registered
        cls: the class to be registered
        tp: the type of the class, used to dynamically infer the arguments
    """
    name = name.replace('_', '-').lower()

    def register_network_fn(target: Callable) -> Callable:
        # check if the name is already registered
        if name in register:
            raise ValueError(f"Name {name} already registered!")

        signature = inspect.signature(target)

        parsable_args = infer_args_from_signature(signature)
        register[name] = {'class': target, 'parsable_args': parsable_args}
        return target

    return register_network_fn
