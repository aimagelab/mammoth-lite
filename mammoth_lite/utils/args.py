import sys
from typing import Optional

if __name__ == '__main__':
    import os
    mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, mammoth_path)

from argparse import ArgumentParser

from backbone import REGISTERED_BACKBONES
from datasets import get_dataset_names
from models import get_model_names
from . import binary_to_boolean_type # type: ignore


def add_initial_args(parser) -> ArgumentParser:
    """
    Returns the initial parser for the arguments.
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=get_dataset_names(names_only=True),
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_model_names())
    parser.add_argument('--backbone', type=str, help='Backbone network name.', choices=list(REGISTERED_BACKBONES.keys()))

    return parser


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.

    Args:
        parser: the parser instance

    Returns:
        None
    """
    exp_group = parser.add_argument_group('Experiment arguments', 'Arguments used to define the experiment settings.')

    exp_group.add_argument('--lr', required=True, type=float, help='Learning rate. This should either be set as default by the model '
                           '(with `set_defaults <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.set_defaults>`_),'
                           ' by the dataset (with `set_default_from_args`, see :ref:`module-datasets.utils`), or with `--lr=<value>`.')
    exp_group.add_argument('--batch_size', type=int, help='Batch size.')
    exp_group.add_argument('--joint', type=int, choices=(0, 1), default=0, help='Train model on Joint (single task)?')
    exp_group.add_argument('--n_epochs', type=int,
                           help='Number of epochs. Used only if `fitting_mode=epochs`.')

    opt_group = parser.add_argument_group('Optimizer and learning rate scheduler arguments', 'Arguments used to define the optimizer and the learning rate scheduler.')

    opt_group.add_argument('--optimizer', type=str, default='sgd', help='Optimizer.')
    opt_group.add_argument('--optim_wd', type=float, default=0., help='optimizer weight decay.')
    opt_group.add_argument('--optim_mom', type=float, default=0., help='optimizer momentum.')
    opt_group.add_argument('--optim_nesterov', type=binary_to_boolean_type, default=0, help='optimizer nesterov momentum.')


def add_management_args(parser: ArgumentParser) -> None:
    """
    Adds the management arguments.

    Args:
        parser: the parser instance

    Returns:
        None
    """
    mng_group = parser.add_argument_group('Management arguments', 'Generic arguments to manage the experiment reproducibility, logging, debugging, etc.')

    mng_group.add_argument('--num_workers', type=int, default=None, help='Number of workers for the dataloaders (default=infer from number of cpus).')
    mng_group.add_argument('--debug_mode', type=binary_to_boolean_type, default=0, help='Run only a few training steps per epoch. This also disables logging on wandb.')
    mng_group.add_argument('--savecheck', choices=['last', 'task'], type=str, help='Save checkpoint every `task` or at the end of the training (`last`).')
    mng_group.add_argument('--loadcheck', type=str, default=None, help='Path of the checkpoint to load (.pt file for the specific task)')

    wandb_group = parser.add_argument_group('Wandb arguments', 'Arguments to manage logging on Wandb.')

    wandb_group.add_argument('--wandb_name', type=str, default=None,
                             help='Wandb name for this run. Overrides the default name (`args.model`).')
    wandb_group.add_argument('--wandb_entity', type=str, help='Wandb entity')
    wandb_group.add_argument('--wandb_project', type=str, help='Wandb project name')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods

    Args:
        parser: the parser instance

    Returns:
        None
    """
    group = parser.add_argument_group('Rehearsal arguments', 'Arguments shared by all rehearsal-based methods.')

    group.add_argument('--buffer_size', type=int, required=True,
                       help='The size of the memory buffer.')
    group.add_argument('--minibatch_size', type=int,
                       help='The batch size of the memory buffer.')


def check_multiple_defined_arg_during_string_parse() -> None:
    """
    Check if an argument is defined multiple times during the string parsing.
    Prevents the user from typing the same argument multiple times as:
    `--arg1=val1 --arg1=val2`.
    """
    arg_name: Optional[str]

    cmd_args = sys.argv[1:]
    keys = set()
    for i, arg in enumerate(cmd_args):
        if '=' in arg:
            arg_name = arg.split('=')[0]
        else:
            arg_name = arg if arg.startswith('-') else None
        if arg_name is not None and arg_name in keys:
            raise ValueError(f"Argument `{arg_name}` is defined multiple times.")
        keys.add(arg_name)
