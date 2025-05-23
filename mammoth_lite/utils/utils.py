import json
import os
from argparse import Namespace, ArgumentParser
from typing import Any, Tuple, TYPE_CHECKING

from .args import add_initial_args, add_management_args, add_experiment_args, add_rehearsal_args

from models import get_model
from datasets import get_dataset
from backbone import get_backbone

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset

def load_runner(model_name: str, dataset_name: str, experiment_args: dict[str, Any]) -> Tuple['ContinualModel', 'ContinualDataset']:
    """
    Load the model and dataset for the given experiment.
    Args:
        model_name (str): The name of the model to be used.
        dataset_name (str): The name of the dataset to be used.
        experiment_args (dict): A dictionary containing the arguments for the experiment.

    Returns:
        Tuple[ContinualModel, ContinualDataset]: A tuple containing the model and dataset.
    """
    assert 'lr' in experiment_args, "Learning rate not specified in experiment arguments. Add 'lr' to the experiment arguments."
    assert 'batch_size' in experiment_args, "Batch size not specified in experiment arguments. Add 'batch_size' to the experiment arguments."
    assert 'n_epochs' in experiment_args, "Number of epochs not specified in experiment arguments. Add 'n_epochs' to the experiment arguments."

    args = initalize_args(model_name, dataset_name, experiment_args)
    dataset = get_dataset(args)
    args.backbone = args.backbone if args.backbone else dataset.get_backbone()
    args.num_classes = dataset.N_CLASSES
    backbone = get_backbone(args)
    model = get_model(args, backbone, dataset.get_loss(), dataset.get_transform(), dataset)
    
    os.environ['MAMMOTH_ARGS'] = json.dumps(vars(args))

    return model, dataset
    

def initalize_args(model_name: str, dataset_name: str, experiment_args: dict[str, Any]) -> Namespace:
    parser = ArgumentParser(allow_abbrev=False, add_help=False)

    add_initial_args(parser)
    add_management_args(parser)
    add_experiment_args(parser)

    if 'buffer_size' in experiment_args:
        add_rehearsal_args(parser)

    exp_str = [f"--{k}={v}" for k, v in experiment_args.items() if k not in ['model', 'dataset', 'backbone'] and v is not None]
    exp_str += ['--dataset', dataset_name, '--model', model_name]
    if 'backbone' in experiment_args:
        exp_str += ['--backbone', experiment_args['backbone']]

    return parser.parse_args(exp_str)

def get_avail_args() -> Tuple[dict[str, dict], dict[str, dict]]:
    """
    Get the available arguments for the Mammoth Lite framework.
    This function returns two lists: one for required arguments and one for optional arguments.
    Each list contains dictionaries with:
    - the name of the argument as the key
    - a dictionary with the 'default' and 'description' of the argument as the value
    Note that the 'default' key is only present for optional arguments.
    """
    parser = ArgumentParser(allow_abbrev=False, add_help=False)

    add_initial_args(parser)
    add_management_args(parser)
    add_experiment_args(parser)

    required_args, optional_args = {}, {}
    for group in parser._action_groups:
        for action in group._group_actions:
            if action.dest not in ['help', 'debug_mode']:
                if action.required:
                    required_args[action.dest] = {
                        'description': action.help
                    }
                else:
                    optional_args[action.dest] = {
                        'default': action.default,
                        'description': action.help
                    }
    return required_args, optional_args