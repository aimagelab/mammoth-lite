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

def load_runner(model: str, dataset: str, args: dict[str, Any]) -> Tuple['ContinualModel', 'ContinualDataset']:
    """
    Load the model and dataset for the given experiment.
    Args:
        model_name (str): The name of the model to be used.
        dataset_name (str): The name of the dataset to be used.
        args (dict): A dictionary containing the arguments for the experiment.

    Returns:
        Tuple[ContinualModel, ContinualDataset]: A tuple containing the model and dataset.
    """
    assert 'lr' in args, "Learning rate not specified in experiment arguments. Add 'lr' to the experiment arguments."
    assert 'batch_size' in args, "Batch size not specified in experiment arguments. Add 'batch_size' to the experiment arguments."
    assert 'n_epochs' in args, "Number of epochs not specified in experiment arguments. Add 'n_epochs' to the experiment arguments."

    exp_args = initalize_args(model, dataset, args)
    mammoth_dataset = get_dataset(exp_args)
    exp_args.backbone = exp_args.backbone if exp_args.backbone else mammoth_dataset.get_backbone()
    exp_args.num_classes = mammoth_dataset.N_CLASSES
    backbone = get_backbone(exp_args)
    mammoth_model = get_model(exp_args, backbone, mammoth_dataset.get_loss(), mammoth_dataset.get_transform(), mammoth_dataset)
    
    os.environ['MAMMOTH_ARGS'] = json.dumps(vars(exp_args))

    return mammoth_model, mammoth_dataset
    

def initalize_args(model_name: str, dataset_name: str, args: dict[str, Any]) -> Namespace:
    parser = ArgumentParser(allow_abbrev=False, add_help=False)

    add_initial_args(parser)
    add_management_args(parser)
    add_experiment_args(parser)

    if 'buffer_size' in args:
        add_rehearsal_args(parser)

    exp_str = [f"--{k}={v}" for k, v in args.items() if k not in ['model', 'dataset', 'backbone'] and v is not None]
    exp_str += ['--dataset', dataset_name, '--model', model_name]
    if 'backbone' in args:
        exp_str += ['--backbone', args['backbone']]

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