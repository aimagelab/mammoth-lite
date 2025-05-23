from argparse import Namespace, ArgumentParser
from typing import Any, Tuple, TYPE_CHECKING

from .args import add_initial_args, add_management_args, add_experiment_args, add_rehearsal_args

from models import get_model
from datasets import get_dataset
from backbone import get_backbone

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset

def load_runner(model_name: str, dataset_name: str, experiment_args: dict[str, Any]) -> Tuple[Namespace, 'ContinualModel', 'ContinualDataset']:
    assert 'lr' in experiment_args, "Learning rate not specified in experiment arguments. Add 'lr' to the experiment arguments."
    assert 'batch_size' in experiment_args, "Batch size not specified in experiment arguments. Add 'batch_size' to the experiment arguments."
    assert 'n_epochs' in experiment_args, "Number of epochs not specified in experiment arguments. Add 'n_epochs' to the experiment arguments."

    args = initalize_args(model_name, dataset_name, experiment_args)
    dataset = get_dataset(args)
    args.backbone = args.backbone if args.backbone else dataset.get_backbone()
    args.num_classes = dataset.N_CLASSES
    backbone = get_backbone(args)
    model = get_model(args, backbone, dataset.get_loss(), dataset.get_transform(), dataset)
    
    return args, model, dataset
    

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

def get_avail_args():
    """
    Returns the available arguments for the model and dataset.
    """
    parser = ArgumentParser(allow_abbrev=False, add_help=False)

    add_initial_args(parser)
    add_management_args(parser)
    add_experiment_args(parser)

    return parser.format_usage()