"""
This script is the main entry point for the Mammoth project. It contains the main function `main()` that orchestrates the training process.

The script performs the following tasks:
- Imports necessary modules and libraries.
- Parses command-line arguments.
- Initializes the dataset, model, and other components.
- Trains the model using the `train()` function.

To run the script, execute it directly or import it as a module and call the `main()` function.
"""

# needed (don't change it)
import numpy  # noqa

import os
import sys
import time
import socket
import datetime
import uuid
import argparse
import torch

torch.set_num_threads(2)

mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

print(f"Running Mammoth! on {socket.gethostname()}")


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def add_help(parser):
    """
    Add the help argument to the parser
    """
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')


def parse_args():
    """
    Parse command line arguments for the mammoth program and sets up the `args` object.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    from utils import create_if_not_exists
    from utils.args import add_initial_args, add_management_args, add_experiment_args

    from models import get_all_models, get_model_class

    parser = argparse.ArgumentParser(description='Mammoth - A benchmark Continual Learning framework for Pytorch', allow_abbrev=False, add_help=False)

    # 1) add arguments that include model, dataset, and backbone. These define the rest of the arguments.
    #   the backbone is optional as may be set by the dataset or the model. The dataset and model are required.
    add_initial_args(parser)
    args = parser.parse_known_args()[0]

    if args.backbone is None:
        print('No backbone specified. Using default backbone (set by the dataset).')

    add_help(parser)

    # 2) add the remaining arguments

    # - add the main Mammoth arguments
    add_management_args(parser)
    add_experiment_args(parser)

    # - add the model specific arguments
    model_class = get_model_class(args)
    model_class.get_parser(parser)

    # 3) Once all arguments are in the parser, we can parse
    args = parser.parse_args()

    # 4) final checks and updates to the arguments
    models_dict = get_all_models()
    args.model = models_dict[args.model]

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # Add the current git commit hash to the arguments if available
    try:
        import git
        repo = git.Repo(path=mammoth_path)
        args.conf_git_hash = repo.head.object.hexsha
    except Exception:
        print("ERROR: Could not retrieve git hash.")
        args.conf_git_hash = None

    if args.savecheck:
        if not os.path.isdir('checkpoints'):
            create_if_not_exists("checkpoints")

        now = time.strftime("%Y%m%d-%H%M%S")
        uid = args.conf_jobnum.split('-')[0]
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}{args.model}_{args.dataset}_{args.dataset_config}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}_{uid}"
        print(f"Saving checkpoint into: {args.ckpt_name}")

    # legacy print of the args, to make automatic parsing easier
    print(args)

    return args

def main(args=None):
    from utils.conf import get_device
    from models import get_model
    from datasets import get_dataset
    from utils.training import train
    from backbone import get_backbone

    lecun_fix()
    if args is None:
        args = parse_args()

    args.device = get_device() if args.device is None else args.device

    dataset = get_dataset(args)

    args.backbone = args.backbone if args.backbone is not None else dataset.get_backbone()
    args.num_classes = dataset.N_CLASSES
    
    backbone = get_backbone(args)
    print(f"Using backbone: {args.backbone}")

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform(), dataset=dataset)

    train(model, dataset, args)


if __name__ == '__main__':
    main()
