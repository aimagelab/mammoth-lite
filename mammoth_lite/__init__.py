import sys
import os

mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

os.environ["MAMMOTH_BASE_PATH"] = mammoth_path

from models import get_model_names, register_model, ContinualModel
from datasets import get_dataset_names, register_dataset, ContinualDataset
from datasets.utils.continual_dataset import MammothDataset
from backbone import get_backbone_names, register_backbone, MammothBackbone, ReturnTypes
from utils.utils import load_runner, get_avail_args
from utils.training import train
from utils.conf import base_path, get_device

# API:
# - get all datasets, backbones, models by name
# - initialize a learning scenario
# - load the dataset and model for the scenario

__all__ = [
    "get_dataset_names",
    "get_model_names",
    "get_backbone_names",
    "load_runner",
    "get_avail_args",
    "train",
    "register_model",
    "register_dataset",
    "register_backbone",
    "ContinualModel",
    "ContinualDataset",
    "MammothBackbone",
    "MammothDataset",
    "base_path",
    "get_device",
    "ReturnTypes",
]