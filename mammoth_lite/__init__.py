import sys
import os

mammoth_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mammoth_path)

os.environ["MAMMOTH_BASE_PATH"] = mammoth_path

from models import get_model_names
from datasets import get_dataset_names
from backbone import get_backbone_names
from utils.utils import load_runner, get_avail_args
from utils.training import train

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
    "train"
]