
from argparse import Namespace
from collections.abc import Iterable
import logging
from typing import Dict, List, Tuple, Union, TYPE_CHECKING
import numpy as np
import torch
import os

if TYPE_CHECKING:
    from models import ContinualModel

def to_parsable_obj(r: Union[Dict, Namespace, list, torch.Tensor, np.ndarray]) -> Union[Dict, list, str, int, float, bool]:
    """
    Convert a non-builtin object to a parsable (and loadable with `weights_only=True`) object.
    Looking at you, Namespace.
    """

    if isinstance(r, Namespace):
        return to_parsable_obj(vars(r))
    if isinstance(r, list):
        return [to_parsable_obj(x) for x in r]
    if isinstance(r, dict):
        return {k: to_parsable_obj(v) for k, v in r.items()}
    else:
        if isinstance(r, torch.Tensor):
            r = r.detach().cpu().numpy().tolist()
        elif isinstance(r, np.ndarray):
            r = r.tolist()
        if not isinstance(r, str) and isinstance(r, Iterable) and len(r) > 1:
            return [to_parsable_obj(x) for x in r]
        # check if type of r is builtin
        if isinstance(r, (int, float, str, bool)):
            try:
                r = r.item()  # could be numpy scalar
            except BaseException:
                return r
        raise ValueError(f"Cannot convert {type(r)} to parsable object")

def _load_mammoth_model(dict_keys, model: torch.nn.Module, args):
    for k in list(dict_keys):
        if args.distributed != 'dp':
            dict_keys[k.replace('module.', '')] = dict_keys.pop(k)
        elif 'module' not in k:
            dict_keys[k.replace('net.', 'net.module.')] = dict_keys.pop(k)

    for k in list(dict_keys):
        if '_features' in dict_keys:
            dict_keys.pop(k)

    if 'lucir' in args.model.lower():
        model.register_buffer('classes_so_far', torch.zeros_like(
            dict_keys['classes_so_far']).to('cpu'))

    model.load_state_dict(dict_keys)
    model.net.to(model.device)
    return model

def mammoth_load_checkpoint(args, model: 'ContinualModel') -> Tuple['ContinualModel', Tuple[List[float], List[float]]]:
    """
    Loads the keys from the given checkpoint.
    - Handles DataParallel and DistributedDataParallel checkpoints.
    - Handles checkpoints from previous versions of the code.
    - Handles head initialization for LUCIR.

    Args:
        args: the model with the checkpoint loaded.
        model: the model to be loaded.
        ignore_classifier: whether to ignore the classifier weights.

    Returns:
        the model with the checkpoint loaded.
    """
    if not os.path.exists(args.loadcheck):
        raise ValueError('The given checkpoint does not exist.')

    saved_obj = torch.load(args.loadcheck, map_location=torch.device("cpu"), weights_only=True)

    saved_obj['args'] = Namespace(**saved_obj['args']) # convert back to Namespace

    # Mammoth checkpoint
    model = _load_mammoth_model(saved_obj['model'], model, args)
    if 'buffer' in saved_obj:
        loading_model = saved_obj['args'].model
        if args.model != loading_model:
            print(f'WARNING: The loaded model was trained with a different model: {loading_model}')
        model.load_buffer(saved_obj['buffer'])

    return model, saved_obj['results']

def save_mammoth_checkpoint(task: int, end_task: int, args: Namespace, model: 'ContinualModel', results: Tuple[List[float], List[float]],
                            optimizer_st: Dict[str, torch.Tensor]):
    """
    Save a checkpoint for the model for the given task.
    Handles saving as a single file (will require `weights_only=False)` or separate weights (can be loaded safely with `weights_only=True`).
    """
    if args.savecheck == 'task':
        checkpoint_name = f'checkpoints/{args.ckpt_name}_joint' if args.joint else f'checkpoints/{args.ckpt_name}_{task}'
    elif args.savecheck == 'last':
        if task == end_task - 1:
            checkpoint_name = f'checkpoints/{args.ckpt_name}_joint' if args.joint else f'checkpoints/{args.ckpt_name}_last'
        else:
            return
    else:
        raise ValueError(f'Invalid savecheck mode: {args.savecheck}')

    save_obj = {
        'model': model.state_dict(),
        'optimizer': optimizer_st,
        'args': to_parsable_obj(vars(args)),  # avoid Namespace and other non-builtin types
        'results': results,  # avoid numpy, torch, and non-builtin types
    }
    if 'buffer_size' in vars(args) and hasattr(model, 'buffer'):
        save_obj['buffer'] = model.buffer.serialize() # type: ignore

    torch.save(save_obj, checkpoint_name + '.pt')
    print(f"Checkpoint for task {task} saved at {checkpoint_name}")