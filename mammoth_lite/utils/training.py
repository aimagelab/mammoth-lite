import json
import os
import numpy as np
import math
from argparse import Namespace
from typing import Iterable, Optional
import sys
import torch
# Check if we're in a notebook environment
if 'ipykernel' not in sys.modules:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint
from utils.evaluate import evaluate


def train_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       args: Namespace,
                       epoch: int,
                       pbar: tqdm):
    """
    Trains the model for a single epoch.

    Args:
        model: the model to be trained
        train_loader: the data loader for the training set
        args: the arguments from the command line
        epoch: the current epoch
        pbar: the progress bar to update

    Returns:
        the number of iterations performed in the current epoch
    """
    for i, data in enumerate(train_loader):
        if args.debug_mode and i > 5:
            break
        
        inputs, labels, not_aug_inputs = data[0], data[1], data[2]
        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
        not_aug_inputs = not_aug_inputs.to(model.device)

        loss = model.observe(inputs, labels, not_aug_inputs, epoch=epoch)

        assert not math.isnan(loss)

        pbar.set_postfix({'loss': loss, 'lr': model.opt.param_groups[0]['lr']}, refresh=False)
        pbar.update()

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Optional[Namespace] = None) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution. If None, it will be loaded from the environment variable 'MAMMOTH_ARGS'.
    """
    if args is None:
        env_args = os.getenv('MAMMOTH_ARGS')
        if env_args is None:
            raise ValueError("No arguments provided. Did you run the `load_runner` function?")
        args = Namespace(**json.loads(env_args))
        os.environ['MAMMOTH_ARGS'] = json.dumps(vars(args))

    dataset.reset() # reset the dataset to the initial state

    model.net.to(model.device)
    torch.cuda.empty_cache()

    results: list[float] = []
    results_mask_classes: list[float] = []

    if args.loadcheck is not None:
        model, past_res = mammoth_load_checkpoint(args, model)

        if not args.disable_log and past_res is not None:
            results, results_mask_classes = past_res

        print('Checkpoint Loaded!')

    torch.cuda.empty_cache()
    try:
        for t in range(dataset.N_TASKS):
            model.net.train()
            train_loader, _ = dataset.meta_get_data_loaders()

            model.begin_task(dataset)

            with tqdm(train_loader, total=len(train_loader) * args.n_epochs, mininterval=0.1) as train_pbar:
                for epoch in range(args.n_epochs):
                    train_pbar.set_description(f"Task {t + 1} - Epoch {epoch + 1}")

                    model.begin_epoch(epoch, dataset)

                    train_epoch(model, train_loader, args, pbar=train_pbar, epoch=epoch)

                    model.end_epoch(epoch, dataset)

            model.end_task(dataset)

            accs, avg_loss = evaluate(model, dataset)

            mean_acc = np.mean(accs, axis=1)
            print(f'Accuracy for task {t + 1}\t[Class-IL]: {mean_acc[0]:.2f}% \t[Task-IL]: {mean_acc[1]:.2f}%')
            print(f'Average loss: {avg_loss:.6f}%')

            results.append(accs[0])
            results_mask_classes.append(accs[1])
            
            if args.savecheck:
                save_mammoth_checkpoint(t, dataset.N_TASKS, args, model,
                                        results=(results, results_mask_classes), # type: ignore
                                        optimizer_st=model.opt.state_dict())
    except KeyboardInterrupt:
        print("Training interrupted!")