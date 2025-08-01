import sys
from typing import TYPE_CHECKING, Tuple
import torch
# Check if we're in a notebook environment
if 'ipykernel' not in sys.modules:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset


def mask_classes(outputs: torch.Tensor, dataset: 'ContinualDataset', k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    num_classes = dataset.N_CLASSES
    start_c, end_c = dataset.get_offsets(k)
    outputs[:, :start_c] = -float('inf')
    outputs[:, end_c:num_classes] = -float('inf')


@torch.no_grad()
def evaluate(model: 'ContinualModel', dataset: 'ContinualDataset', last_only=False) -> Tuple[list, list]:
    """
    Evaluates the single-class top-1 accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last_only: if True, only the last task is evaluated. Defaults to False.

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task.
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    tot_seen_samples = 0
    total_len = sum(len(x) for x in dataset.test_loaders) if hasattr(dataset.test_loaders[0], '__len__') else None
    loss = 0

    loss_fn = dataset.get_loss()
    pbar = tqdm(dataset.test_loaders, total=total_len, desc='Evaluating', leave=False, unit='batch')
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > 5:
                break
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            outputs = model(inputs)
            loss += loss_fn(outputs, labels).item()

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1
            pbar.set_postfix({f'acc_task_{k + 1}': max(0, correct / total * 100)}, refresh=False)
            pbar.set_description(f"Evaluating Task {k + 1}", refresh=False)
            pbar.update(1)

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        tot_seen_samples += int(total)

        if correct > correct_mask_classes and dataset.SETTING == 'class-il':
            print("WARNING: Task-IL accuracy is LOWER than Class-IL accuracy. "
                            "This should NEVER happen and probably means there is a bug somewhere. "
                            "Hint: check if the dataloader returns the targets in the correct order.")
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    pbar.close()

    model.net.train(status)
    return (accs, accs_mask_classes), loss/i
