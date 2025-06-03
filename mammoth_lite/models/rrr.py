"""
This module is an implementation of Remembering for the Right Reasons
"""
from copy import deepcopy
import os
import numpy as np
import torch
from torchvision.datasets.folder import default_loader
from models import register_model
from models.utils.continual_model import ContinualModel
from utils.evaluate import evaluate

from models.rrr_utils import RAdam



class EvidenceSet(torch.utils.data.Dataset):

    def __init__(self, args, task_id, transforms):
        self.args = args
        sal_path = os.path.join('checkpoints', 'sal_{}.pth'.format(task_id))
        mem_path = os.path.join('checkpoints', 'mem_{}.pth'.format(task_id))
        pred_path = os.path.join('checkpoints', 'pred_{}.pth'.format(task_id))

        memory_set = torch.load(mem_path, weights_only=False)
        self.data = memory_set.data
        self.target = memory_set.targets
        self.saliencies =torch.load(sal_path, weights_only=False)
        self.predictions = torch.load(pred_path, weights_only=False)
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.target[idx]
        saliency, pred = self.saliencies[idx], self.predictions[idx]

        if not torch.is_tensor(img):
            img = default_loader(img)
            img = self.transform(img)

        return img, target, saliency, pred



@register_model('rrr')
class RRR(ContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = 'rrr'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser):
         # TODO: check defaults
        parser.add_argument('--lr_saliency', type=float, default=0.0005)
        parser.add_argument('--saliency_loss_type', type=str, choices=['l1','l2'], default='l1')
        parser.add_argument('--saliency_method', type=str, choices=['gc','smooth','bp','gbp','deconv'], default='gc')
        parser.add_argument('--saliency_reg', type=float, default=100)
        
        parser.add_argument('--fscil', type=int, choices=[0,1], default=0)
        parser.add_argument('--lr_multiplier', type=float, default=1)
        parser.add_argument('--lr_factor', type=float, required=True)
        parser.add_argument('--lr_patience', type=float, required=True)
        parser.add_argument('--train_schedule', type=int, nargs='+', default=[20,40,60])


        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(RRR, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.current_task = 0

        self.lrs = [self.args.lr for _ in range(self.n_tasks)]
        if self.args.fscil:
            self.lrs[1:] = [self.args.lr_multiplier * lr for lr in self.lrs[1:]]

        self.lrs_exp = [self.args.lr_saliency for _ in range(self.n_tasks)]
        self.lr_min=[lr/1000. for lr in self.lrs]
        self.lr_factor=self.args.lr_factor
        self.lr_patience=self.args.lr_patience
        self.in_size = dataset.SIZE[0]

        if self.args.saliency_loss_type == "l1":
            self.sal_loss = torch.nn.L1Loss().to(device=self.args.device.name)
        elif self.args.saliency_loss_type == "l2":
            self.sal_loss = torch.nn.MSELoss().to(device=self.args.device.name)
        else:
            raise NotImplementedError

        self.get_optimizer()
        self.get_optimizer_explanations()


        if self.args.saliency_method == 'gc':
            print ("Using GradCAM to obtain saliency maps")
            from models.rrr_utils.explanations import GradCAM as Explain
        elif self.args.saliency_method == 'smooth':
            print ("Using SmoothGrad to obtain saliency maps")
            from models.rrr_utils.explanations import SmoothGrad as Explain
        elif self.args.saliency_method == 'bp':
            print ("Using BackPropagation to obtain saliency maps")
            from models.rrr_utils.explanations import BackPropagation as Explain
        elif self.args.saliency_method == 'gbp':
            print ("Using Guided BackPropagation to obtain saliency maps")
            from models.rrr_utils.explanations import GuidedBackPropagation as Explain
        elif self.args.saliency_method == 'deconv':
            from models.rrr_utils.explanations import Deconvnet as Explain

        self.explainer = Explain(self.args)
        
    def get_optimizer(self, params=None, lr=None):
        params = params if params is not None else self.net.parameters()
        if lr is None: lr=self.lrs[self.current_task]

        self.opt = RAdam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0)

    def get_optimizer_explanations(self, lr=None):
        if lr is None: lr=self.lrs_exp[self.current_task]

        if self.args.opt=="sgd":
            self.optimizer_explanations = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=self.args.train.wd)
            self.scheduler_exp_opt = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_explanations,
                                                                         patience=self.lr_patience,
                                                                          factor=self.lr_factor/10,
                                                                          min_lr=self.lr_min[self.current_task], verbose=True)
        elif(self.args.opt=="adam"):
            self.optimizer_explanations= torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0.0, amsgrad=False)

        elif (self.args.opt=="radam"):
            self.optimizer_explanations = RAdam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)


    def end_task(self, dataset):
        # Restore best validation model
        if self.args.n_epochs > 1:
            self.net.load_state_dict(deepcopy(self.best_model))

        self.save_model(self.current_task, deepcopy(self.net.state_dict()))

        if self.current_task == 1:
            self.compute_memory()

        self.update_memory()

        self.current_task += 1

    def compute_memory(self):
        ssize = 1
        print ("***"*200)
        print ("saliency_size", self.saliency_size)
        for s in self.saliency_size:
            ssize *= s
        saliency_memory = 4 * self.args.buffer_size * ssize

        size = self.in_size
        image_size = 3 * size * size

        samples_memory = 4 * self.args.buffer_size * image_size
        count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        print('Num parameters in the entire model    = %s ' % (count))
        architecture_memory = 4 * count

        print("-------------------------->  Saliency memory size: (%sB)" % (saliency_memory))
        print("-------------------------->  Episodic memory size: (%sB)" % (samples_memory))
        print("------------------------------------------------------------------------------")
        print("                             TOTAL:  %sB" % (architecture_memory+samples_memory+saliency_memory))

    def update_memory(self):
        # Get memory set for each task seen so far with the updated samples per class and return new spc
        self.update_saliencies()

        evidence_set = []
        for t in range(self.current_task+1):
            evidence = EvidenceSet(self.args, t, self.transform)
            evidence_set.append(evidence)

        evidence_sets = torch.utils.data.ConcatDataset(evidence_set)

        self.saliency_loaders = torch.utils.data.DataLoader(evidence_sets,
                                                      batch_size=self.args.minibatch_size,
                                                      num_workers=self.args.num_workers,
                                                      shuffle=True)


    def begin_task(self, dataset):
        # early stopping setup
        self.best_loss=np.inf
        if self.args.n_epochs > 1:
            self.best_model=deepcopy(self.net.state_dict())
        self.lr = self.lrs[self.current_task]
        self.lr_exp = self.lrs_exp[self.current_task]
        self.patience=self.lr_patience
        self.best_acc = 0

        self.get_optimizer()
        self.get_optimizer_explanations()
    
    def begin_epoch(self, dataset):
        self.adjust_learning_rate()

        if self.current_task > 0:
            for idx, (inputs, _, sal, _) in enumerate(self.saliency_loaders):
                sal = sal.to(self.device)
                inputs = inputs.to(self.device)
                explanations, self.net , _, _ = self.explainer(inputs, self.net, self.current_task)

                self.saliency_size = explanations.size()

                # To make predicted explanations (Bx7x7) same as ground truth ones (Bx1x7x7)
                sal_loss = self.sal_loss(explanations.view_as(sal), sal)
                sal_loss *= self.args.saliency_reg

                self.optimizer_explanations.zero_grad()
                sal_loss.backward(retain_graph=True)
                self.optimizer_explanations.step()

    def end_epoch(self, dataset):
        # NOTE: validation on the test!
        # check the official code: pc_valid==0 for the given config
        accs, loss = evaluate(dataset.test_loaders[-1])
        mean_acc = np.mean(accs[0])
        
        if (self.args.opt == "sgd"):
            self.scheduler_opt.step(loss)
            self.scheduler_exp_opt.step(loss)

        if mean_acc > self.best_acc:
            self.best_model = deepcopy(self.net.state_dict())
        self.best_acc = max(mean_acc, self.best_acc)
        
    def adjust_learning_rate(self, epoch):
        if epoch in self.args.train.schedule:
            for param_group in self.opt.param_groups:
                param_group['lr'] *= self.args.train.gamma
            print("Reducing learning rate to ", param_group['lr'])


    def update_saliencies(self):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        # Generate saliency for images from the last seen task
        memory_path = os.path.join('checkpoints', 'mem_{}.pth'.format(self.current_task))
        memory_set = torch.load(memory_path, weights_only=False)
        num_samples = len(memory_set)
        single_loader = torch.utils.data.DataLoader(memory_set, batch_size=1, num_workers=0, shuffle=False)

        saliencies, predictions = [], []
        for idx, (img, y, tt) in enumerate(single_loader):

            img = img.to(self.device)
            sal, self.net, _, _ = self.explainer(img, self.net, self.current_task)
            output = self.net.forward(img)

            _, pred = output.max(1)

            saliencies.append(sal)
            predictions.append(pred)

        sal_path = os.path.join('checkpoints', 'sal_{}.pth'.format(self.current_task))
        pred_path = os.path.join('checkpoints', 'pred_{}.pth'.format(self.current_task))
        torch.save(saliencies, sal_path)
        torch.save(predictions, pred_path)

        # Reduce previous saliencies
        for t in range(self.current_task):
            # Read the stored saliency file
            sal_path = os.path.join('checkpoints', 'sal_{}.pth'.format(t))
            saliencies = torch.load(sal_path, weights_only=False)
            before = len(saliencies)

            pred_path = os.path.join('checkpoints', 'pred_{}.pth'.format(t))
            predictions = torch.load(pred_path, weights_only=False)

            # Extract the required number of samples and save them again
            saliencies = saliencies[:num_samples]
            after = len(saliencies)
            torch.save(saliencies, sal_path)
            print ("Reduced saliencies for task {} from {} to {}".format(t, before, after))

            predictions = predictions[:num_samples]
            torch.save(predictions, pred_path)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        output = self.net.forward(inputs)
        loss = self.loss(output, labels)

        # Backward
        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        self.opt.step()

        # NOTE: this should be balancoir instead of reservoir
        self.buffer.add_data(
            inputs=inputs,
            labels=labels,
            not_aug_inputs=not_aug_inputs
        )

        return loss.item()
