import torch
from torch.nn import functional as F


class _BaseWrapper(object):
    def __init__(self, model, args, device):
        super(_BaseWrapper, self).__init__()
        self.device = device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        # self.image_shape = image.shape[2:]
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        # self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """
    def __init__(self, args, device):
        # model, candidate_layers=None):
        self.model = None # model is set when module is called
        super(GradCAM, self).__init__(self.model, args, device)
        self.args = args

        self.fmap_pool = {}
        self.grad_pool = {}

        self.target_layer = args.target_layer


    def __call__(self, inputs, model, task_id):
        self.model = model
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output#.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0]#.detach()
            return backward_hook
        
        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.target_layer is None or name == self.target_layer:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

        probs, ids = self.forward(inputs)
        ids_ = torch.LongTensor([ids[:, 0].tolist()]).T.to(device=self.device)
        self.backward(ids_)
        gradcams, fmaps = self.generate(target_layer=self.target_layer)
        gradcams = gradcams.squeeze(1) # shape: [B, H, W]
        return gradcams, self.model, probs[:,0], ids[:,0]


    def forward(self, image):
        self.image_shape = image.shape[2:]

        return super(GradCAM, self).forward(image)

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        return gcam, fmaps

