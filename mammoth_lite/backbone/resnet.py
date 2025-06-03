from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from torch.nn.functional import avg_pool2d, relu
from torch import Tensor
import torchvision
from torchvision.models import ResNet18_Weights

from backbone import MammothBackbone, register_backbone


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        self.return_prerelu = False
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # type: ignore
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width) # type: ignore
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width) # type: ignore
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion) # type: ignore
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.

        Args:
            in_planes: the number of input channels
            planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.return_prerelu = False
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)
        return out


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: Union[Type[BasicBlock], Type[Bottleneck]], num_blocks: List[int],
                 num_classes: int, nf: int, initial_conv_k=3) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
            initial_conv_k: the kernel size of the initial convolution
        """
        super(ResNet, self).__init__()
        self.return_prerelu = False
        self.device = "cpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        if initial_conv_k != 3:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, nf * 1 * block.expansion, kernel_size=initial_conv_k, stride=2, padding=3, bias=False)
        else:
            self.conv1 = conv3x3(3, nf * 1 * block.expansion)
        self.bn1 = nn.BatchNorm2d(nf * 1 * block.expansion)
        self.feature_dim = nf * 8 * block.expansion
        self.layer1 = self._make_layer(block, nf * 1 * block.expansion, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2 * block.expansion, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4 * block.expansion, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.feature_dim, num_blocks[3], stride=2)

        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.pool_fn = avg_pool2d

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable

    def _make_layer(self, block: Union[Type[BasicBlock], Type[Bottleneck]], planes: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out'):
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = self.pool_fn(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'full'] but got {}".format(returnt))


@register_backbone("resnet18")
def resnet18(num_classes: int, num_filters: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_filters)

@register_backbone("resnet18_7x7_pt")
def resnet18_7x7(num_classes: int, num_filters: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        ResNet network
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_filters, initial_conv_k=7)
    # load pretrained weights from torchvision
    st = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    for k in list(st.keys()):
        if 'downsample' in k:
            st[k.replace('downsample.', 'shortcut.')] = st.pop(k)

    missing, unexpected = model.load_state_dict(st, strict=False)
    assert len([m for m in missing if 'classifier' not in m]) == 0, \
        "Some weights are missing in the pretrained model: {}".format(missing)
    assert len([u for u in unexpected if 'fc' not in u]) == 0, \
        "Some unexpected weights in the pretrained model: {}".format(unexpected)
    return model

@register_backbone("resnet50")
def resnet50(num_classes: int, num_filters: int = 64) -> ResNet:
    """
    Instantiates a ResNet50 network.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_filters)
