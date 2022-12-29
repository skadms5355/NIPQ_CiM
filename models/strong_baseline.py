"""This model is a implementation of the Strong Baseline model in Real-To-Binary paper.
    Reference:
        https://openreview.net/attachment?id=BJg4NgBKvH&name=original_pdf
    Model description:
        
        Each Resnet block is modified version which order is
        BatchNorm -> Binarization -> BinaryConv -> Activation.
        Also these blocks have double-skipping connections.

        All Downsample layers are real-values, and first and last layers are not
        binarized.

        XNOR++ style scaling factor is not implemented!! Standard scaling factor used (XNOR-Net).
"""

import torch
import torch.nn as nn
from .binarized_modules import *

__all__ = ['strongbaseline18', 'strongbaseline34']


def conv3x3(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "3x3 convolution with padding"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=False)

def conv1x1(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "1x1 convolution"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=1, stride=stride, padding_mode=padding_mode, bias=False)

class BasicBlock(nn.Module):
    """
        Resnet-like double skipping connection block.
        Args:
            inplanes (int): Number of channels in the input tensor.
            planes (int): Number of output channels.
            stride (int / tuple): Stride of the conv3x3.
            downsample: if the value is not None, the block uses downsample layer.
            **kwargs: input keyword arguments.
        Note:
            For the blocks with downsample layers, a channel size changes during passing the gating function.
            We modulate it with linear2.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binact = BinAct(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        self.conv1 = conv3x3(inplanes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], padding_mode=kwargs['padding_mode'], stride=stride)
        self.conv2 = conv3x3(planes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], padding_mode=kwargs['padding_mode'], stride=1)
        self.prelu1 = nn.PReLU(planes)
        self.prelu2 = nn.PReLU(planes)

        self.downsample = downsample

    def forward(self, x):
        identity1 = x
        out1 = self.bn1(x)
        out1 = self.binact(out1)
        out1 = self.conv1(out1)
        out1 = self.prelu1(out1)

        if self.downsample is not None:
            identity1 = self.downsample(x)

        out1 += identity1

        identity2 = out1
        out2 = self.bn2(out1)
        out2 = self.binact(out2)
        out2 = self.conv2(out2)
        out2 = self.prelu2(out2)

        out2 += identity2

        return out2


class StrongBaseline(nn.Module):
    """
        Constructs a StrongBaseline model.
        Note: As cifar100 dataset have 32x32 small images, kernel_size=7 for conv1 is too large.
              So we rather use kernel_size=3 conv, and skip maxpooling for cifar100.
              This structure gets reasonable accuracy reported in the paper.

    """

    def __init__(self, block, layers, **kwargs):

        super(StrongBaseline, self).__init__()

        self.inplanes = 64

        self.dataset = kwargs['dataset']
        if self.dataset == 'cifar100':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.PReLU(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)

        # self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        # self.relu2 = nn.PReLU(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        ## FP Last layer
        self.fc = nn.Linear(512 * block.expansion, kwargs['num_classes'])

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):

        """Make a stack of block using the specified parameters and blocks.

           Args:
                 block (Module): A block to stack.
                 planes (int): Number of output channel in every layer (except the downsampling layer).
                 blocks (int): Number of stacks to construct.
                 stride (int, optional): The stride used for downsampling layer. Default: 1
        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(self.inplanes, planes * block.expansion, 32, 0, False, 'zeros'),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.dataset == 'imagenet':
            x = self.maxpool(x)
        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.bn2(x)
        # x = self.relu2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def strongbaseline18(**kwargs):
    """Constructs 18-layer Strong baseline model."""

    dataset = kwargs['dataset']
    if dataset == 'cifar100' or dataset == 'imagenet':
        return StrongBaseline(BasicBlock, [2, 2, 2, 2], **kwargs)
    else:
        raise NotImplementedError


def strongbaseline34(**kwargs):
    """Constructs 34-layer Strong baseline model."""

    dataset = kwargs['dataset']
    if dataset == 'cifar100' or dataset == 'imagenet':
        return StrongBaseline(BasicBlock, [3, 4, 6, 3], **kwargs)
    else:
        raise NotImplementedError

