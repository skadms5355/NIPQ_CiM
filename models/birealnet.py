"""This model contains modified 18, 34-layer BiRealNet for Imagenet dataset introduced in BiRealNet.

    Reference:
        https://arxiv.org/abs/1808.00278

    Model description:
        The BiRealNet-18 has a convolutional layer with kernel size of 7, followed by 16 block-
        structures(pooling done every 4 layer), and finalized by a single fully-connected layer.
        The BiRealNet-34 has the same architecture except that its 32 blocks are pooled every
        6, 8, 12, 6 layer.

        Every block is a 1-conv-layer-per-block structure and there is an identity shortcut
        connecting the real activations (before the sign fuction) to the activations of the
        consecutive block.

        The first and the last layer is not binarized.
        Note that this model has a ReLU before the avg-pooling layer after the 4 stacks of
        BasicBlocks.

"""

import torch
import torch.nn as nn
from .binarized_modules import *

__all__ = ['birealnet18', 'birealnet34']


class BasicBlock(nn.Module):
    """A 1-conv-layer-per-block structure containing an identity shortcut.

    This class contains the information on how a single block is strucured. The input tensor goes
    through a binary activation, binary convolution, and a batchnormalization. It is then added with
    the original input, forming an shortcut between the real activation of consecutive blocks.

    Args:
        inplanes (int): Number of channels in the input tensor.
        planes (int): Number of output channels.
        stride (int or tuple, optional): Stride of the BinConv. Default: 1
        downsample: if specified, the tensor goes through the downsampling layer before being added
        in a identity shortcut.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        wbits: Bit resolution of weights. 1 or 32.
        abits: Bit resolution of activations. 1 or 32.
        ste: An option that decides the gradients in the backward pass.
        mode: An option that decides how the activations are represented in binarized mode.
            'signed' uses -1, 1 activations and 'unsigned' uses 0, 1 activations.
        padding_mode: padding mode to use e.g. zeros, ones, alter.
        weight_clip: An optional value to use for weight clipping. Default: 0

    """

    expansion = 1
    """(int) A parameter in order to increase the out_channels if needed. """

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.downsample = downsample

        self.activation = nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        self.binary_conv = BinConv(inplanes, planes, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=stride, padding=1, padding_mode=kwargs['padding_mode'], bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        """A forward function for the BasicBlock """

        identity = x

        out = self.activation(x)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out



class BiRealNet(nn.Module):
    """Constructs a modified BiRealNet with specified parameters.

        After the 4 stacks of Basic blocks, there is an added nonlinear layer before
        pooling.

    """

    def __init__(self, block, layers, **kwargs):
        super(BiRealNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)
        self.activation = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, kwargs['num_classes'])


    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        """Make a stack of block using the specified parameters and blocks.

        Args:
            block (Module): A block to stack.
            planes (int): Number of output channel in every layer (except the downsampling layer).
            blocks (int): Number of stacks to construct.
            stride (int, optional): The stride used for downsampling layer. Default: 1

        Note:
            There is a slight difference in downsampling layer compared to the ResNet model. Instead
            of having a convolution of stride 2, this implementation has an average pooling of
            kernel_size 2.

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                # BinConv(self.inplanes, planes * block.expansion, kwargs['wbits'], kwargs['weight_clip'],
                         # kernel_size=1, stride=1, padding_mode=kwargs['padding_mode'], bias=False),
                BinConv(self.inplanes, planes * block.expansion, wbits=32,
                         kernel_size=1, stride=1, padding_mode=kwargs['padding_mode'], bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.activation(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def birealnet18(**kwargs):
    """Constructs a BiRealNet-18 model. """
    assert kwargs['dataset'] == 'imagenet', f"BiRealNet only supports ImageNet dataset."
    return BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)


def birealnet34(**kwargs):
    """Constructs a BiRealNet-34 model. """
    assert kwargs['dataset'] == 'imagenet', f"BiRealNet only supports ImageNet dataset."
    return BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
