"""This model contains ReActNet-A and ReAct-BiRealNet-18 for ImageNet dataset.

    Reference:
        https://arxiv.org/abs/2003.03488

    Model description:
            reactnet: ReActNet-A model described in the paper.
            reactresnet18: ReAct-BiRealNet-18 model described in the paper.
"""

import torch
import torch.nn as nn
from .binarized_modules import *

__all__ = ['reactnet', 'reactresnet18']

def conv3x3(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "3x3 convolution with padding"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=False)

def conv1x1(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "1x1 convolution"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=1, stride=stride, padding_mode=padding_mode, bias=False)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class BasicBlock(nn.Module):
    """
        ResNet-variant of ReActNet.
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binact = BinAct(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        self.binary3x3 = conv3x3(inplanes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], kwargs['padding_mode'], stride)
        self.bn = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binact(out)
        out = self.binary3x3(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class BasicBlockReAct(nn.Module):
    """
        MobileNet-like double skipping connection block.
        Args:
            inplanes (int): Number of channels in the input tensor.
            planes (int): Number of output channels.
            stride (int / tuple): Stride of the conv3x3.
            **kwargs: input keyword arguments.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(BasicBlockReAct, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        self.move11 = LearnableBias(inplanes)
        self.binact1 = BinAct(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        self.binary3x3 = conv3x3(inplanes, inplanes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], kwargs['padding_mode'], stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        if inplanes != planes:
            self.pooling = nn.AvgPool2d(2,2)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)
        self.binact2 = BinAct(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])

        if inplanes == planes:
            self.binary1x1 = conv1x1(inplanes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], kwargs['padding_mode'])
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.binary1x1_down1 = conv1x1(inplanes, inplanes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], kwargs['padding_mode'])
            self.binary1x1_down2 = conv1x1(inplanes, inplanes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], kwargs['padding_mode'])
            self.bn2_1 = nn.BatchNorm2d(inplanes)
            self.bn2_2 = nn.BatchNorm2d(inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

 
    def forward(self, x):
        out1 = self.move11(x)

        out1 = self.binact1(out1)
        out1 = self.binary3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binact2(out2)

        if self.inplanes == self.planes:
            out2 = self.binary1x1(out2)
            out2 = self.bn2(out2)
            out2 += out1
        else:
            assert self.planes == self.inplanes * 2, "The number of output channel should be twice the number of input channel."

            out2_1 = self.binary1x1_down1(out2)
            out2_2 = self.binary1x1_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

class ReActNet_BiRealNet(nn.Module):
    """
        Constructs a ReActNet based BiRealNet.
    """

    def __init__(self, block, layers, **kwargs):
        super(ReActNet_BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, kwargs['num_classes'])

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion, 32, kwargs['weight_clip'], kwargs['weight_scale'], kwargs['padding_mode'], stride=1),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ReActNet_MobileNet(nn.Module):
    """
        Constructs a ReActNet model.
    """

    def __init__(self, **kwargs):
        super(ReActNet_MobileNet, self).__init__()
        self.inplanes = 32

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        layers = []
        layers.append(BasicBlockReAct(self.inplanes, 64, 1, **kwargs))
        layers.append(BasicBlockReAct(64, 128, 2, **kwargs))
        layers.append(BasicBlockReAct(128, 128, 1, **kwargs))
        layers.append(BasicBlockReAct(128, 256, 2, **kwargs))
        layers.append(BasicBlockReAct(256, 256, 1, **kwargs))
        layers.append(BasicBlockReAct(256, 512, 2, **kwargs))
        layers.append(BasicBlockReAct(512, 512, 1, **kwargs))
        layers.append(BasicBlockReAct(512, 512, 1, **kwargs))
        layers.append(BasicBlockReAct(512, 512, 1, **kwargs))
        layers.append(BasicBlockReAct(512, 512, 1, **kwargs))
        layers.append(BasicBlockReAct(512, 512, 1, **kwargs))
        layers.append(BasicBlockReAct(512, 1024, 2, **kwargs))
        layers.append(BasicBlockReAct(1024, 1024, 1, **kwargs))

        self.body = nn.Sequential(*layers)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, kwargs['num_classes'])

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.body(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def reactnet(**kwargs):
    """Construct MobileNetV1-based ReActNet-A."""
    
    return ReActNet_MobileNet(**kwargs)



def reactresnet18(**kwargs):
    """Construct ResNet18-based ReActNet."""

    return ReActNet_BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)


