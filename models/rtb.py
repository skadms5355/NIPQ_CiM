"""This model is a implementation of the Real-To-Binary paper.
    Reference:
        https://openreview.net/attachment?id=BJg4NgBKvH&name=original_pdf
    Model description:
        Real-To-Binary model is based on ResNet-18, which has a convolution layer,
        followed by 4 basic block structures, and fully-connected layer.

        Each Resnet block is modified version which order is
        BatchNorm -> Binarization -> BinaryConv -> Activation.
        Also these blocks have double-skipping connections.

        All Downsample layers are real-values, and first and last layers are not
        binarized.
"""

import torch
import torch.nn as nn
from .binarized_modules import *

__all__ = ['resnet18teacher', 'resnet34teacher', 'rtb18fp', 'rtb34fp', 'rtb18ban', 'rtb34ban', 'rtb18bnn', 'rtb34bnn']


def conv3x3(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "3x3 convolution with padding"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=False)

def conv1x1(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "1x1 convolution"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=1, stride=stride, padding_mode=padding_mode, bias=False)

class BasicBlockRtB(nn.Module):
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
        super(BasicBlockRtB, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.binact1 = BinAct(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        self.binact2 = BinAct(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        self.conv1 = conv3x3(inplanes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs ['weight_scale'], padding_mode=kwargs['padding_mode'], stride=stride)
        self.conv2 = conv3x3(planes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], padding_mode=kwargs['padding_mode'], stride=1)
        self.prelu1 = nn.PReLU(planes)
        self.prelu2 = nn.PReLU(planes)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.r = 8
        self.linear1 = nn.Linear(inplanes, int(inplanes/self.r))
        self.linear_1 = nn.Linear(planes, int(planes/self.r))
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(int(inplanes/self.r), planes)
        self.linear_2 = nn.Linear(int(planes/self.r), planes)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        self.downsample = downsample

    def forward(self, x):
        identity1 = x
        out1 = self.bn1(x)

        G1 = out1
        G1 = self.avgpool1(G1)
        G1 = G1.reshape(G1.size(0), -1)
        G1 = self.linear1(G1)
        G1 = self.relu1(G1)
        G1 = self.linear2(G1)
        G1 = self.sigmoid1(G1)
        G1 = G1.unsqueeze(2)
        G1 = G1.unsqueeze(3)

        out1 = self.binact1(out1)
        out1 = self.conv1(out1)

        out1 *= G1
        out1 = self.prelu1(out1)

        if self.downsample is not None:
            identity1 = self.downsample(x)

        out1 += identity1
        identity2 = out1

        out2 = self.bn2(out1)
        G2 = out2
        G2 = self.avgpool2(G2)
        G2 = G2.reshape(G2.size(0), -1)
        G2 = self.linear_1(G2)
        G2 = self.relu2(G2)
        G2 = self.linear_2(G2)
        G2 = self.sigmoid2(G2)
        G2 = G2.unsqueeze(2)
        G2 = G2.unsqueeze(3)

        out2 = self.binact2(out2)
        out2 = self.conv2(out2)

        out2 *= G2
        out2 = self.prelu2(out2)

        out2 += identity2

        return out2

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, 32, 0, False, padding_mode='zeros', stride=stride)
        self.conv2 = conv3x3(planes, planes, 32, 0, False, padding_mode='zeros', stride=1)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class RtB(nn.Module):
    """
        Constructs a Real-To-Binary(RtB) model.
        Note: As cifar100 dataset have 32x32 small images, kernel_size=7 for conv1 is too large.
              So we rather use kernel_size=3 conv, and skip maxpooling for cifar100.
              This structure gets reasonable accuracy reported in the paper.

    """

    def __init__(self, block, layers, **kwargs):

        super(RtB, self).__init__()
        self.transfer = True if (kwargs['transfer_mode'] >= 2) else False
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

        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.relu2 = nn.PReLU(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        ## FP Last layer
        self.fc = nn.Linear(512 * block.expansion, kwargs['num_classes'])


    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):

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
        if self.dataset == 'imagenet':
            x = self.maxpool(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # as author informed, 4 transfer points each at the end of a stage are used.
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        x = self.bn2(f4)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if not self.transfer:
            return x
        else:
            return x, [f1, f2, f3, f4]


class ResNet(nn.Module):
    def __init__(self, block, layers, **kwargs):

        super(ResNet, self).__init__()
        self.transfer = True if (kwargs['transfer_mode'] >= 2) else False
        self.inplanes = 64
        self.dataset = kwargs['dataset']

        if self.dataset == 'cifar100':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, kwargs['num_classes'])

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):

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
        if self.dataset == 'imagenet':
            x = self.maxpool(x)
        x = self.bn1(x)
        x = self.relu1(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        x = self.bn2(f4)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if not self.transfer:
            return x
        else:
            return x, [f1, f2, f3, f4]


def rtb18fp(**kwargs):
    """Constructs 18-layer RtB model with full precision activation and weight."""
    dataset = kwargs['dataset']
    kwargs['wbits'] = 32
    kwargs['abits'] = 32
    if dataset == 'cifar100' or dataset == 'imagenet':
        return RtB(BasicBlockRtB, [2, 2, 2, 2], **kwargs)
    else:
        raise NotImplementedError

def rtb34fp(**kwargs):
    """Constructs 34-layer RtB model with full precision activation and weight."""

    dataset = kwargs['dataset']
    kwargs['wbits'] = 32
    kwargs['abits'] = 32
    if dataset == 'cifar100' or dataset == 'imagenet':
        return RtB(BasicBlockRtB, [3, 4, 6, 3], **kwargs)
    else:
        raise NotImplementedError

def rtb18ban(**kwargs):
    """Constructs 18-layer RtB model with binary activation and full precision weight."""

    dataset = kwargs['dataset']
    kwargs['wbits'] = 32
    kwargs['abits'] = 1
    if dataset == 'cifar100' or dataset == 'imagenet':
        return RtB(BasicBlockRtB, [2, 2, 2, 2], **kwargs)
    else:
        raise NotImplementedError

def rtb34ban(**kwargs):
    """Constructs 34-layer RtB model with binary activation and full precision weight."""

    dataset = kwargs['dataset']
    kwargs['wbits'] = 32
    kwargs['abits'] = 1
    if dataset == 'cifar100' or dataset == 'imagenet':
        return RtB(BasicBlockRtB, [3, 4, 6, 3], **kwargs)
    else:
        raise NotImplementedError

def rtb18bnn(**kwargs):
    """Constructs 18-layer RtB model with binary activation and weight."""

    dataset = kwargs['dataset']
    kwargs['wbits'] = 1
    kwargs['abits'] = 1
    if dataset == 'cifar100' or dataset == 'imagenet':
        return RtB(BasicBlockRtB, [2, 2, 2, 2], **kwargs)
    else:
        raise NotImplementedError

def rtb34bnn(**kwargs):
    """Constructs 34-layer RtB model with binary activation and weight."""

    dataset = kwargs['dataset']
    kwargs['wbits'] = 1
    kwargs['abits'] = 1
    if dataset == 'cifar100' or dataset == 'imagenet':
        return RtB(BasicBlockRtB, [3, 4, 6, 3], **kwargs)
    else:
        raise NotImplementedError


def resnet18teacher(**kwargs):
    """Constructs 18-layer ResNet model with full precision activation and weight."""

    dataset = kwargs['dataset']
    kwargs['wbits'] = 32
    kwargs['abits'] = 32
    if dataset == 'cifar100' or dataset == 'imagenet':
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    else:
        raise NotImplementedError


def resnet34teacher(**kwargs):
    """Constructs 34-layer ResNet model with full precision activation and weight."""

    dataset = kwargs['dataset']
    kwargs['wbits'] = 32
    kwargs['abits'] = 32
    if dataset == 'cifar100' or dataset == 'imagenet':
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    else:
        raise NotImplementedError

