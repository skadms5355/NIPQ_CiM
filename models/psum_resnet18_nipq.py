""" This is ResNet model defined in torchvision with psum quantization operation.

We alter the code slightly to match our script, but there are no modifications
model-wise.
"""

import torch
import torch.nn as nn
from .nipq_quantization_module import QuantOps as Q
from .nipq_hwnoise_psum_module import PsumQuantOps as PQ

__all__ = ['psum_resnet18_nipq']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PQ.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                      bias=False, act_func=Q.ReLU())

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x 

        out = self.relu(x) # floating point short current with no activation function (higer accruacy)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class PNIPQ_ResNet(nn.Module):
    def __init__(self, block, layers, **kwargs):

        super(PNIPQ_ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # this layer works for any size of input.
        self.fc = nn.Linear(512 * block.expansion, 1000) ## assume that Last layer is FP


    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        #print(downsample)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.bn2(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PNIPQ_ResNet_cifar10(nn.Module):
    def __init__(self, block, layers, **kwargs):

        super(PNIPQ_ResNet_cifar10, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)

        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # this layer works for any size of input.
        self.fc = nn.Linear(512 * block.expansion, 10) ## assume that Last layer is FP


    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(self.inplanes, planes * block.expansion, 32, 0, False, 'zeros', stride=1),
                nn.BatchNorm2d(planes * block.expansion),
            )
            #if kwargs['downsample'] == 'avgpool':
            #    downsample = nn.Sequential(
            #        nn.AvgPool2d(kernel_size=2, stride=2),
            #        conv1x1(self.inplanes, planes * block.expansion, 32, 0, False, 'zeros', stride=1),
            #        nn.BatchNorm2d(planes * block.expansion),
            #    )
            #elif kwargs['downsample'] == 'maxpool':
            #    downsample = nn.Sequential(
            #        nn.MaxPool2d(kernel_size=2, stride=2),
            #        conv1x1(self.inplanes, planes * block.expansion, 32, 0, False, 'zeros', stride=1),
            #        nn.BatchNorm2d(planes * block.expansion),
            #    )
            #elif kwargs['downsample'] == 'stride':
            #    downsample = nn.Sequential(
            #        conv1x1(self.inplanes, planes * block.expansion, 32, 0, False, 'zeros', stride=2),
            #        nn.BatchNorm2d(planes * block.expansion),
            #    )
            #else:
            #    downsample = nn.Sequential(
            #        nn.AvgPool2d(kernel_size=2, stride=2),
            #        conv1x1(self.inplanes, planes * block.expansion, 32, 0, False, 'zeros', stride=1),
            #        nn.BatchNorm2d(planes * block.expansion),
            #    )
                #assert False, 'ResNet does not provide downsample {}'.format(kwargs['downsample'])


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def psum_resnet18_nipq(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    dataset = kwargs['dataset']
    if dataset == 'imagenet':
        return PNIPQ_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif dataset == 'cifar10':
        return PNIPQ_ResNet_cifar10(BasicBlock, [2, 2, 2, 2], **kwargs)
    else:
        assert False, 'resnet18 does not support dataset {}'.format(dataset)