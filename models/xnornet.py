"""This model is pytorch version implementation of the XNOR-Net.

   Reference: https://github.com/jiecaoyu/XNOR-Net-PyTorch

   Model description:
    xnor_alexnet and xnor_resnet18 models are supported for ImageNet dataset.

"""
import torch
import torch.nn as nn

from .binarized_modules import *

__all__ = ['xnor_alexnet', 'xnor_resnet18']

class Xnor_alexnet(nn.Module):
    def __init__(self, **kwargs):
        super(Xnor_alexnet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
           
            nn.BatchNorm2d(96),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            BinConv(96, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=5, stride=1, padding=2, padding_mode=kwargs['padding_mode']),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            BinConv(256, 384, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            
            nn.BatchNorm2d(384),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            BinConv(384, 384, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            
            nn.BatchNorm2d(384),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            BinConv(384, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            
            nn.BatchNorm1d(256 * 6 * 6),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            BinLinear(256 * 6 * 6, 4096, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=True),

            nn.BatchNorm1d(4096),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            BinLinear(4096, 4096, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=True),

            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, kwargs['num_classes']),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def xnor_alexnet(**kwargs):
    assert kwargs['dataset'] == 'imagenet', 'Currently xnor_alexnet model is for ImageNet dataset.'
    return Xnor_alexnet(**kwargs)




def conv3x3(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "3x3 convolution with padding"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=False)

def conv1x1(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "1x1 convolution"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=1, stride=stride, padding_mode=padding_mode, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.act = nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        self.conv1 = conv3x3(inplanes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], padding_mode=kwargs['padding_mode'], stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'], padding_mode=kwargs['padding_mode'], stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.act(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, **kwargs):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # this layer works for any size of input.
        self.fc = nn.Linear(512 * block.expansion, kwargs['num_classes']) ## assume that Last layer is FP


    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #nn.AvgPool2d(kernel_size=2, stride=2),
                conv1x1(self.inplanes, planes * block.expansion, 32, 0, True, 'zeros', stride),
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
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def xnor_resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    assert kwargs['dataset'] == 'imagenet', f"xnor_resnet18 model is supported for ImageNet dataset."

    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

