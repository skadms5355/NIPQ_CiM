"""This model is a light version of VGG-7 based on VGG-Net by Karen Simonyan and Andrew Zisserman.

    Reference:
        https://arxiv.org/abs/1409.1556

    Model description:
        The model has 4 convolutional layers and 3 fully-connected layers.
        Since the three fully-connected layers with 4096 neurons each in original VGG models are over-parameterized for
        CIFAR-10 dataset,
        we reduced the number of neurons in each layer to 512.
        The numbers of channels in convolutional layers are also reduced to 64, 64, 128, 128.

"""

import torch
import torch.nn as nn
from .binarized_modules import *

__all__ = ['tinyvgg7']


class Tinyvgg7(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super(Tinyvgg7, self).__init__()


        self.features = nn.Sequential(
            BinConv(3, 64, wbits=32, kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.BatchNorm2d(64),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            BinConv(64, 64, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinConv(64, 128, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1,padding_mode=kwargs['padding_mode'], bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinConv(128, 128, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])
        )
        self.classifier = nn.Sequential(
            BinLinear(128 * 4 * 4, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], bias=False),
            nn.BatchNorm1d(512),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinLinear(512, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], bias=False),
            nn.BatchNorm1d(512),
            nonlinear(abits=32, mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinLinear(512, num_classes, wbits=32, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def tinyvgg7(**kwargs):
    if kwargs['dataset'] == 'imagenet':
        assert False, 'We do not support tinyvgg7 architecture for ImageNet dataset'

    return Tinyvgg7(**kwargs)
