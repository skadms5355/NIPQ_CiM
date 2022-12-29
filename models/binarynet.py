"""This model is pytorch version implementation of the BinaryNet.

    Reference:
        - "Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" 
        - https://arxiv.org/abs/1602.02830

    Model description:
        - binarynet_mlp is a 4-layer MLP model for MNIST dataset.
        - binarynet_vgg is a VGG-9-like model for CIFAR-10 dataset.
        - binarynet_alexnet is for ImageNet dataset.

        - Unlike other models, binarynet binarized all the weights including the first and the last layers.

"""
import torch
import torch.nn as nn
from .binarized_modules import *

__all__ = ['binarynet_mlp', 'binarynet_vgg', 'binarynet_alexnet']


class Binarynet_MLP(nn.Module):
    def __init__(self, **kwargs):
        super(Binarynet_MLP, self).__init__()

        self.fc1 = BinLinear(784, 2048, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn1 = nn.BatchNorm1d(2048)
        self.act = nonlinear(kwargs['abits'], kwargs['binary_mode'], kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])

        self.fc2 = BinLinear(2048, 2048, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn2 = nn.BatchNorm1d(2048)

        self.fc3 = BinLinear(2048, 2048, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn3 = nn.BatchNorm1d(2048)

        self.fc4 = BinLinear(2048, kwargs['num_classes'], kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn4 = nn.BatchNorm1d(kwargs['num_classes'])

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x

class Binarynet_vgg(nn.Module):
    def __init__(self, **kwargs):
        super(Binarynet_vgg, self).__init__()

        self.features = nn.Sequential(
            BinConv(3, 128, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinConv(128, 128, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinConv(128, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinConv(256, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinConv(256, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.BatchNorm2d(512),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinConv(512, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

        )
        self.classifier = nn.Sequential(
            BinLinear(512 * 4 * 4, 1024, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale = kwargs['weight_scale']),
            nn.BatchNorm1d(1024),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinLinear(1024, 1024, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(1024),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinLinear(1024, kwargs['num_classes'], wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(kwargs['num_classes'])
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Binarynet_alexnet(nn.Module):
    def __init__(self, **kwargs):
        super(Binarynet_alexnet, self).__init__()


        self.features = nn.Sequential(
            BinConv(3,96, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=11, stride=4, padding=2, padding_mode=kwargs['padding_mode']),
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
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
        )
        self.classifier = nn.Sequential(
            
            BinLinear(256 * 6 * 6, 4096, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(4096),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinLinear(4096, 4096, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(4096),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            BinLinear(4096, kwargs['num_classes'], wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(kwargs['num_classes'])
            
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def binarynet_mlp(**kwargs):
    assert kwargs['dataset'] == 'mnist', f"binarynet mlp model only supports MNIST dataset."
    return Binarynet_MLP(**kwargs)

def binarynet_vgg(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"binarynet vgg model only supports CIFAR-10 dataset."
    return Binarynet_vgg(**kwargs)

def binarynet_alexnet(**kwargs):
    assert kwargs['dataset'] == 'imagenet', f"binarynet alexnet model only supports ImageNet dataset."
    return Binarynet_alexnet(**kwargs)


