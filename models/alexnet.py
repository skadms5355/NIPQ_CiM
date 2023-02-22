"""AlexNet (full or quanitzed precision expect binary)
"""

import torch
import torch.nn as nn
from .quantized_lsq_modules import *
from .quantized_basic_modules import *

__all__ = ['lsq_alexnet']

class conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, wbits, kernel_size=3, stride=1, padding=1, max_pool=False, abits=32):
        super(conv2d_block, self).__init__()
        """Generate convolution block"""
        self.conv = QConv(in_channels, out_channels, wbits=wbits, \
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if max_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        else:
            self.maxpool = None
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = add_act(abits=abits)

    def forward(self, x):
        out = self.conv(x)
        if self.maxpool is not None:
            out = self.maxpool(out)
        out = self.bn(out)
        out = self.act(out)
        return out

class fc_block(nn.Module):
    def __init__(self, in_features, out_features, wbits, abits):
        super(fc_block, self).__init__()
        """Generate fc block"""
        self.fc = QLinear(in_features, out_features, wbits=wbits, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = add_act(abits=abits)

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        return out


class LSQ_Alexnet(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(LSQ_Alexnet, self).__init__()

        self.wbits = kwargs['wbits']
        self.abits = kwargs['abits']
        self.padding_mode = kwargs['padding_mode']
        self.dropRate = kwargs['drop']

        # convolution layers for feature extraction
        conv_layers = []
        #layer 1
        conv_layers.append(conv2d_block(3, 64, wbits=self.wbits, kernel_size=11, stride=4, padding=2, max_pool=True,
                        abits=self.abits))
        # layer 2 
        conv_layers.append(conv2d_block(64, 256, self.wbits, kernel_size=5, stride=1, padding=2, max_pool=True,
                        abits=self.abits))
        # layer 3
        conv_layers.append(conv2d_block(256, 384, self.wbits, kernel_size=3, stride=1, padding=1, max_pool=False,
                        abits=self.abits))
        # layer 4
        conv_layers.append(conv2d_block(384, 256, self.wbits, kernel_size=3, stride=1, padding=1, max_pool=False,
                        abits=self.abits))
        # layer 5
        conv_layers.append(conv2d_block(256, 256, self.wbits, kernel_size=3, stride=1, padding=1, max_pool=True,
                        abits=self.abits))


        # generate nn.Sequential with conv_layers
        self.features = nn.Sequential(*conv_layers)

        # FC layer for classification
        fc_layers = []
        # layer 1
        fc_layers.append(fc_block(256 * 6 * 6, 4096, self.wbits, abits=self.abits))
        # layer 2
        fc_layers.append(fc_block(4096, 4096, self.wbits, abits=self.abits))
        # layer 3
        fc_layers.append(nn.Dropout(self.dropRate))
        fc_layers.append(nn.Linear(4096, num_classes)) # bias=True (default)
        
        # generate nn.Sequential with fc_layers
        self.classifier = nn.Sequential(*fc_layers)
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def lsq_alexnet(**kwargs):
    assert kwargs['dataset']=='imagenet', 'We only support alexnet for imagenet dataset'

    return LSQ_Alexnet(**kwargs)