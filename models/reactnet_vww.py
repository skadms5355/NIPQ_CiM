import torch
import torch.nn as nn
from .binarized_modules import *

__all__ = ['reactnet_vww']

def conv3x3(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "3x3 convolution with padding"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=3, stride=stride, padding=1,
                    padding_mode = padding_mode, bias=False)

def conv1x1(in_planes, out_planes, wbits, weight_clip, weight_scale, padding_mode, stride=1):
    "1x1 convolution"
    return BinConv(in_planes, out_planes, wbits, weight_clip, weight_scale, kernel_size=1, stride=stride,
                    padding_mode = padding_mode, bias=False)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
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


class ReActNet_MobileNet(nn.Module):
    """
        Constructs a ReActNet model.
    """

    def __init__(self, **kwargs):
        super(ReActNet_MobileNet, self).__init__()
        self.inplanes = 8
        self.num_classes = kwargs['num_classes']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        layers = []
        layers.append(BasicBlockReAct(self.inplanes, 16, 1, **kwargs))
        layers.append(BasicBlockReAct(16, 32, 2, **kwargs))
        layers.append(BasicBlockReAct(32, 32, 1, **kwargs))
        layers.append(BasicBlockReAct(32, 64, 2, **kwargs))
        layers.append(BasicBlockReAct(64, 64, 1, **kwargs))
        layers.append(BasicBlockReAct(64, 128, 2, **kwargs))
        layers.append(BasicBlockReAct(128, 128, 1, **kwargs))
        layers.append(BasicBlockReAct(128, 128, 1, **kwargs))
        layers.append(BasicBlockReAct(128, 128, 1, **kwargs))
        layers.append(BasicBlockReAct(128, 128, 1, **kwargs))
        layers.append(BasicBlockReAct(128, 128, 1, **kwargs))
        layers.append(BasicBlockReAct(128, 256, 2, **kwargs))
        layers.append(BasicBlockReAct(256, 256, 1, **kwargs))

        self.body = nn.Sequential(*layers)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.body(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def reactnet_vww(**kwargs):
    """
    Construct MobileNetV1-based ReActNet-A model for VWW dataset.
    This model is for vww dataset:
        1) width factor of 0.25 used (number of channels is scaled by 1/4).
        2) last layer is modified for 2-class classification.
    """

    return ReActNet_MobileNet(**kwargs)



