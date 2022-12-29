from re import I
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter 
from numpy import inf
from .binarized_modules import *
from .bitserial_modules import *

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LSQ(nn.Module):
    def __init__(self, bit, half_range=False, symmetric=False, per_channel=False, bitserial=False, quant_group=None):
        super().__init__()

        self.bit = bit
        self.bitserial = bitserial
        if (bit != 32) and (bit != 1):

            if half_range:
                self.Qn = 0
                self.Qp = 2 ** bit - 1
            else:
                if symmetric: 
                    self.Qn = - 2 ** (bit - 1) + 1
                    self.Qp = 2 ** (bit - 1) - 1
                else:
                    self.Qn = - 2 ** (bit - 1)
                    self.Qp = 2 ** (bit - 1) - 1

            self.per_channel = per_channel
            if per_channel:
                self.s = nn.Parameter(torch.zeros(quant_group, 1, 1, 1))
                self.s.data.fill_(1)
            else:
                self.s = nn.Parameter(torch.Tensor(1)[0])
            self.register_buffer('init_state', torch.zeros(1))
            # self.register_buffer('s_scale', torch.zeros(1))

    def init_from(self, x):
        if self.per_channel:
            self.s.data.copy_(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.Qp ** 0.5))
        else:
            #self.s = nn.Parameter(torch.cuda.HalfTensor([(x.detach().abs().mean() * 2 / (self.Qp ** 0.5))]))
#            self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.Qp ** 0.5))
            self.s.data.fill_(x.detach().abs().mean() * 2 / (self.Qp ** 0.5))
#        print(self.s)

    def forward(self, x):
        if self.bit == 32:
            return x
        elif self.bit == 1:
            return fw(x, self.bit, 1, False)
        else:
            if self.training and self.init_state == 0:
                self.init_from(x)
                self.init_state.fill_(1)
                #print('init_state done, device : ', torch.cuda.current_device())
            s_grad_scale = 1.0 / ((self.Qp * x.numel()) ** 0.5)
            # self.s_scale.fill_(grad_scale(self.s, s_grad_scale))
            s_scale = grad_scale(self.s, s_grad_scale)
            
            x = x / s_scale
            x = torch.clamp(x, self.Qn, self.Qp)
            x = round_pass(x)
            x = x * s_scale

            # call cls func to store bitserial value when self.bitserial is False 
            Bitserial.get_value(self.bit, s_scale, self.bitserial)

            return x

class LSQReturnScale(nn.Module):
    def __init__(self, bit, half_range=False, symmetric=False, per_channel=True, quant_group=None):
        super().__init__()
        
        self.bit = bit
        if (bit != 32) and (bit != 1):

            if half_range:
                self.Qn = 0
                self.Qp = 2 ** bit - 1
            else:
                if symmetric: 
                    self.Qn = - 2 ** (bit - 1) + 1
                    self.Qp = 2 ** (bit - 1) - 1
                else:
                    self.Qn = - 2 ** (bit - 1)
                    self.Qp = 2 ** (bit - 1) - 1
            
            self.per_channel = per_channel
            if per_channel:
                self.s = nn.Parameter(torch.zeros(quant_group, 1, 1, 1))
                self.s.data.fill_(1)
            else:
                self.s = nn.Parameter(torch.Tensor(1)[0])
            self.register_buffer('init_state', torch.zeros(1))
            # self.register_buffer('s_scale', torch.zeros(1))

    def init_from(self, x):
        if self.per_channel:
            self.s.data.copy_(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.Qp ** 0.5))
        else:
            #self.s = nn.Parameter(torch.cuda.HalfTensor([(x.detach().abs().mean() * 2 / (self.Qp ** 0.5))]))
#            self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.Qp ** 0.5))
            self.s.data.fill_(x.detach().abs().mean() * 2 / (self.Qp ** 0.5))
#        print(self.s)

    def forward(self, x):
        if self.bit == 32:
            return x
        elif self.bit == 1:
            return fw(x, self.bit, 1, False)
        else:
            if self.training and self.init_state == 0:
                self.init_from(x)
                self.init_state.fill_(1)
                #print('init_state done, device : ', torch.cuda.current_device())
            s_grad_scale = 1.0 / ((self.Qp * x.numel()) ** 0.5)
            # self.s_scale.fill_(grad_scale(self.s, s_grad_scale))
            s_scale = grad_scale(self.s, s_grad_scale)
            
            x = x / s_scale
            x = torch.clamp(x, self.Qn, self.Qp)
            x = round_pass(x)
            x = x * s_scale

            # print('weight s_scale:', s_scale)
            return x, s_scale

class QConvWS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, wbits=32, kernel_size=3, stride=1, padding=0, groups=1, bias=True):
        super(QConvWS, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, groups=groups, bias=bias)
        self.wbits = wbits
        self.padding = padding
        self.stride = stride
        #self.boundary = 1.0
        #self.weight_clip_scale = nn.Parameter(torch.Tensor([self.boundary]))
        #self.quant_group = out_channels
        #self.weight_clip_scale = nn.Parameter(torch.zeros(self.quant_group, 1, 1, 1))
        #self.weight_clip_scale.data.fill_(self.boundary)
        self.quan_w_fn = LSQ(bit=self.wbits, half_range=False, per_channel=True, quant_group=out_channels)
        #self.quan_w_fn.init_from(self.weight)

    def forward(self, input):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-6
        weight = weight / std.expand_as(weight)

        output =  F.conv2d(input, self.quan_w_fn(weight), bias=self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        
        return output

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', wbits={wbits}'
        return s.format(**self.__dict__)


class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, wbits=32, kernel_size=3, stride=1, padding=0, groups=1, symmetric=False, bias=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, groups=groups, bias=bias)
        self.wbits = wbits
        self.padding = padding
        self.stride = stride
        #self.boundary = 1.0
        #self.weight_clip_scale = nn.Parameter(torch.Tensor([self.boundary]))
        #self.quant_group = out_channels
        #self.weight_clip_scale = nn.Parameter(torch.zeros(self.quant_group, 1, 1, 1))
        #self.weight_clip_scale.data.fill_(self.boundary)
        self.quan_w_fn = LSQ(bit=self.wbits, half_range=False, symmetric=symmetric, per_channel=False)
        #self.quan_w_fn.init_from(self.weight)

    def forward(self, input):
        output =  F.conv2d(input, self.quan_w_fn(self.weight), bias=self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', wbits={wbits}'
        return s.format(**self.__dict__)

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, wbits, symmetric=False, bias=False):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)
        self.wbits = wbits
        #self.weight_clip_scale = nn.Parameter(torch.Tensor([self.boundary]))
        #self.weight_clip_scale = nn.Parameter(torch.zeros(self.quant_group, 1, 1, 1))
        #self.weight_clip_scale.data.fill_(self.boundary)
        self.quan_w_fn = LSQ(bit=self.wbits, half_range=False, symmetric=symmetric, per_channel=False)

    def forward(self, input):
        output =  F.linear(input, self.quan_w_fn(self.weight), bias=self.bias)
        return output

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        return 'in_features={}, out_features={}, bias={}, wbits={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.wbits
        )

class Q_act(LSQ):
    def __init__(self, abits, half_range=True, per_channel=False, bitserial=False):
        super().__init__(bit=abits, half_range=True, per_channel=False, bitserial=bitserial)

def add_act(abits, bitserial=False):
    if abits == 32:
        return nn.ReLU(inplace=True)
    else:
        return Q_act(abits=abits, bitserial=bitserial)

def quant_or_not(abits):
    if abits == 32:
        return nn.Identity()
    else:
        return LSQ(bit=abits, half_range=False, per_channel=False)