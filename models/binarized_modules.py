"""Custom network modules for BNN.

This module contains custom network modules specialized for Binarized Neural Network (BNN).
Binarized activations, weights and their front/backward pass are defined. It also provides
more complex module such as binarized convolution.

"""

import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _pair #YUL: added to use _pair for padding & stride

import utils.padding as Pad
import conv_sweight_cuda


class BinActFunc(torch.autograd.Function):
    """ Binarized activation functions with torch.autograd extension."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, mode='signed', ste='hardtanh', width=1):
        """Performs binarized activation.

        Args:
            ctx: A context ctx is used to store input tensor and ste option.
            x: An input tensor.
            mode(str): Mode to choose a binarization method.
            ste (str): An option that decides the gradients in the backward pass.
            width (float) : width for gradient clipping.

        Returns:
            A tensor containing the sign of input tensor.

        """
        ctx.save_for_backward(x)
        ctx.mode = mode
        ctx.ste = ste
        ctx.width = width

        if mode == 'signed':
            return x.sign()
        elif mode == 'unsigned':
            ## This version uses th=0.5 as default.
            # return torch._C._nn.hardtanh(x, 0, 1).round()
	    
	    ## This version uses th=0. threshold will be covered by higher level class using offset.
            return x.sign().add(1).mul(0.5)

        else:
            assert False, "Binary activation mode {} is not supported.".format(mode)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """Defines a formula for differentiating binarized activation.

        There are several ways to approximate the non-differentiable sign() activation.
        STE passes the gradient as if its derivative is 1, hardtanh clips the gradient
        in (-1,1) range. Approx-sign, introduced in BiRealNet, provides a piecewise linear
        derivative, claiming a closer approximation.

        Args:
            ctx: A context ctx is used to retrieve the saved tensor and related option.
            grad_output: Gradient with respect to the given output

        Returns:
            Gradient with respect to the corresponding input.

        """
        input, = ctx.saved_tensors

        if ctx.mode == 'signed':
            if ctx.ste == 'hardtanh':
                deriv = ((input > -ctx.width) & (input < ctx.width))
                grad_input = grad_output * deriv
            elif ctx.ste == 'linear':
                deriv = ((input > -ctx.width) & (input < ctx.width)) / ctx.width
                grad_input = grad_output * deriv
            elif ctx.ste == 'aprx': # proposed by bireal-net. Also called polynomial
                deriv = input.sign() * input
                deriv = deriv.mul_(-2.0).add_(2.0)
                deriv = deriv * (deriv >= 0)
                grad_input = grad_output * deriv
            elif ctx.ste == 'swish': # proposed by BNN+. Also called sign-swish function
                assert False, "STE function {} is not supported.".format(ctx.ste)
            elif ctx.ste == 'vanilla': #
                grad_input = grad_output.clone()
            elif ctx.ste == 'clippedrelu':
                assert False, "ClippedReLU function is not supported with signed mode."
            else:
                assert False, "STE function {} is not supported.".format(ctx.ste)

        elif ctx.mode == 'unsigned':
            ## this version is for th=0.5.
            # if ctx.ste == 'clippedrelu':
            #     deriv = ((input > 0) & (input < 1))
            #     grad_input = grad_output * deriv
            # elif ctx.ste == 'vanilla':
            #     grad_input = grad_output.clone()
            # elif ctx.ste == 'hardtanh':
            #     assert False, "hardtanh function is not supported with unsigned mode."
            # else:
            #     assert False, "STE function {} is not supported.".format(ctx.ste)

            # this version is for th=0.
            if ctx.ste == 'hardtanh':
                deriv = ((input > -ctx.width/2) & (input < ctx.width/2))
                grad_input = grad_output * deriv
            elif ctx.ste == 'linear':
                deriv = ((input > -ctx.width/2) & (input < ctx.width/2)) / ctx.width
                grad_input = grad_output * deriv
            elif ctx.ste == 'vanilla':
                grad_input = grad_output.clone()
            else:
                assert False, "STE function {} is not supported.".format(ctx.ste)


        return grad_input, None, None, None


class BinWeightFunc(torch.autograd.Function):
    """ Method to binarize weight with torch.autograd extension."""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, weight_clip, weight_scale):
        """Performs weight binarization.

        Args:
            ctx: A context ctx.
            x: An input tensor.
            weight_clip (int): If not 0, clamp the tensor before scaling.

        Returns:
            A scaled binarized tensor(usually weight), optionally clipped with a specified value.

        """
        if weight_clip != 0:
            x.clamp_(-1*weight_clip, weight_clip)
        if weight_scale:
            return x.sign() * x.abs().mean(dim=tuple(range(1,x.dim())),keepdim=True)
        else:
            return x.sign()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        """Defines a formula for differentiating weight binarization."""
        return grad, None, None

def fw(x, wbits, weight_clip, weight_scale):
    """Performs weight quantization in regard to wbits."""
    if wbits == 32:
        return x
    x = BinWeightFunc.apply(x, weight_clip, weight_scale)
    return x


def fa(x, abits, mode, ste, offset, width):
    """Performs quantized activation in regard to abits."""
    if abits == 32:
        return x.tanh()
    x = BinActFunc.apply(x-offset, mode, ste, width)
    return x


###########################################################################################


class BinConv(nn.Conv2d):
    """Applies a Binarized Conv2d operation over an input tensor of several planes.

    The binarized convolution module is different from the full-precision model in some aspect.
    First, it uses quantized weights (output of fw function).
    Second, it provides padding with a value of 1.
    Third, it provides additional features for BNN such as weight_clip and glr.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        wbits (int): Bit resolution of weights.
        weight_clip (int, optional): An optional value to use for weight clipping. Default: 0
        kernel_size (int or tuple, optional): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Padding added to both sides of the input. Padding value
            differs with respect to wbits. If wbits is 32, 0 is padded. Else pval is padded.
        pval: Value to pad when wbit is not 32.

    Shape:
        - Input:
        - Output:

    """
    def __init__(self, in_channels, out_channels, wbits, weight_clip=0, weight_scale=True, kernel_size=3, stride=1, padding=0, padding_mode='zeros', bias=False):
        super(BinConv, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        self.wbits = wbits
        self.padding = padding
        self.stride = stride
        self.padding_mode = padding_mode
        self.glr = 1/math.sqrt(1.5/((in_channels + out_channels)*(kernel_size**2)))

        if self.wbits == 32: # ignore 1-padding and weight clipping when using full precision weight
            # self.padding_value = 0
            self.weight_clip = 0
            self.weight_scale = False
        else:
            self.weight_clip = weight_clip
            self.weight_scale = weight_scale

        if self.padding_mode == 'zeros':
            self.padding_value = 0
        elif self.padding_mode == 'ones':
            self.padding_value = 1
        elif self.padding_mode == 'alter':
            self.padding_value = 0


    def forward(self, input):
        #YUL: delete padding_shpe & additional padding operation by matching padding/stride format with nn.Conv2d
        if self.padding > 0:
            padding_shape = (self.padding, self.padding, self.padding, self.padding)
            input = Pad.pad(input, padding_shape, self.padding_mode, self.padding_value)

        output = F.conv2d(input, fw(self.weight, self.wbits, self.weight_clip, self.weight_scale), bias=self.bias,
                          stride=self.stride, dilation=self.dilation, groups=self.groups)

        return output

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
            if self.padding_value != 0:
                s += ', padding_value={padding_value}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', wbits={wbits}'
        return s.format(**self.__dict__)


class BinLinear(nn.Linear):
    """Applies a Binarized Linear transformation over an input tensor.

    The binarized linear module is different from the full-precision model in that it uses
    quantized weights (output of fw function) and that it provides additional features for BNN
    such as weight_clip and glr.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        wbits (int): Bit resolution of weights.
        weight_clip (int, optional): An optional value to use for weight clipping. Default: 0

    Shape:
        - Input:
        - Output:

    """
    def __init__(self, in_features, out_features, wbits, weight_clip=0, weight_scale=True, bias=False):
        super(BinLinear, self).__init__(in_features, out_features, bias=bias)
        self.wbits = wbits
        self.glr = 1/math.sqrt(1.5/(in_features + out_features))
        self.weight_clip = weight_clip
        self.weight_scale = weight_scale

    def forward(self, input):
        output = F.linear(input, fw(self.weight, self.wbits, self.weight_clip, self.weight_scale), bias=self.bias)
        return output

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        return 'in_features={}, out_features={}, bias={}, wbits={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.wbits
        )


class BinAct(nn.Module):
    """Applies a Binarized activation over an input tensor of several planes.

    This class wraps the function fa to bring it to the same level of abstraction with other
    modules defined in binarized_modules.py

    Args:
        abits (int): Bit resolution of activation.
        mode(str): Mode to choose a binarization method.
        ste (str): An option that decides the gradients in the backward pass.

    Shape:
        - Input:
        - Output:

    """
    def __init__(self, abits, mode='signed', ste='hardtanh', offset=0, width=1):
        super(BinAct, self).__init__()
        self.abits = abits
        self.mode = mode
        self.ste = ste
        self.offset = offset
        self.width = width

    def forward(self, input):
        return fa(input, self.abits, self.mode, self.ste, self.offset, self.width)

    def extra_repr(self):
        """Provides layer information, including abits, when print(model) is called."""
        s = ('abits={abits}')
        s += ', mode={mode}'
        s += ', ste={ste}'
        s += ', offset={offset}'
        s += ', width={width}'
        return s.format(**self.__dict__)


def nonlinear(abits, mode='signed', ste='hardtanh', offset=0, width=1):
    """An activation that can be used as either FP or binary precision

    Args:
        abits (int): Bit resolution of activation. If set to 32, uses ReLU instead.
        mode(str): Mode to choose a binarization method.
        ste (str): An option that decides the gradients in the backward pass.
    """

    if abits == 32:
        return nn.ReLU(inplace=True)
    else:
        return BinAct(abits=abits, mode=mode, ste=ste, offset=offset, width=width)


#######################################################################

class BinaryPsum(torch.autograd.Function):
    """ Sense Amplifier functions with torch.autograd extension."""   

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x):
        """Performs weight binarization.

        Args:
            x: An input tensor.
        Returns:
            A scaled binarized tensor(usually weight), optionally clipped with a specified value.
        """
        output = x.ge(0).type_as(x)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """Defines a formula for differentiating weight binarization."""
        grad_input = grad_output.clone()
        return grad_input

def bfp(x, pbits=1):
    """ Performs quantized psum in regard to pbits. (above 2bits)"""
    if pbits == 1:
        return BinaryPsum.apply(x)
    else:
        assert False, "QuantPsumFunc does not support {} pbits.".format(pbits)


class BinarizedNeurons(nn.Module):
    def __init__(self):
        super(BinarizedNeurons, self).__init__()

    def forward(self, input):
        return bfp(input)
        

