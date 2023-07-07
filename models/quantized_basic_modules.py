"""Custom network modules for QNN.
This module contains custom network modules specialized for Quantized Neural Network (QNN).
Quantized activations, weights and their front/backward pass are defined. It also provides
more complex module such as quantized convolution.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd.function import Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair #YUL: added to use _pair for padding & stride

import utils.padding as Pad
import pquant_group_merge_cuda
import pquant_cuda

class QuantActFunc(torch.autograd.Function):
    """ Quantized activation functions with torch.autograd extension."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, abits=4, ste='clippedrelu', abit_serial=False, amax=1):
        """Performs quantized activation. (based on ReLU1)
        Args:
            ctx: A context ctx is used to store input tensor and ste option.
            input: An input tensor.
            abits (int): Bit resolution of activation.
            ste (str): An option that decides the gradients in the backward pass.
            abit_serial (bool): Bit-serial output or not.
        Returns:
            A tensor containing quantized input tensor.
        """
        if ste == 'clippedrelu':
            deriv = ((input > 0) & (input < amax))
            ctx.save_for_backward(deriv)
        ctx.ste = ste
        ctx.abits = abits
        ctx.abit_serial = abit_serial
        bitserial_step = 1 / (2.**abits - 1.)
        step = amax * bitserial_step
        #ctx.step = step
        ctx.amax = amax
        ## is there any other way to implement this faster?
        # Clamp input in range (0,amax) [ReLU-amax]
        output = input.clamp(0, amax)
        # scale up *xxx_(): inplace xxx function
        output.div_(step)
        # round -> generate quantization stages
        output.round_()

        # scale down
        output.mul_(step)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """Defines a formula for differentiating quantized activation.
        There are several ways to approximate the non-differentiable activation.
        STE passes the gradient as if its derivative is 1, clippedrelu clips the gradient
        in (0,1) range.
        Args:
            ctx: A context ctx is used to retrieve the saved tensor and related option.
            grad_output: Gradient with respect to the given output
        Returns:
            Gradient with respect to the corresponding input.
        """
        grad_tmp = grad_output

        # compute grad_input
        if ctx.ste == 'clippedrelu':
            deriv, = ctx.saved_tensors
            grad_input = grad_tmp * deriv
        elif ctx.ste == 'vanilla':
            grad_input = grad_tmp.clone()
        else:
            assert False, "STE function {} is not supported.".format(ctx.ste)

        return grad_input, None, None, None, None

class QuantWeightFunc(torch.autograd.Function):
    """ Method to quantize weight with torch.autograd extension."""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, wbits, wquant="fixed", weight_mask=None):
        """Performs weight Quantization
        """
        # mask the weight if it exist
        if weight_mask is not None:
            input_masked = input[weight_mask.gt(0.5)]
        else:
            input_masked = input

        half_num_level = 2.**(wbits-1)

        # step for unit variance 
        if wquant == "fixed":
            if wbits == 2:
                step = 0.996
            elif wbits == 3:
                step = 0.586
            elif wbits == 4:
                step = 0.335
            elif wbits == 5:
                step = 0.1882
            else:
                step = 2/(2.**(wbits - 1.))

            ## scale the step with std (output-channel wise std)
            #w_std = input.std(tuple(range(1, input.dim())), keepdim=True)
            # scale the step with std (layer-wise std)
            w_std = input_masked.std()
            w_step = w_std.mul(step)

        elif wquant == "sawb":
            assert wbits==2, "swab only support 2 bit weight quantization now"
            q_range = 3.2 * input_masked.std() + 2.1 * input_masked.abs().mean()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        elif wquant == "sawb_no_std":
            assert wbits==2, "swab only support 2 bit weight quantization now"
            q_range = 2.1 * input_masked.abs().mean()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        elif wquant == "3std":
            q_range = 3. * input_masked.std()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        elif wquant == "2std":
            q_range = 2. * input_masked.std()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        else:
            assert False, "weight quantization does not support option {}".format(wquant)

        # scale & shift the weight to make sure quantized values are integer with distance 1
        w_q = input / w_step
        w_q.add_(0.5)

        # quantize the weight with round function
        w_q.round_()
        # clamp the weight values to have (2^wbits) levels
        w_q.clamp_(-1*half_num_level + 1, half_num_level)

        # return the weight values
        w_q.add_(-0.5)

        # mask the weight if it exist
        if weight_mask is not None:
            w_q.mul_(weight_mask)
       
        return w_q * w_step

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        """Defines a formula for differentiating weight quantization."""
        # return as many tensors as there were inputs
        return grad, None, None, None


class QuantWeightReturnScaleFunc(torch.autograd.Function):
    """ Method to quantize weight with torch.autograd extension."""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, wbits, wquant="fixed", weight_mask=None):
        """Performs weight Quantization
            1. wquant: fixed
            determine the steps of quantized weights based on Fixed Point Quant (ICML 2016) - 2b ~ 5b
            For other bits, the step size is 1/(2.**wbits - 1.)
            Reference:
               https://arxiv.org/pdf/1511.06393.pdf
            2. wquant: sawb
            quant range = c1 * std - c2 * abs_mean (c1=3.2, c2=-2.1 for 2b)
            Reference:
                https://mlsys.org/Conferences/2019/doc/2019/168.pdf
            3. wquant: sawb_no_std
            quant range = - c2 * mean (c2=-2.1 for 2b)
            Reference:
                https://mlsys.org/Conferences/2019/doc/2019/168.pdf
            4. wquant: 3std
            quant range: 3std
            5. wquant: 2std
            quant range: 2std
            Args:
               ctx: A context ctx.
               x: An input tensor.
               wbits (int): Bit resolution of weights.
           Returns:
               A quantized tensor(usually weight)
        """
        # mask the weight if it exist
        if weight_mask is not None:
            input_masked = input[weight_mask.gt(0.5)]
        else:
            input_masked = input

        half_num_level = 2.**(wbits-1)

        # step for unit variance 
        if wquant == "fixed":
            if wbits == 2:
                step = 0.996
            elif wbits == 3:
                step = 0.586
            elif wbits == 4:
                step = 0.335
            elif wbits == 5:
                step = 0.1882
            else:
                step = 2/(2.**(wbits - 1.))

            ## scale the step with std (output-channel wise std)
            #w_std = input.std(tuple(range(1, input.dim())), keepdim=True)
            # scale the step with std (layer-wise std)
            w_std = input_masked.std()
            w_step = w_std.mul(step)

        elif wquant == "sawb":
            assert wbits==2, "swab only support 2 bit weight quantization now"
            q_range = 3.2 * input_masked.std() + 2.1 * input_masked.abs().mean()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        elif wquant == "sawb_no_std":
            assert wbits==2, "swab only support 2 bit weight quantization now"
            q_range = 2.1 * input_masked.abs().mean()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        elif wquant == "3std":
            q_range = 3. * input_masked.std()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        elif wquant == "2std":
            q_range = 2. * input_masked.std()
            w_step = 2. * q_range / ((2.**wbits) - 1.)

        else:
            assert False, "weight quantization does not support option {}".format(wquant)

        # scale & shift the weight to make sure quantized values are integer with distance 1
        w_q = input / w_step
        w_q.add_(0.5)

        # quantize the weight with round function
        w_q.round_()
        # clamp the weight values to have (2^wbits) levels
        w_q.clamp_(-1*half_num_level + 1, half_num_level)

        # return the weight values
        w_q.add_(-0.5)

        # mask the weight if it exist
        if weight_mask is not None:
            w_q.mul_(weight_mask)

        #ctx.wbits = wbits
        ctx.wscale = w_step/2
        return w_q * w_step, ctx.wscale

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output1, grad_output2):
        """Defines a formula for differentiating weight quantization."""
        # return as many tensors as there were inputs
        #grad_input = grad_output1.div(ctx.wscale) ## NOTE: this is moved to bitserial! -- just removed
        #return grad_input, None, None, None
        return grad_output1, None, None, None

class BinWeightFunc(torch.autograd.Function):
    """ Method to binarize weight with torch.autograd extension."""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, weight_clip, weight_scale, fan_in, groups, return_scale, weight_mask=None):
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
            # mask weight
            if weight_mask is not None:
                x.mul_(weight_mask)
            # calculate scale
            scale = x.abs().sum(tuple(range(1, x.dim())), keepdim=True)
            # adapt scale to conv split
            split_scale = scale.chunk(groups, dim=0)
            for i in range(1, groups):
                split_scale[0].add_(split_scale[i])
            # compute average
            split_scale[0].div_(fan_in)
            # copy scale throughout groups
            if return_scale:
                return x.sign(), split_scale[0]
            else:
                for i in range(1, groups):
                    split_scale[i].copy_(split_scale[0])
                return x.sign() * scale
            #scale = x.abs().mean(tuple(range(1, x.dim())), keepdim=True)
            #return x.sign() * scale
        else:
            return x.sign()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        """Defines a formula for differentiating weight binarization."""
        # return as many tensors as there were inputs
        return grad, None, None, None, None, None, None

class MaskWeight(torch.autograd.Function):
    """ Method to mask weight with torch.autograd extension."""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, weight_mask=None):
        # mask weight
        if weight_mask is not None:
            return x.mul(weight_mask)
        else:
            return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        # return as many tensors as there were inputs
        return grad, None


def fw(x, wbits, weight_clip, weight_scale, fan_in, groups=1, wquant="fixed", return_scale=False, weight_mask=None):
    """Performs weight quantization in regard to wbits."""
    if wbits == 32:
        x = MaskWeight.apply(x, weight_mask)
    elif wbits > 1:
        if return_scale:
            x = QuantWeightReturnScaleFunc.apply(x, wbits, wquant, weight_mask)
        else:
            x = QuantWeightFunc.apply(x, wbits, wquant, weight_mask)
    else:
        x = BinWeightFunc.apply(x, weight_clip, weight_scale, fan_in, groups, return_scale, weight_mask)
    return x


def fa(x, abits, ste, abit_serial, amax):
    """Performs quntized activation."""
    x = QuantActFunc.apply(x, abits, ste, abit_serial, amax)
    return x

###########################################################################################

class QuantConv(nn.Conv2d):
    """Applies a Quantized Conv2d operation over an input tensor of several planes.
    The quantized convolution module is different from the full-precision model in some aspect.
    First, it uses quantized weights (output of fw function).
    Second, it provides padding with a value of 1.
    Third, it provides additional features for QNN such as weight_clip and glr.
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        wbits (int): Bit resolution of weights.
        weight_clip (int, optional): An optional value to use for weight clipping. Default: 0
        kernel_size (int or tuple, optional): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Padding added to both sides of the input. Padding value
            differs with respect to wbits. If wbits is 32, 0 is padded. Else pval is padded.
        padding_mode: Decide padding values.
    Shape:
        - Input:
        - Output:
    """
    def __init__(self, in_channels, out_channels, wbits, weight_clip=0, weight_scale=True, kernel_size=3, stride=1, padding=0, padding_mode='zeros', groups=1, bias=False, wquant="fixed"):
        super(QuantConv, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, groups=groups, bias=bias)
        self.wbits = wbits
        self.padding = padding
        self.stride = stride
        self.padding_mode = padding_mode
        self.glr = 1/math.sqrt(1.5/(in_channels + out_channels)*kernel_size)
        self.weight_clip = weight_clip
        self.weight_scale = weight_scale
        self.fan_in = in_channels * kernel_size * kernel_size
        self.wquant = wquant

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
        # weight quantization
        weight_q = fw(self.weight, self.wbits, self.weight_clip, self.weight_scale, self.fan_in, wquant=self.wquant)
        #print('weight_q: {}'.format(weight_q.dtype))
        #print(weight_q)
        # Conv2D
        output = F.conv2d(input, weight_q, bias=self.bias,
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
        s += ', wbits={wbits}, wquant={wquant}'
        return s.format(**self.__dict__)


class QuantLinear(nn.Linear):
    """Applies a Quantized Linear transformation over an input tensor.
    The Quantized linear module is different from the full-precision model in that it uses
    quantized weights (output of fw function) and that it provides additional features for QNN
    such as weight_clip and glr.
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        wbits (int): Bit resolution of weights.
        weight_clip (int, optional): An optional value to use for weight clipping (Binary Weight). Default: 0
        weight_scale (bool, optional): An optional value to use for weight scale (Binary Weight). Default: True
    Shape:
        - Input:
        - Output:
    """
    def __init__(self, in_features, out_features, wbits, weight_clip=0, weight_scale=True, bias=False, wquant="fixed"):
        super(QuantLinear, self).__init__(in_features, out_features, bias=bias)
        self.wbits = wbits
        self.glr = 1/math.sqrt(1.5/(in_features + out_features))
        self.weight_clip = weight_clip
        self.weight_scale = weight_scale
        self.wquant = wquant

    def forward(self, input):
        # weight quantization
        weight_q = fw(self.weight, self.wbits, self.weight_clip, self.weight_scale, self.in_features, wquant=self.wquant)
        # Linear
        output = F.linear(input, weight_q, bias=self.bias)
        return output

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        return 'in_features={}, out_features={}, bias={}, wbits={}, wquant={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.wbits, self.wquant
        )


class QuantAct(nn.Module):
    """Applies a Quantized activation over an input tensor of several planes.
    This class wraps the function fa to bring it to the same level of abstraction with other
    modules defined in quantized_modules.py
    Args:
        abits (int): Bit resolution of activation.
        ste (str): An option that decides the gradients in the backward pass.
    Shape:
        - Input:
        - Output:
    """
    def __init__(self, abits, ste='clippedrelu', abit_serial=False, amax=1):
        super(QuantAct, self).__init__()
        self.abits = abits
        self.ste = ste
        self.abit_serial = abit_serial
        self.amax = amax

    def forward(self, input):
        return fa(input, self.abits, self.ste, self.abit_serial, self.amax)

    def extra_repr(self):
        """Provides layer information, including activation resolution, when print(model) is called."""
        s = ('abits={abits}')
        s += ', ste={ste}, abit_serial={abit_serial}, amax={amax}'
        return s.format(**self.__dict__)


def Qnonlinear(abits, ste='clippedrelu', abit_serial=False, amax=1):
    """An activation that can be used as either FP or N-bit precision
    Args:
        abits (int): Bit resolution of activation. If set to 32, uses ReLU instead.
        ste (str): An option that decides the gradients in the backward pass.
    """

    if abits == 32:
        return nn.ReLU(inplace=True)
    else:
        return QuantAct(abits, ste=ste, abit_serial=abit_serial, amax=amax)


###########################################################################################

class QuantPsumMerge(torch.autograd.Function):
    """ Quantized partial sum functionsi & merge group with torch.autograd extension."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, output, input, pbits=32, step=1, half_num_levels=1, 
                    pbound=None, pste='clipped', weight=1, center=0, groups=1, pzero=False, debug=False):
        """Performs partial sum quantization. (in bitserial layer) & merge conv/linear split group

        Args:
            ctx: A context ctx is used to store input tensor and ste option.
            input: An input tensor.
            bound (int): bound of quantization range ([-bound, +bound])
            pste (str): An option that decides the gradients in the backward pass.
        Returns:
            A tensor containing quantized input tensor.

        """
        ctx.mark_dirty(output)
        ctx.pste = pste
        ##return pquant_merge_cpp.forward(output, input, pbits, step, half_num_levels, weight, center, pzero) 
        return pquant_group_merge_cuda.forward(output, input, pbits, step, half_num_levels, weight, center, groups, pzero) 


        # output_cat= None
        # if ( pbits == 32 ):
        #     input_merge = input[0]
        #     for g in range(1, groups):
        #         input_merge += input[g]
        #     output += input_merge
        # else:
        #     step_tmp = step * weight
        #     for g in range(0, groups):
        #         out_tmp = input[g].sub(center)
        #         out_tmp.div_(step_tmp)
        #         if (pzero):
        #             if ( pbits == 1):
        #                 out_tmp.ge_(0)
        #                 out_tmp = out_tmp * 2 - 1
        #             else:
        #                 out_tmp.round_()
        #                 out_tmp.clamp_(-1 * half_num_levels+1, half_num_levels)
        #         else:
        #             #print('not pzero')
        #             if ( pbits == 1):
        #                 out_tmp.ge_(0)
        #                 out_tmp = out_tmp * 2 - 1
        #             else:
        #                 out_tmp.round_()
        #                 out_tmp.clamp_(0, 2*half_num_levels-1)

        #         out_tmp.mul_(step_tmp)
        #         out_tmp.add_(center)
        #         output_cat = torch.cat([output_cat, out_tmp], dim=1) if g!=0 else out_tmp
        #         output += out_tmp
           
            # output_set = list(set(output_cat.cpu().numpy().ravel()))
            # print(half_num_levels)
            # print(step)
            # print(weight)
            # print('test', sorted(output_set))
            # import pdb; pdb.set_trace()

        return output


    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        return grad_output, grad_output, None, None, None, None, None, None, None, None

def psum_quant_merge(output, input, pbits=32, step=1, half_num_levels=1, pbound=None, pste='clipped', weight=1, center=None, groups=1, pzero=False):
    if center is None:
        center = 0
    if output is None:
        output = torch.zeros_like(input[0])

    return QuantPsumMerge.apply(output, input, pbits, step, half_num_levels, pbound, 
                            pste, weight, center, groups, pzero)

class QuantPsum(torch.autograd.Function):
    """ Quantized partial sum function with torch.autograd extension."""
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, output, input, pbits=32, step=1, half_num_levels=1, 
                    pbound=None, weight=1, center=0, groups=1, pzero=False, debug=False):
        """Performs partial sum quantization. (in bitserial layer)

        Args:
            ctx: A context ctx is used to store input tensor and ste option.
            input: An input tensor.
            bound (int): bound of quantization range ([-bound, +bound])
            pste (str): An option that decides the gradients in the backward pass.
        Returns:
            lists (input tensor size) containing quantized input tensor.

        """
        # ctx.mark_dirty(output)
        return pquant_cuda.forward(output, input, pbits, step, half_num_levels, weight, center, groups, pzero) 


    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        return grad_output, grad_output, None, None, None, None, None, None, None, None
    
def psum_quant(output, input, pbits=32, step=1, half_num_levels=1, pbound=None, weight=1, center=None, groups=1, pzero=False):
    if center is None:
        center = 0
    if output is None:
        output = []
        for g in range(groups):
            output.append(torch.zeros_like(input[0]))

    return QuantPsum.apply(output, input, pbits, step, half_num_levels, pbound, 
                            weight, center, groups, pzero)


class QuantizePsum(torch.autograd.Function):
    """ Binarized activation functions with torch.autograd extension."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, pbits=5, maxVal=256, minVal=0):
        level_step = 1 / (2.**pbits-1)
        step = (maxVal - minVal) * level_step
        ctx.save_for_backward(x)
        ctx.maxVal = maxVal
        ctx.minVal = minVal
        x.clamp_(minVal, maxVal)
        x.div_(step)
        # round -> generate quantization stages
        x.round_() # pbits range

        # scale down
        x.mul_(step)

        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output*((input>=ctx.minVal)&(input<ctx.maxVal))
        return grad_output, grad_input, None, None, None, None

def fp(x, pbits, maxVal=256, minVal=0):
    """ Performs quantized psum in regard to pbits. (above 2bits)"""
    if pbits == 32:
        return x
    elif pbits < 16 and pbits != 1:
        return QuantizePsum.apply(x, pbits, maxVal, minVal)
    else:
        assert False, "QuantPsumFunc does not support {} pbits.".format(pbits)