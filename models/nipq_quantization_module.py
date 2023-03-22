from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict
from .noise_cell import Noise_cell

# Uniform random Noise (rand_like)
class qnoise_make(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO randomly sample random seed
        rseed = torch.randint(0, 32765, (1,))
        ctx.save_for_backward(rseed)
        torch.manual_seed(rseed)

        # LSQ range
        # step size 1 => Uniform random Noise U[-0.5, 0.5]
        noise = torch.rand_like(x) - 0.5 

        return noise

    @staticmethod
    def backward(ctx, grad_output):
        rseed = ctx.saved_tensors[0]
        torch.manual_seed(rseed)

        return grad_output

class Quantizer(nn.Module):
    def __init__(self, sym, noise, hnoise=False, offset=0., is_stochastic=True, is_discretize=True):
        super(Quantizer, self).__init__()
        self.bit = Parameter(torch.Tensor(1).zero_())
        self.alpha = Parameter(torch.Tensor(1).fill_(1))
        
        self.sym = sym
        self.offset = offset
        self.noise = noise
        self.hnoise = hnoise
        self.is_stochastic = is_stochastic
        self.is_discretize = is_discretize
        self.register_buffer('init_state', torch.zeros(1))


    def lsq_forward(self, data, bit, alpha, sym):
        if sym:
            Qp = 2 ** (bit.detach() - 1) - 1
            Qn = - 2 ** (bit.detach() - 1) # all level is used
        else:
            Qp = 2 ** bit.detach() - 1
            Qn = torch.zeros(1, device=Qp.device)
        
        data_q = torch.clamp(data / alpha, Qn, Qp)
        out = (data_q.round() + (data_q - data_q.detach())) * alpha
        
        return out, Qn, Qp

    def lsq_init(self, x, bit):
        Qp = 2 ** (bit.detach() - 1) - 1 if self.sym else 2 ** bit.detach() - 1
        self.alpha.data[0] = (x.detach().abs().mean() * 2 / (Qp ** 0.5))
    
    def hnoise_init(self, cbits, mapping_mode, co_noise=0.01, noise_type='prop', res_val='rel', max_epoch=-1):
        bit = 2 + torch.sigmoid(self.bit)*12
        w_format = 'state' if res_val == 'abs' or noise_type == 'meas' else 'weight'
        self.noise_cell = Noise_cell(bit.round().squeeze(), cbits, mapping_mode, co_noise, noise_type=noise_type, \
                                    res_val=res_val, w_format=w_format, max_epoch=max_epoch)
    
    def get_alpha(self):
        return self.alpha

    def get_bit(self):
        bit = 2 + torch.sigmoid(self.bit)*12
        return bit.round_().squeeze()

    def forward(self, x, is_training=True, serial=False):
        # parameter define
        sym = self.sym
        offset = self.offset
        noise = self.noise 
        hnoise = self.hnoise
        is_stochastic = self.is_stochastic
        is_discretize = self.is_discretize
        
        bit = 2 + torch.sigmoid(self.bit)*12

        #Stochastic Rounding
        if is_training and noise and is_stochastic :
            bit += (torch.rand_like(bit) - 0.5)
    
        if not is_training or is_discretize :
            bit = bit.round() + (bit - bit.detach())
        
        if is_training and self.init_state == 0:
            self.lsq_init(x, bit.round())
            self.init_state.fill_(1)
        
        # alpha = F.softplus(self.alpha)
        alpha = self.alpha # LSQ version 
        lsq, Qn, Qp = self.lsq_forward(x+offset, bit.round(), alpha, sym)

        if is_training and noise:
            x = (x + offset) / alpha + qnoise_make.apply(x)
            x = torch.clamp(x, Qn, Qp) 
            if hnoise:
                x = self.noise_cell(x, float_comp=True, w_split=serial)
            return x * alpha - offset
        else:
            if hnoise and not serial:
                # import seaborn as sns
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                # # xlabels = [np.arange(-8, 7)]
                # # ax.set_xticks(xlabels)
                # ax.set_xticks(np.arange(-8, 8, 1))
                # sns.histplot(data=(lsq/alpha).detach().cpu().numpy().ravel())
                # plt.savefig('./weight_hist.png')
                # plt.close()
                lsq = self.noise_cell((lsq / alpha).round(), w_split=serial) * alpha
                # fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                # ax.set_xticks(np.arange(-8, 8, 1))
                # sns.histplot(data=(lsq/alpha).detach().cpu().numpy().ravel())
                # plt.savefig('./weight_hist_noise.png')
                # import pdb; pdb.set_trace()
            return lsq - offset


class Q_ReLU(nn.Module):
    def __init__(self):
        super(Q_ReLU, self).__init__()
        self.quant = False
        self.quant_func = Quantizer(sym=False, noise=True, offset=0, is_stochastic=True, is_discretize=True)


    def forward(self, x):
        if self.quant is False:
            return x        
        return self.quant_func(x, self.training)

    
class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.quant = False
        self.quant_func = Quantizer(sym=True, noise=True, offset=0, is_stochastic=True, is_discretize=True)

    # symmetric & zero-included quant
    # TODO: symmetric & zero-excluded quant 
    def forward(self, x):
        if self.quant is False:
            return x        
        return self.quant_func(x, self.training)


class Q_HSwish(nn.Module):
    def __init__(self):
        super(Q_HSwish, self).__init__()
        self.quant = False
        self.quant_func = Quantizer(sym=False, noise=True, offset=3/8, is_stochastic=True, is_discretize=True)

    def forward(self, x):
        if self.quant is False:
            return x
        return self.quant_func(x, self.training)


class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.quant = False
        self.quant_func = Quantizer(sym=True, noise=True, offset=0, is_stochastic=True, is_discretize=True)

    
    def _weight_quant(self):
        if self.quant is False:
            return self.weight
        return self.quant_func(self.weight, self.training)
        
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        if self.act_func is not None:
            x = self.act_func(x)

        return F.conv2d(x, self._weight_quant(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
    

class Q_Linear(nn.Linear):
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.quant = False
        self.quant_func = Quantizer(sym=True, noise=True, offset=0, is_stochastic=True, is_discretize=True)

    
    def _weight_quant(self):
        if self.quant is False:
            return self.weight
        return self.quant_func(self.weight, self.training)
        
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
            
        return F.linear(x, self._weight_quant(), self.bias)
    
    
def initialize(model, act=False, weight=False, noise=True, is_stochastic=True, is_discretize=True, fixed_bit=-1):
    for name, module in model.named_modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:
            module.quant = True
            module.quant_func.noise = noise

            module.quant_func.is_stochastic = is_stochastic
            module.quant_func.is_discretize = is_discretize

            if fixed_bit != -1 :
                bit = ( fixed_bit+0.00001 -2 ) / 12
                bit = np.log(bit/(1-bit))
                module.quant_func.bit.data.fill_(bit)
                module.quant_func.bit.requires_grad = False
            
            #module.bit.data.fill_(-2)

        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight:
            module.quant = True
            module.quant_func.noise = noise
            
            module.quant_func.is_stochastic = is_stochastic
            module.quant_func.is_discretize = is_discretize

            #module.bit.data.fill_(-2)

            if fixed_bit != -1 :
                bit = ( fixed_bit -2 ) / 12
                bit = np.log(bit/(1-bit))
                module.quant_func.bit.data.fill_(bit)
                module.quant_func.bit.requires_grad = False
            
def hnoise_initilaize(model, weight=False, hnoise=True, cbits=4, mapping_mode=None, co_noise=0.01, noise_type='prop', res_val='rel', max_epoch=-1):
    for name, module in model.named_modules():
        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight and hnoise:
            module.quant_func.hnoise = True

            if noise_type == 'grad':
                assert max_epoch != -1, "Enter max_epoch in hnoise_initialize function"
            if hnoise:
                module.quant_func.hnoise_init(cbits=cbits, mapping_mode=mapping_mode, co_noise=co_noise, noise_type=noise_type, res_val=res_val, max_epoch=max_epoch)

def sample_activation_size(model, x):
    # forward hook for bops calculation (output h & w)
    # `out_height`, `out_width` for Conv2d
    hooks = []

    def forward_hook(module, inputs, outputs):
        module.out_shape = outputs.shape

    for _, module in model.named_modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)):
            hooks.append(module.register_forward_hook(forward_hook))
    
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(forward_hook))

    # with torch.no_grad():
    model.eval()
    model.cuda()
    model(x.cuda())

    for hook in hooks:
        hook.remove()

    return

def compute_bops(
    kernel_size, in_channels, filter_per_channel, h, w, bits_w=32, bits_a=32
):
    conv_per_position_flops = (
        kernel_size * kernel_size * in_channels * filter_per_channel
    )
    active_elements_count = h * w
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bops = overall_conv_flops * bits_w * bits_a
    return bops

def bops_cal(model):
    bops_total = torch.Tensor([0]).cuda()

    # except for first & Last layer
    for name, module in model.named_modules():
        if hasattr(module, "quant_func") and hasattr(module, "weight") and module.quant:

            bits_weight = (2 + torch.sigmoid(module.quant_func.bit)*12).round()
            
            if module.act_func is not None :
                bits_act = (2 + torch.sigmoid(module.act_func.quant_func.bit)*12).round()
            else :
                bits_act = 32
            
            if isinstance(module, nn.Conv2d):
                _, _, h, w = module.out_shape
                bop = compute_bops(
                    module.kernel_size[0],
                    module.in_channels,
                    module.out_channels // module.groups, h, w,
                    bits_weight,
                    bits_act,
                    )
            else :
                bop = compute_bops(
                    1,
                    module.in_features,
                    module.out_features, 1, 1,
                    bits_weight,
                    bits_act,
                    )

            bops_total += bop
               
    return bops_total

def bit_cal(model):
    numel_a = 0
    numel_w = 0
    loss_bit_a = 0
    loss_bit_au = 0
    loss_bit_w = 0
    loss_bit_wu = 0

    w_bit=-1
    a_bit=-1
    au_bit=-1
    wu_bit=-1
    for name, module in model.named_modules():
        if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:
            bit = 2 + torch.sigmoid(module.bit)*12
            loss_bit_w += bit * module.weight.numel()
            loss_bit_wu += torch.round(bit) * module.weight.numel()
            numel_w += module.weight.numel()
            
        if hasattr(module, "bit") and hasattr(module, "out_shape") and module.quant:
            bit = 2 + torch.sigmoid(module.bit)*12
            loss_bit_a += bit * np.prod(module.out_shape)
            loss_bit_au += torch.round(bit) * np.prod(module.out_shape)
            numel_a += np.prod(module.out_shape)
        
    if numel_a > 0:
        a_bit = (loss_bit_a / numel_a).item()
        au_bit = (loss_bit_au / numel_a).item()

    if numel_w > 0:
        w_bit = (loss_bit_w / numel_w).item()
        wu_bit = (loss_bit_wu / numel_w).item()
    
    return a_bit, au_bit, w_bit, wu_bit
    

class QuantOps(object):
    initialize = initialize
    hnoise_initilaize = hnoise_initilaize
    sample_activation_size = sample_activation_size
    ReLU = Q_ReLU
    Sym = Q_Sym
    Conv2d = Q_Conv2d
    Linear = Q_Linear
    HSwish = Q_HSwish
    
class QuantActs(object):
    ReLU = Q_ReLU
    Sym = Q_Sym
    HSwish = Q_HSwish