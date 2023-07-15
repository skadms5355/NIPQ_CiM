from re import I
import torch
import torch.nn.functional as F
import torch.nn as nn
from numpy import inf
from .noise_cell import Noise_cell
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
            return x, None
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

class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, wbits=32, kernel_size=3, stride=1, padding=0, groups=1, symmetric=False, bias=True, hwnoise=False):
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
        self.hwnoise = hwnoise
        self.quan_w_fn = LSQReturnScale(bit=self.wbits, half_range=False, symmetric=symmetric, per_channel=False)

    def hwnoise_init(self, cbits, mapping_mode, co_noise=0.01, noise_type='prop', res_val='rel', max_epoch=-1):
        w_format = 'state' if res_val == 'abs' or noise_type == 'meas' else 'weight'
        self.noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type=noise_type, \
                                    res_val=res_val, w_format=w_format, max_epoch=max_epoch)
        self.mapping_mode = mapping_mode
        self.res_val = res_val
        self.noise_type = noise_type
        if noise_type == 'interp':
            w_format = 'state'
            res_val = 'abs'
        elif not (noise_type == 'prop' or 'interp'):
            noise_type = 'prop'
        self.inf_noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type=noise_type, \
                                    res_val=res_val, w_format=w_format, max_epoch=max_epoch)
        self.qweight_noise = None

    def forward(self, input):

        if self.wbits == 32:
            return F.conv2d(input, self.weight, bias=self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            
        qweight, scale = self.quan_w_fn(self.weight)

        with torch.no_grad():
            if self.hwnoise:
                if self.training:
                    if self.qweight_noise is not None:
                        self.qweight_noise = None

                    if self.noise_type == "interp" and self.res_val == 'abs':
                        if self.mapping_mode == '2T2R':
                            pqweight = torch.where(qweight>0, qweight, 0)
                            nqweight = torch.where(qweight<0, abs(qweight), 0)
                            cat_weight = torch.cat((pqweight, nqweight))
                            cat_weight = self.inf_noise_cell((cat_weight / scale).round()) * scale
                            split_weight = torch.chunk(cat_weight, 2)
                            qweight_noise = split_weight[0] - split_weight[1]
                        else:
                            assert False, "Check noise type and res_val at training condition"
                    else:
                        qweight_noise = self.noise_cell((qweight / scale).round()) * scale
                else:
                    if self.qweight_noise is None:
                        if self.noise_type == "interp":
                            if self.mapping_mode == '2T2R':
                                pqweight = torch.where(qweight>0, qweight, 0)
                                nqweight = torch.where(qweight<0, abs(qweight), 0)
                                cat_weight = torch.cat((pqweight, nqweight))
                                cat_weight = self.inf_noise_cell((cat_weight / scale).round()) * scale
                                split_weight = torch.chunk(cat_weight, 2)
                                qweight_noise = split_weight[0] - split_weight[1]
                            elif 'ref' in self.mapping_mode:
                                shift_v = 2**(self.wbits-1)
                                shift_w = shift_v*torch.ones(qweight.size()).cuda()
                                value_w = torch.add((qweight/scale), shift_v)
                                cat_weight = torch.cat((value_w, shift_w))
                                cat_weight = self.inf_noise_cell(cat_weight.round_()) * scale
                                split_weight = torch.chunk(cat_weight, 2)
                                qweight_noise = split_weight[0] - split_weight[1]
                            else: 
                                assert False, "Only support 2T2R mapping mode"
                        else:
                            qweight_noise = self.inf_noise_cell((qweight / scale).round()) * scale
                        self.qweight_noise = qweight_noise
                    else:
                        qweight_noise = self.qweight_noise

                if self.training:
                    qweight.copy_(qweight_noise)
                else:
                    qweight = qweight_noise

        output = F.conv2d(input, qweight, bias=self.bias,
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
    def __init__(self, in_features, out_features, wbits, symmetric=False, bias=False, hwnoise=False):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)
        self.wbits = wbits
        #self.weight_clip_scale = nn.Parameter(torch.Tensor([self.boundary]))
        #self.weight_clip_scale = nn.Parameter(torch.zeros(self.quant_group, 1, 1, 1))
        #self.weight_clip_scale.data.fill_(self.boundary)
        self.hwnoise = hwnoise
        self.quan_w_fn = LSQReturnScale(bit=self.wbits, half_range=False, symmetric=symmetric, per_channel=False)

    def hwnoise_init(self, cbits, mapping_mode, co_noise=0.01, noise_type='prop', res_val='rel', max_epoch=-1):
        w_format = 'state' if res_val == 'abs' or noise_type == 'meas' else 'weight'
        self.noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type=noise_type, \
                                    res_val=res_val, w_format=w_format, max_epoch=max_epoch)
        self.mapping_mode = mapping_mode
        self.res_val = res_val
        self.noise_type = noise_type
        if noise_type == 'interp':
            w_format = 'state'
            res_val = 'abs'
        elif not (noise_type == 'prop' or 'interp'):
            noise_type = 'prop'
        self.inf_noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type=noise_type, \
                                    res_val=res_val, w_format=w_format, max_epoch=max_epoch)
        self.qweight_noise = None
        
    def forward(self, input):

        if self.wbits == 32:
            return F.linear(input, self.weight, bias=self.bias)
        
        qweight, scale = self.quan_w_fn(self.weight)

        with torch.no_grad():
            if self.hwnoise:
                if self.training:
                    if self.qweight_noise is not None:
                        self.qweight_noise = None

                    if self.noise_type == "interp" and self.res_val == 'abs':
                        pqweight = torch.where(qweight>0, qweight, 0)
                        nqweight = torch.where(qweight<0, abs(qweight), 0)
                        cat_weight = torch.cat((pqweight, nqweight))
                        cat_weight = self.inf_noise_cell((cat_weight / scale).round()) * scale
                        split_weight = torch.chunk(cat_weight, 2)
                        qweight_noise = split_weight[0] - split_weight[1]
                    else:
                        qweight_noise = self.noise_cell((qweight / scale).round()) * scale
                else:
                    if self.qweight_noise is None:
                        if self.noise_type == "interp":
                            if self.mapping_mode == '2T2R':
                                pqweight = torch.where(qweight>0, qweight, 0)
                                nqweight = torch.where(qweight<0, abs(qweight), 0)
                                cat_weight = torch.cat((pqweight, nqweight))
                                cat_weight = self.inf_noise_cell((cat_weight / scale).round()) * scale
                                split_weight = torch.chunk(cat_weight, 2)
                                qweight_noise = split_weight[0] - split_weight[1]
                            elif 'ref' in self.mapping_mode:
                                shift_v = 2**(self.wbits-1)
                                shift_w = shift_v*torch.ones(qweight.size()).cuda()
                                value_w = torch.add((qweight/scale), shift_v)
                                cat_weight = torch.cat((value_w, shift_w))
                                cat_weight = self.inf_noise_cell(cat_weight.round_()) * scale
                                split_weight = torch.chunk(cat_weight, 2)
                                qweight_noise = split_weight[0] - split_weight[1]
                            else:
                                assert False, "Only support 2T2R mapping mode"
                        else:
                            qweight_noise = self.inf_noise_cell((qweight / scale).round()) * scale
                        self.qweight_noise = qweight_noise
                    else:
                        qweight_noise = self.qweight_noise

                if self.training:
                    qweight.copy_(qweight_noise)
                else:
                    qweight = qweight_noise

        output =  F.linear(input, qweight, bias=self.bias)
        return output

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        return 'in_features={}, out_features={}, bias={}, wbits={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.wbits
        )

class Q_act(LSQ):
    def __init__(self, abits, half_range=True, per_channel=False, bitserial=False):
        super().__init__(bit=abits, half_range=True, per_channel=False, bitserial=bitserial)

def add_act(abits, bitserial=False, mode='signed', ste='hardtanh', offset=0, width=1):
    if abits == 32:
        return nn.ReLU(inplace=True)
    elif abits == 1:
        return BinAct(abits=abits, mode=mode, ste=ste, offset=offset, width=width)
    else:
        return Q_act(abits=abits, bitserial=bitserial)

def hwnoise_initialize(model, hwnoise=True, cbits=4, mapping_mode=None, co_noise=0.01, noise_type='prop', res_val='rel', max_epoch=-1):
    for name, module in model.named_modules():
        if isinstance(module, (QConv)) and hwnoise:
            if module.wbits != 32 and (module.in_channels != 3):
                module.hwnoise = True

                if noise_type == 'grad':
                    assert max_epoch != -1, "Enter max_epoch in hwnoise_initialize function"
                if hwnoise:
                    module.hwnoise_init(cbits=cbits, mapping_mode=mapping_mode, co_noise=co_noise, noise_type=noise_type, res_val=res_val, max_epoch=max_epoch)
        
        if isinstance(module, (QLinear)) and hwnoise:
            if module.wbits != 32:
                module.hwnoise = True

                if noise_type == 'grad':
                    assert max_epoch != -1, "Enter max_epoch in hwnoise_initialize function"
                if hwnoise:
                    module.hwnoise_init(cbits=cbits, mapping_mode=mapping_mode, co_noise=co_noise, noise_type=noise_type, res_val=res_val, max_epoch=max_epoch)

def quant_or_not(abits):
    if abits == 32:
        return nn.Identity()
    else:
        return LSQ(bit=abits, half_range=False, per_channel=False)