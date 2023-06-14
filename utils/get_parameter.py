import torch
import torch.nn as nn
import pickle
from models.psum_modules import *
from models.quantized_lsq_modules import *

def get_parameter(model, x):
    # forward hook for input and output data 
    hooks = []
    sinput = []
    sweight = []
    psum_scale = []
    adc = []
    output = []
    bn_weight = []
    bn_bias = []
    next_scale = []

    def psum_forward_hook(module, inputs, outputs):
        sinput.append(module.sinput)
        sweight.append(module.weight_group)
        adc.append(module.adc_list)
        output.append(module.output)
        psum_scale.append(module.psum_scale)

    def bn_forward_hook(module, inputs, outputs):
        bn_weight.append(module.weight)
        bn_bias.append(module.bias)
    
    def scale_forward_hook(module, inputs, outputs):
        next_scale.append(module.s)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d)):
            print('batch', module)
            hooks.append(module.register_forward_hook(bn_forward_hook))

        if isinstance(module, (PsumQConv)):
            print('weight, s', module)
            hooks.append(module.register_forward_hook(psum_forward_hook))

        if isinstance(module, (Q_act)):
            print('s', module)
            hooks.append(module.register_forward_hook(scale_forward_hook))

    model.eval()
    model.cuda()
    model(x.cuda())
    import pdb; pdb.set_trace()

    for hook in hooks:
        hook.remove()
    
    return

