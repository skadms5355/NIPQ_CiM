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
    bpbs = []
    output = []
    bnweight = []
    bnbias = []
    next_scale = []
    bn_input = []
    bn_output = []
    act_output = []

    def psum_forward_hook(module, inputs, outputs):
        sinput.append(module.sinput)
        sweight.append(module.weight_group)
        adc.append(module.adc_list)
        bpbs.append(module.bpbs)
        output.append(module.output)
        psum_scale.append(module.psum_scale)

    def bn_forward_hook(module, inputs, outputs):
        weight = (module.weight/torch.sqrt(module.running_var+module.eps)).detach()
        bias = (module.bias - module.running_mean*weight).detach()
        bnweight.append(weight)
        bnbias.append(bias)
        bn_input.append(inputs[0].detach())
        bn_output.append(outputs.detach())
    
    def scale_forward_hook(module, inputs, outputs):
        next_scale.append(module.s)
        act_output.append(outputs.detach())

    for name, module in model.named_modules():
        if isinstance(module, (PsumQConv)):
            hooks.append(module.register_forward_hook(psum_forward_hook))

        if isinstance(module, (nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(bn_forward_hook))

        if isinstance(module, (Q_act)):
            hooks.append(module.register_forward_hook(scale_forward_hook))

    model.eval()
    model.cuda()
    model(x.cuda())

    layer = 5 # only psum layer (conv)
    max_range = 2**14 - 1 
    min_range = -2**13
    max_hw = 2**15-1
    min_hw= -2**15 
    delta = (max_range-min_range)/(2**4)

    inputs = {}
    weights = {}
    group_outputs = {}
    merge_outputs = {}
    bpbs_outputs = {}
    bn_weight = {}
    bn_bias = {}
    act_weight = {}
    act_bias = {}
    
    for i in range(1, layer+1):
        inputs[i] = sinput[i-1]
        weights[i] = sweight[i-1]
        bpbs_outputs[i] = bpbs[i-1]
        group_outputs[i] = adc[i-1]
        merge_outputs[i] = output[i-1]
        bn_weight[i] = bnweight[i]
        bn_bias[i] = bnbias[i]
        p = (next_scale[i]/(psum_scale[i-1]*bn_weight[i])).detach()
        q = (bn_bias[i]/next_scale[i]).detach()
        m = ((delta / p).detach())
        n = ((q*delta + min_range+ delta/2).detach())
        # min_val = -p*(q+1/2)
        # min_clip[i] = min_val.to(torch.int16)
        # max_clip[i] = ((2**4-1) * next_scale[i]).to(torch.int16)

        # hw_act = torch.round((torch.clip((merge_outputs[i]*m+n), min_range+delta/2, max_range-delta/2)-min_range-delta/2)/delta)
        print('\n\nm max, min', m.max(), m.min())
        print('n max, min', n.max(), n.min())
        n_clip = torch.clip(n, min_hw, max_hw)
        n_clip_int = n_clip.round().to(torch.int16) 
        m_int = m.round().to(torch.int16)
        n_int = n.round().to(torch.int16)
        print('\nm_int max, min', m_int.max(), m_int.min())
        print('n_clip_int max, min', n_clip_int.max(), n_clip_int.min())
        print('n_int max, min', n_int.max(), n_int.min())
        act_weight[i] = m_int
        act_bias[i] = n_int
        m = m.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        n = n.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        m_int = m_int.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        n_int = n_int.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        n_clip_int = n_clip_int.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        y_hw = (merge_outputs[i]*m).to(torch.int32)
        y_hw_clip = torch.clip(y_hw, min_hw, max_hw)
        print('\ny_hw max, min', y_hw.max(), y_hw.min())
        print('y_hw_clip max, min', y_hw_clip.max(), y_hw_clip.min())
        y_hw_bias = torch.clip(y_hw_clip+n_int, min_hw, max_hw).to(torch.int16)
        y_hw_clip_int = torch.clip(y_hw_clip+n_clip_int, min_hw, max_hw).to(torch.int16)
        print('\ny_hw+n max, min', y_hw_bias.max(), y_hw_bias.min())
        print('y_hw+n_int max, min', y_hw_clip_int.max(), y_hw_clip_int.min())

        # results comparision
        sf_act = act_output[i]/next_scale[i]
        hw_act = torch.round((torch.clip(y_hw_bias, min_range+delta/2, max_range-delta/2)-min_range-delta/2)/delta)
        hw_act_int = torch.round((torch.clip(y_hw_clip_int, min_range+delta/2, max_range-delta/2)-min_range-delta/2)/delta)
        # hw_ci = torch.round((torch.clip(y_hw_clip, min_range+delta/2, max_range-delta/2)-min_range-delta/2)/delta)
        if sf_act.shape[2] == hw_act_int.shape[2]:
            index_b=torch.where(sf_act.round().to(torch.uint8) != hw_act.round().to(torch.uint8))
            index=torch.where(sf_act.round().to(torch.uint8) != hw_act_int.round().to(torch.uint8))
            # index_i=torch.where(sf_act.round().to(torch.uint8) != hw_ci.round().to(torch.uint8))

            print(index_b[0].size())
            print(index[0].size())

    paramters = {
        'inputs' : inputs,
        'weights': weights,
        'bpbs_outputs': bpbs_outputs,
        'group_outputs': group_outputs,
        'merge_outputs': merge_outputs,
        'act_weight': act_weight,
        'act_bias': act_bias,
        'bn_weight': bn_weight,
        'bn_bias': bn_bias
    }

    file_path = os.getcwd()+'/vgg9_parameters.pkl'

    # Save the dictionary to the pickle file 
    with open(file_path, 'wb') as f:
        pickle.dump(paramters, f)

    with open(file_path, 'rb') as f:
        load_parameters = pickle.load(f)

    import pdb; pdb.set_trace()

    for hook in hooks:
        hook.remove()
    
    return

