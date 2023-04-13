import numpy as np
import torch
import math
import os
import copy
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils.padding as Pad
from .bitserial_modules import *
from .split_modules import *
from .quantized_basic_modules import psum_quant_merge, psum_quant
from .nipq_quantization_module import QuantActs, Quantizer
# custom kernel
import conv_sweight_cuda

"""
    This module does not support training mode in nipq quantization
    So, nipq quantization noise + hardware noise are not used.
"""

# accurate mode max thresuld (included)
MAXThres = 7

# split convolution layer across input channel
def split_conv(weight, nWL):
    nIC = weight.shape[1]
    nWH = weight.shape[2]*weight.shape[3]
    nMem = int(math.ceil(nIC/math.floor(nWL/nWH)))
    nIC_list = [int(math.floor(nIC/nMem)) for _ in range(nMem)]
    for idx in range((nIC-nIC_list[0]*nMem)):
        nIC_list[idx] += 1

    return nIC_list

# split fully connected layer across input channel
def split_linear(weight, nWL):
    nIF = weight.shape[1] #in_features
    nMem = int(math.ceil(nIF/nWL))
    nIF_list = [int(math.floor(nIF/nMem)) for _ in range(nMem)]
    for idx in range((nIF-nIF_list[0]*nMem)):
        nIF_list[idx] += 1

    return nIF_list

def calculate_groups(arraySize, fan_in):
    if arraySize > 0:
        # groups
        groups = int(np.ceil(fan_in / arraySize))
        while fan_in % groups != 0:
            groups += 1
    else:
        groups = 1

    return groups

class Psum_QConv2d(SplitConv):
    """
        Quant(Nipq) Conv + Psum quantization
    """
    def __init__(self, *args, act_func=None, padding=0, padding_mode='zeros', **kargs):
        super(Psum_QConv2d, self).__init__(*args,  **kargs)
        self.act_func = act_func
        self.padding = padding
        self.padding_mode = padding_mode
        
        if self.padding_mode == 'zeros':
            self.padding_value = 0
        elif self.padding_mode == 'ones':
            self.padding_value = 1
        elif self.padding_mode == 'alter':
            self.padding_value = 0

        self.quant_func = Quantizer(sym=True, noise=False, offset=0, is_stochastic=True, is_discretize=True)
        self.bits = self.quant_func.get_bit()
        self.hwnoise = False

        ## for psum quantization
        self.mapping_mode = '2T2R' # Array mapping method [2T2R, ref_a]]
        self.wbit_serial = None
        self.cbits = 4  # only support one weight in one cell or SLC
        self.psum_mode = None
        self.pclipmode = 'layer'
        self.pbits = 32
        # for scan version
        self.pstep = None
        self.pzero = None  # contain zero value (True)
        self.center = None
        self.pbound = None
        # for sigma version
        self.pclip = None
        self.psigma = None

        ## for accurate mode (SRAM)
        self.accurate = False

        # for logging
        self.bitserial_log = False
        self.layer_idx = -1
        self.checkpoint = None
        self.info_print = True

    def model_split_groups(self, arraySize):
        self.arraySize = arraySize
        self.split_groups = calculate_groups(arraySize, self.fan_in)
        if self.fan_in % self.split_groups != 0:
            raise ValueError('fan_in must be divisible by groups')
        self.group_fan_in = int(self.fan_in / self.split_groups)
        self.group_in_channels = int(np.ceil(self.in_channels / self.split_groups))
        residual = self.group_fan_in % self.kSpatial
        if residual != 0:
            if self.kSpatial % residual != 0:
                self.group_in_channels += 1
        ## log move group for masking & group convolution
        self.group_move_in_channels = torch.zeros(self.split_groups-1, dtype=torch.int)
        group_in_offset = torch.zeros(self.split_groups, dtype=torch.int).to(self.weight.device)
        self.register_buffer('group_in_offset', group_in_offset)
        ## get group conv info
        self._group_move_offset()

        # sweight
        sweight = torch.Tensor(self.out_channels*self.split_groups, self.group_in_channels, self.kernel_size[0], self.kernel_size[1]).to(self.weight.device)
        self.register_buffer('sweight', sweight)

    def setting_pquant_func(self, pbits=None, center=[], pbound=None):
        # setting options for pquant func
        if pbits is not None:
            self.pbits = pbits
        if pbound is not None:
            self.pbound = pbound
            # get pquant step size
            self.pstep = 2 * self.pbound / ((2.**self.pbits) - 1)

        if (self.mapping_mode == 'two_com') or (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
            self.pzero = False
        else:
            self.pzero = True

        # get half of pquant levels
        self.phalf_num_levels = 2.**(self.pbits-1)

        # get center value
        self.setting_center(center=center)

    def setting_center(self, center=[]):
        if self.pzero:
            self.center = 0 # center is zero value in 2T2R mode
        else:
        # centor's role is offset in two_com mode or ref_d mode
            self.center = center

    def reset_layer(self, wbit_serial=None, groups=None, 
                        pbits=None, pbound=None, center=[]):
        if wbit_serial is not None:
            self.wbit_serial = wbit_serial

        self.setting_pquant_func(pbits=pbits, center=center, pbound=pbound)

    def print_ratio(self, bits, input, weight, graph=False):
        # LSB > MSB 
        if graph:
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 6))
            ax = axes.flatten()
        write_file = f'{self.checkpoint}/model_ratio1.txt'

        if os.path.isfile(write_file) and (self.layer_idx == 0):
            option = 'w'
        else:
            option = 'a'

        with open(write_file, option) as file:
            if self.layer_idx == 0:
                file.write(f'Input & Weight ratio 1\n')
            file.write(f'\nConv Layer {self.layer_idx}')
                
            for b in range(bits):
                file.write(f'\nGroup  Mean Input:{b}    Min Input:{b}     Max Input:{b}    Mean Weight:{b}   Min Weight:{b}     Max Weight:{b}\n')

                #split_weight
                split_weight = weight[b].chunk(self.split_groups, dim=0)
                
                input_channel = 0
                for i in range(0, self.split_groups):
                    split_input = input[b].narrow(1, input_channel, self.group_in_channels) / 2**b
                    w_one = torch.ones(size=split_weight[i].size(), device=split_input.device)
                    weight_group = split_weight[i].reshape(split_weight[i].shape[0], -1) # ratio of output channels
                    
                    sum_weight = torch.sum(weight_group, axis=1)
                    ratio_w = (sum_weight/self.arraySize)*100
                    sum_input = F.conv2d(split_input, w_one, bias=None, stride=self.stride, padding=0, dilation=self.dilation)
                    ratio_i = torch.mean((sum_input/self.arraySize)*100, dim=1)
                    
                    if i == 0:
                        group_w = ratio_w
                        group_i = ratio_i
                    else:
                        group_w += ratio_w
                        group_i += ratio_i

                    file.write(f'Group_{i}  {ratio_i.mean()}    {ratio_i.min()}     {ratio_i.max()}     {ratio_w.mean()}    {ratio_w.min()}     {ratio_w.max()}\n')
                    # prepare for the next stage
                    if i < self.split_groups - 1:
                        input_channel += self.group_move_in_channels[i]
                if graph:
                    colors=sns.color_palette("husl", bits)
                    sns.histplot(data=(group_i/self.split_groups).cpu().reshape(-1), bins=100, linewidth=0, alpha=0.7, ax=ax[b], color=colors[b], stat="count")
                    sns.histplot(data=(group_w/self.split_groups).cpu().reshape(-1), bins=100, linewidth=0, alpha=0.7, ax=ax[b+4], color=colors[b], stat="count")
                    ax[b].set_title('Input {}'.format(b), fontsize=15, fontdict=dict(weight='bold'))
                    ax[b+4].set_title('Weight {}'.format(b), fontsize=15, fontdict=dict(weight='bold'))
                    ax[b].set_xlabel('Ratio [%]', fontsize=13)
                    ax[b+4].set_xlabel('Ratio [%]', fontsize=13)
                    fig.tight_layout(pad=1.0, h_pad=2)
                    plt.savefig(f'{self.checkpoint}/Ratio_layer{self.layer_idx}.png')
        
    def bitserial_split(self, input, act=True):
        """
            input: [batch, channel, H, W]
            output: [batch, abits * channel, H, W]
        """
        bits = self.bits
        if act:
            scale = self.act_func.quant_func.get_alpha()
        else: 
            scale = 1
        int_input = input / scale  # remove remainder value ex).9999 
        
        output_dtype = int_input.round_().dtype
        output_uint8= int_input.to(torch.uint8)
        # bitserial_step = 1 / (2.**(bits - 1.))

        output = output_uint8 & 1
        for i in range(1, bits):
            out_tmp = output_uint8 & (1 << i)
            output = torch.cat((output, out_tmp), 1)
        output = output.to(output_dtype)
        # output.mul_(bitserial_step) ## for preventing overflow

        return output.round_(), scale
    
    def _bitserial_log_forward(self, input):
        print(f'[layer{self.layer_idx}]: bitserial mac log')
        # delete padding_shpe & additional padding operation by matching padding/stride format with nn.Conv2d
        if self.padding > 0:
            padding_shape = (self.padding, self.padding, self.padding, self.padding)
            input = Pad.pad(input, padding_shape, self.padding_mode, self.padding_value)

        # local parameter settings
        bitplane_idx = 0

        with torch.no_grad():
            # get quantization parameter and input bitserial 
            bits = self.bits
            sinput, a_scale = self.bitserial_split(input)
            w_serial = True if self.cbits==1 and self.mapping_mode == 'two_com' else False
            qweight = self.quant_func(self.weight, self.training, serial=w_serial)
            w_scale = self.quant_func.get_alpha()

            ## get dataframe
            logger = f'{self.checkpoint}/layer{self.layer_idx}_mac_static.pkl'
            df = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

            layer_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
            network_hist = f'{self.checkpoint}/hist/network_hist.pkl'

            #plane hist
            
            ### in-mem computation mimic (split conv & psum quant/merge)
            input_chunk = torch.chunk(sinput, bits, dim=1)
            self.sweight = conv_sweight_cuda.forward(self.sweight, qweight / w_scale, self.group_in_offset, self.split_groups)
            if w_serial:
                sweight, _ = self.bitserial_split(self.sweight, act=False) 
                sweight = torch.where(sweight>0, 1., 0.)
                if self.hwnoise:
                    sweight = self.quant_func.noise_cell(sweight)
                    std_offset = self.quant_func.noise_cell.get_offset()
                wsplit_num = int(self.bits / self.cbits)
            else:
                sweight = self.sweight
                wsplit_num = 1
            weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)
            if w_serial:
                w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

            psum_scale = w_scale * a_scale 

            out_tmp = None
            layer_hist_dict = {}
            for abit, input_s in enumerate(input_chunk):
                abitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_a:{abit}_hist.pkl'
                a_hist_dict = {}
                if w_serial and self.hwnoise:
                    out_one = std_offset * self._split_forward(input_s, w_one, padded=True, ignore_bias=True,
                                                weight_is_split=True, infer_only=True, merge_group=True)
                for wbit, weight_s in enumerate(weight_chunk):
                    wabitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_w:{wbit}_a:{abit}_hist.pkl'
                    wa_hist_dict = {}
                    mag = 2**(abit)
                    out_tmp = self._split_forward(input_s, weight_s, padded=True, ignore_bias=True,
                                                    weight_is_split=True, infer_only=True) 

                    out_array = out_tmp.round() / mag # noise bound set to round function
                    ## NOTE
                    df.loc[bitplane_idx] = [wbit, abit,
                                                    float(out_array.mean()), 
                                                    float(out_array.std()), 
                                                    float(out_array.min()), 
                                                    float(out_array.max())] 

                    out_min = out_array.min()
                    out_max = out_array.max()

                    # update hist
                    for val in range(int(out_min), int(out_max)+1):
                        count = out_array.eq(val).sum().item()
                        # get wa_hist
                        wa_hist_dict[val] = count
                        # get w_hist
                        if val in a_hist_dict.keys():
                            a_hist_dict[val] += count
                        else:
                            a_hist_dict[val] = count

                    # save wabitplane_hist
                    df_hist = pd.DataFrame(list(wa_hist_dict.items()), columns = ['val', 'count'])
                    # wabitplane hist
                    if os.path.isfile(wabitplane_hist):
                        print(f'[{self.layer_idx}] Update wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
                        df_wabitplane_hist = pd.read_pickle(wabitplane_hist) 
                        df_merge = pd.merge(df_wabitplane_hist, df_hist, how="outer", on="val")
                        df_merge = df_merge.replace(np.nan, 0)
                        df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                        df_merge = df_merge[['val', 'count']]
                        df_merge.to_pickle(wabitplane_hist)
                    else:
                        print(f'[{self.layer_idx}] Create wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
                        df_hist.to_pickle(wabitplane_hist)

                    # split output merge
                    output_chunk = out_tmp.chunk(self.split_groups, dim=1)
                    for g in range(0, self.split_groups):
                        if g==0:
                            out_tmp = output_chunk[g]
                        else:
                            out_tmp += output_chunk[g]
                    
                    if w_serial:
                        out_tmp = 2**(wbit)*(out_tmp - out_one) if self.hwnoise else 2**(wbit)*(out_tmp)
                        if wsplit_num == wbit+1:
                            out_wsum -= out_tmp
                        else:
                            out_wsum = out_tmp if wbit == 0 else out_wsum + out_tmp
                    else:
                        out_wsum = out_tmp

                    bitplane_idx += 1

                # save abitplane_hist
                df_hist = pd.DataFrame(list(a_hist_dict.items()), columns = ['val', 'count'])
                # wbitplane hist
                if os.path.isfile(abitplane_hist):
                    print(f'[{self.layer_idx}] Update abitplane_hist for a:{abit} ({abitplane_hist})')
                    df_abitplane_hist = pd.read_pickle(abitplane_hist) 
                    df_merge = pd.merge(df_abitplane_hist, df_hist, how="outer", on="val")
                    df_merge = df_merge.replace(np.nan, 0)
                    df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                    df_merge = df_merge[['val', 'count']]
                    df_merge.to_pickle(abitplane_hist)
                else:
                    print(f'[{self.layer_idx}] Create abitplane_hist for a:{abit} ({abitplane_hist})')
                    df_hist.to_pickle(abitplane_hist)

                # update layer hist
                for val, count in a_hist_dict.items():
                    if val in layer_hist_dict.keys():
                        layer_hist_dict[val] += count
                    else:
                        layer_hist_dict[val] = count
                
                output = out_wsum if abit == 0 else output + out_wsum

            # restore output's scale
            output = output * psum_scale

            # add bias
            if self.bias is not None:
                output += self.bias

        # save logger
        df.to_pickle(logger)
        # df_scaled.to_pickle(logger_scaled)

        # save hist
        df_hist = pd.DataFrame(list(layer_hist_dict.items()), columns = ['val', 'count'])
        # layer hist
        if os.path.isfile(layer_hist):
            print(f'[{self.layer_idx}] Update layer_hist ({layer_hist})')
            df_layer_hist = pd.read_pickle(layer_hist) 
            df_merge = pd.merge(df_layer_hist, df_hist, how="outer", on="val")
            df_merge = df_merge.replace(np.nan, 0)
            df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
            df_merge = df_merge[['val', 'count']]
            df_merge.to_pickle(layer_hist)
        else:
            print(f'[{self.layer_idx}] Create layer_hist ({layer_hist})')
            df_hist.to_pickle(layer_hist)
        # network hist
        if os.path.isfile(network_hist):
            print(f'[{self.layer_idx}] Update network_hist ({network_hist})')
            df_network_hist = pd.read_pickle(network_hist) 
            df_merge = pd.merge(df_network_hist, df_hist, how="outer", on="val")
            df_merge = df_merge.replace(np.nan, 0)
            df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
            df_merge = df_merge[['val', 'count']]
            df_merge.to_pickle(network_hist)
        else:
            print(f'[{self.layer_idx}] Create network_hist ({network_hist})')
            df_hist.to_pickle(network_hist)

        # output_real = F.conv2d(input, qweight, bias=self.bias,
        #                     stride=self.stride, dilation=self.dilation, groups=self.groups)
        # import pdb; pdb.set_trace()

        return output

    def _ADC_clamp_value(self):
        # get ADC clipping value for hist [Layer or Network hist]
        if self.psum_mode == 'sigma':
            if self.pclipmode == 'layer':
                phist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
                # phist = f'./hist/layer{self.layer_idx}_hist.pkl'
            elif self.pclipmode == 'network':
                phist = f'{self.checkpoint}/hist/network_hist.pkl'

            if os.path.isfile(phist):
                # print(f'Load ADC_hist ({phist})')
                df_hist = pd.read_pickle(phist)
                mean, std, min, max = get_statistics_from_hist(df_hist)
            else:
                if self.pbits != 32:
                    assert False, "Error: Don't have ADC hist file"
                else:
                    mean, std, min, max = 0.0, 0.0, 0.0, 0.0

            # Why abs(mean) is used not mean?? => Asymmetric quantizaion is occured
            if self.pbits == 32:
                maxVal = 1
                minVal = 0
            else:
                if self.pclip == 'max':
                    maxVal = max
                    minVal = min
                else:
                    maxVal =  (abs(mean) + self.psigma*std).round() 
                    minVal = (abs(mean) - self.psigma*std).round()
                    if (self.mapping_mode == 'two_com') or (self.mapping_mode =='ref_d') or (self.mapping_mode == 'PN'):
                        minVal = min if minVal < 0 else minVal
        else:
            if self.mapping_mode == 'two_com':
                minVal = 0
                if self.pclip == 'max':
                    maxVal = self.arraySize - 1
                elif self.pclip == 'half':
                    maxVal = int(self.arraySize / 2) - 1
                elif self.pclip == 'quarter':
                    maxVal = int(self.arraySize / 4) - 1 
                else:
                    assert False, 'Do not support this clip range {}'.format(self.pclip)
            else:
                assert False, 'Psum mode fix only support two_com mapping mode'

        midVal = (maxVal + minVal) / 2

        if self.info_print:
            if self.psum_mode == 'sigma':
                write_file = f'{self.checkpoint}/Layer_clipping_range.txt'
                if os.path.isfile(write_file) and (self.layer_idx == 0):
                    option = 'w'
                else:
                    option = 'a'
                with open(write_file, option) as file:
                    if self.layer_idx == 0:
                        file.write(f'{self.pclipmode}-wise Mode Psum quantization \n')
                        file.write(f'Layer_information  Mean    Std     Min Max Clip_Min    Clip_Max    Mid \n')
                    file.write(f'Layer{self.layer_idx}  {mean}  {std}   {min}   {max}   {minVal}    {maxVal}    {midVal}\n')
            
            print(f'{self.pclipmode}-wise Mode Psum quantization')
            if self.pbits == 32:
                print(f'Layer{self.layer_idx} information | pbits {self.pbits}')
            else:
                if self.psum_mode == 'sigma':
                    print(f'Layer{self.layer_idx} information | pbits {self.pbits} | Mean: {mean} | Std: {std} | Min: {min} | Max: {max} | Clip Min: {minVal} | Clip Max: {maxVal} | Mid: {midVal}')
                elif self.psum_mode == 'fix':
                    print(f'Layer{self.layer_idx} information | pbits {self.pbits} | Clip Min: {minVal} | Clip Max: {maxVal} | Mid: {midVal}')
            self.info_print = False

        return minVal, maxVal, midVal 

    def _bitserial_comp_forward(self, input):
        # delete padding_shpe & additional padding operation by matching padding/stride format with nn.Conv2d
        if self.padding > 0:
            padding_shape = (self.padding, self.padding, self.padding, self.padding)
            input = Pad.pad(input, padding_shape, self.padding_mode, self.padding_value)

        # get quantization parameter and input bitserial 
        bits = self.bits
        w_serial = True if self.cbits==1 and self.mapping_mode == 'two_com' else False
        qweight = self.quant_func(self.weight, self.training, serial=w_serial)

        if self.wbit_serial:
            with torch.no_grad():
                w_scale = self.quant_func.get_alpha()
                sinput, a_scale = self.bitserial_split(input)

                ### in-mem computation mimic (split conv & psum quant/merge)
                input_chunk = torch.chunk(sinput, bits, dim=1)
                self.sweight = conv_sweight_cuda.forward(self.sweight, qweight/w_scale, self.group_in_offset, self.split_groups)
                if w_serial:
                    sweight, _ = self.bitserial_split(self.sweight, act=False) 
                    sweight = torch.where(sweight>0, 1., 0.)
                    if self.hwnoise:
                        sweight = self.quant_func.noise_cell(sweight)
                        std_offset = self.quant_func.noise_cell.get_offset()
                    wsplit_num = int(self.bits / self.cbits)
                else:
                    sweight = self.sweight
                    wsplit_num = 1
                weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)

                if w_serial:
                    w_one = torch.ones(size=weight_chunk[0].size(), device=weight_chunk[0].device)

                if self.info_print:
                    self.print_ratio(bits, input_chunk, weight_chunk)

                psum_scale = w_scale * a_scale

                if self.psum_mode == 'sigma' or 'fix':
                    minVal, maxVal, midVal = self._ADC_clamp_value()
                    self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
                elif self.psum_mode == 'scan':
                    if self.info_print:
                        self.info_print = False
                    pass
                else:
                    assert False, 'This script does not support {self.psum_mode}'

                for abit, input_s in enumerate(input_chunk):
                    a_mag = 2**(abit)
                    if w_serial and self.hwnoise:
                        out_one = std_offset * self._split_forward(input_s, w_one, padded=True, ignore_bias=True, cat_output=False,
                                                    weight_is_split=True, infer_only=True, merge_group=True)
                    if w_serial and self.accurate:
                        cnt_input = self._split_forward(input_s / a_mag, w_one, padded=True, ignore_bias=True, cat_output=False, weight_is_split=True, infer_only=True)

                    for wbit, weight_s in enumerate(weight_chunk):
                        out_tmp = self._split_forward(input_s, weight_s, padded=True, ignore_bias=True, cat_output=False,
                                                weight_is_split=True, infer_only=True)

                        if not self.accurate:
                            out_adc = None
                            out_adc = psum_quant_merge(out_adc, out_tmp,
                                                        pbits=self.pbits, step=self.pstep, 
                                                        half_num_levels=self.phalf_num_levels, 
                                                        pbound=self.pbound, center=self.center, weight=a_mag,
                                                        groups=self.split_groups, pzero=self.pzero)
                        else:
                            # accurate mode (SRAM)
                            out_quant = None
                            out_quant = psum_quant(out_quant, out_tmp,
                                                        pbits=self.pbits, step=self.pstep, 
                                                        half_num_levels=self.phalf_num_levels, 
                                                        pbound=self.pbound, center=self.center, weight=a_mag,
                                                        groups=self.split_groups, pzero=self.pzero)
                            
                            for g in range(0, self.split_groups):
                                accin_addr = torch.where(cnt_input[g] <= MAXThres)
                                out_quant[g][accin_addr] = out_tmp[g][accin_addr]
                                if g==0:
                                    out_adc = out_quant[g]
                                else:
                                    out_adc += out_quant[g]

                        if w_serial:
                            out_adc = 2**(wbit) * (out_adc - out_one) if self.hwnoise else 2**(wbit) * out_adc
                            if wsplit_num == wbit+1:
                                out_wsum -= out_adc
                            else:
                                out_wsum = out_adc if wbit == 0 else out_wsum + out_adc
                        else:
                            out_wsum = out_adc
                    output = out_wsum if abit == 0 else output+out_wsum

                # restore output's scale
                output = output * psum_scale
        else:
            # no serial computation with psum computation
            output = self._split_forward(input, qweight, padded=True, ignore_bias=True, merge_group=True)

        # add bias
        if self.bias is not None:
            output += self.bias

        # output_real = F.conv2d(input, qweight, bias=self.bias,
        #                         stride=self.stride, dilation=self.dilation, groups=self.groups)
        # import pdb; pdb.set_trace()

        return output

    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
    
        if self.bitserial_log:
            return self._bitserial_log_forward(x)
        else:
            return self._bitserial_comp_forward(x)

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
        # s += ', bits={bits}, wbit_serial={wbit_serial}'
        # s += ', split_groups={split_groups}, mapping_mode={mapping_mode}, cbits={cbits}'
        # s += ', psum_mode={psum_mode}, pbits={pbits}, pbound={pbound}'
        # s += ', bitserial_log={bitserial_log}, layer_idx={layer_idx}'            
        return s.format(**self.__dict__)

class Psum_QLinear(SplitLinear):
    """
        Quant(LSQ)Linear + Psum quantization
    """
    def __init__(self, *args, act_func=None, **kargs):
        super(Psum_QLinear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.hwnoise = False

        self.quant_func = Quantizer(sym=True, noise=False, offset=0, is_stochastic=True, is_discretize=True)
        self.bits = self.quant_func.get_bit()

        # for psum quantization
        self.mapping_mode = '2T2R' # Array mapping method [2T2R, ref_a]
        self.wbit_serial = None
        self.cbits = 4
        self.psum_mode = None
        self.pclipmode = 'layer'
        self.pbits = 32
        # for scan version
        self.pstep = None
        self.pzero = None # contain zero value (True)
        self.center = None
        self.pbound = None
        # for sigma version
        self.pclip = None
        self.psigma = None

        ## for accurate mode (SRAM)
        self.accurate = False

        # for logging
        self.bitserial_log = False
        self.layer_idx = -1
        self.checkpoint = None
        self.info_print = True

    def model_split_groups(self, arraySize=0):
        self.arraySize = arraySize
        self.split_groups = calculate_groups(arraySize, self.in_features)
        if self.in_features % self.split_groups != 0:
            raise ValueError('in_features must be divisible by groups')
        self.group_in_features = int(self.in_features / self.split_groups)

    def setting_pquant_func(self, pbits=None, center=[], pbound=None):
        # setting options for pquant func
        if pbits is not None:
            self.pbits = pbits
        if pbound is not None:
            self.pbound = pbound
            # get pquant step size
            self.pstep = 2 * self.pbound / ((2.**self.pbits) - 1)

        if (self.mapping_mode == 'two_com') or (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
            self.pzero = False
        else:
            self.pzero = True

        # get half of pquant levels
        self.phalf_num_levels = 2.**(self.pbits-1)

        # get center value
        self.setting_center(center=center)
    
    def setting_center(self, center=[]):
        if self.pzero:
            self.center = 0 # center is zero value in 2T2R mode
        else:
        # centor's role is offset in two_com mode or ref_d mode
            self.center = center
    
    def reset_layer(self, wbit_serial=None, groups=None, 
                        pbits=None, pbound=None, center=[]):
        if wbit_serial is not None:
            self.wbit_serial = wbit_serial

        self.setting_pquant_func(pbits=pbits, center=center, pbound=pbound)
    
    def bitserial_split(self, input, act=True):
        """
            input: [batch, channel, H, W]
            output: [batch, abits * channel, H, W]
        """
        bits = self.bits
        if act:
            scale = self.act_func.quant_func.get_alpha()
        else: 
            scale = 1
        int_input = input / scale  # remove remainder value ex).9999 
        
        output_dtype = int_input.round_().dtype
        output_uint8= int_input.to(torch.uint8)
        # bitserial_step = 1 / (2.**(bits - 1.))

        output = output_uint8 & 1
        for i in range(1, bits):
            out_tmp = output_uint8 & (1 << i)
            output = torch.cat((output, out_tmp), 1)
        output = output.to(output_dtype)
        # output.mul_(bitserial_step) ## for preventing overflow

        return output.round_(), scale
    
    def print_ratio(self, bits, input, weight, graph=False):
        # LSB > MSB 
        
        if graph:
            fig, axes = plt.subplots(nrows=2, ncols=bits, figsize=(18, 6))
            ax = axes.flatten()
        write_file = f'{self.checkpoint}/model_ratio1.txt'

        if os.path.isfile(write_file) and (self.layer_idx == 0):
            option = 'w'
        else:
            option = 'a'

        with open(write_file, option) as file:
            if self.layer_idx == 0:
                file.write(f'Input & Weight ratio 1\n')
                
            file.write(f'\nFC Layer {self.layer_idx}')
            for b in range(bits):
                file.write(f'\nGroup  Mean Input:{b}    Min Input:{b}     Max Input:{b}    Mean Weight:{b}   Min Weight:{b}     Max Weight:{b}\n')

                #split
                split_input = input[b].chunk(self.split_groups, dim=-1)
                split_weight = weight[b].chunk(self.split_groups, dim=1)
                
                for i in range(0, self.split_groups):
                    sum_weight = torch.sum(split_weight[i], axis=1)
                    sum_input = torch.sum(split_input[i]/ 2**b, axis=1)

                    ratio_w = (sum_weight/self.arraySize)*100
                    ratio_i = (sum_input/self.arraySize)*100
                    if i == 0:
                        group_w = ratio_w
                        group_i = ratio_i
                    else:
                        group_w += ratio_w
                        group_i += ratio_i
                    file.write(f'Group_{i}  {ratio_i.mean()}    {ratio_i.min()}     {ratio_i.max()}     {ratio_w.mean()}    {ratio_w.min()}     {ratio_w.max()}\n')
                
                if graph:
                    colors=sns.color_palette("husl", bits)
                    sns.histplot(data=(group_i/self.split_groups).cpu(), bins=100, linewidth=0, alpha=0.7, ax=ax[b], color=colors[b], stat="count")
                    sns.histplot(data=(group_w/self.split_groups).cpu(), bins=100, linewidth=0, alpha=0.7, ax=ax[b+4], color=colors[b], stat="count")
                    ax[b].set_title('Input {}'.format(b), fontsize=15, fontdict=dict(weight='bold'))
                    ax[b+4].set_title('Weight {}'.format(b), fontsize=15, fontdict=dict(weight='bold'))
                    ax[b].set_xlabel('Ratio [%]', fontsize=13)
                    ax[b+4].set_xlabel('Ratio [%]', fontsize=13)
                    fig.tight_layout(pad=1.0, h_pad=2)
                    plt.savefig(f'{self.checkpoint}/Ratio_layer{self.layer_idx}.png')

    def _bitserial_log_forward(self, input):
        print(f'[layer{self.layer_idx}]: bitserial mac log')

        # local parameter setting
        bitplane_idx = 0

        # get quantization parameter and input bitserial
        bits = self.bits 
        w_serial = True if self.cbits==1 and self.mapping_mode == 'two_com' else False
        qweight = self.quant_func(self.weight, self.training, serial=w_serial)
        w_scale = self.quant_func.get_alpha()
        sinput, a_scale = self.bitserial_split(input)


        ## get dataframe
        logger = f'{self.checkpoint}/layer{self.layer_idx}_mac_static.pkl'
        df = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

        layer_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
        network_hist = f'{self.checkpoint}/hist/network_hist.pkl'
        
        ### in-mem computation mimic (split conv & psum quant/merge)
        input_chunk = torch.chunk(sinput, bits, dim=1)
        if w_serial:
            sweight, _ = self.bitserial_split(qweight/w_scale, act=False) 
            sweight = torch.where(sweight>0, 1., 0.)
            if self.hwnoise:
                sweight = self.quant_func.noise_cell(sweight)
                std_offset = self.quant_func.noise_cell.get_offset()
            wsplit_num = int(self.bits / self.cbits)
        else:
            sweight = qweight/w_scale
            wsplit_num = 1
        weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)
        if w_serial:
            w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

        psum_scale = w_scale * a_scale

        out_tmp = None
        layer_hist_dict = {}
        for abit, input_s in enumerate(input_chunk):
            abitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_a:{abit}_hist.pkl'
            a_hist_dict = {}
            if w_serial and self.hwnoise:
                out_one = (std_offset) * self._split_forward(input_s, w_one, ignore_bias=True, infer_only=True, merge_group=True)
            for wbit, weight_s in enumerate(weight_chunk):
                wabitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_w:{wbit}_a:{abit}_hist.pkl'
                wa_hist_dict = {}
                a_mag = 2**(abit)
                out_tmp = self._split_forward(input_s, weight_s, ignore_bias=True, infer_only=True)

                out_array = out_tmp.round() / a_mag # noise bound set to round function

                ## NOTE
                df.loc[bitplane_idx] = [wbit, abit,
                                                float(out_array.mean()), 
                                                float(out_array.std()), 
                                                float(out_array.min()), 
                                                float(out_array.max())] 

                out_min = out_array.min()
                out_max = out_array.max()

                # update hist
                for val in range(int(out_min), int(out_max)+1):
                    count = out_array.eq(val).sum().item()
                    # get wa_hist
                    wa_hist_dict[val] = count
                    # get w_hist
                    if val in a_hist_dict.keys():
                        a_hist_dict[val] += count
                    else:
                        a_hist_dict[val] = count

                # save wabitplane_hist
                df_hist = pd.DataFrame(list(wa_hist_dict.items()), columns = ['val', 'count'])
                # wabitplane hist
                if os.path.isfile(wabitplane_hist):
                    print(f'[{self.layer_idx}] Update wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
                    df_wabitplane_hist = pd.read_pickle(wabitplane_hist) 
                    df_merge = pd.merge(df_wabitplane_hist, df_hist, how="outer", on="val")
                    df_merge = df_merge.replace(np.nan, 0)
                    df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                    df_merge = df_merge[['val', 'count']]
                    df_merge.to_pickle(wabitplane_hist)
                else:
                    print(f'[{self.layer_idx}] Create wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
                    df_hist.to_pickle(wabitplane_hist)

                # split output merge
                output_chunk = out_tmp.chunk(self.split_groups, dim=1)
                for g in range(0, self.split_groups):
                    if g==0:
                        out_tmp = output_chunk[g]
                    else:
                        out_tmp += output_chunk[g]

                if w_serial:
                    out_tmp = 2**(wbit)* (out_tmp-out_one) if self.hwnoise else 2**(wbit) * out_tmp
                    if wsplit_num == wbit+1:
                        out_wsum -= out_tmp
                    else:
                        out_wsum = out_tmp if wbit == 0 else out_wsum + out_tmp
                else:
                    out_wsum = out_tmp

                bitplane_idx += 1

            # save abitplane_hist
            df_hist = pd.DataFrame(list(a_hist_dict.items()), columns = ['val', 'count'])
            # wbitplane hist
            if os.path.isfile(abitplane_hist):
                print(f'[{self.layer_idx}] Update abitplane_hist for a:{abit} ({abitplane_hist})')
                df_abitplane_hist = pd.read_pickle(abitplane_hist) 
                df_merge = pd.merge(df_abitplane_hist, df_hist, how="outer", on="val")
                df_merge = df_merge.replace(np.nan, 0)
                df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                df_merge = df_merge[['val', 'count']]
                df_merge.to_pickle(abitplane_hist)
            else:
                print(f'[{self.layer_idx}] Create abitplane_hist for a:{abit} ({abitplane_hist})')
                df_hist.to_pickle(abitplane_hist)

            # update layer hist
            for val, count in a_hist_dict.items():
                if val in layer_hist_dict.keys():
                    layer_hist_dict[val] += count
                else:
                    layer_hist_dict[val] = count
            
            output = out_wsum if abit == 0 else output + out_wsum

        # restore output's scale
        output = output * psum_scale

        # add bias
        if self.bias is not None:
            output += self.bias

        # save logger
        df.to_pickle(logger)
        # df_scaled.to_pickle(logger_scaled)

        # save hist
        df_hist = pd.DataFrame(list(layer_hist_dict.items()), columns = ['val', 'count'])
        # layer hist
        if os.path.isfile(layer_hist):
            print(f'[{self.layer_idx}] Update layer_hist ({layer_hist})')
            df_layer_hist = pd.read_pickle(layer_hist) 
            df_merge = pd.merge(df_layer_hist, df_hist, how="outer", on="val")
            df_merge = df_merge.replace(np.nan, 0)
            df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
            df_merge = df_merge[['val', 'count']]
            df_merge.to_pickle(layer_hist)
        else:
            print(f'[{self.layer_idx}] Create layer_hist ({layer_hist})')
            df_hist.to_pickle(layer_hist)
        # network hist
        if os.path.isfile(network_hist):
            print(f'[{self.layer_idx}] Update network_hist ({network_hist})')
            df_network_hist = pd.read_pickle(network_hist) 
            df_merge = pd.merge(df_network_hist, df_hist, how="outer", on="val")
            df_merge = df_merge.replace(np.nan, 0)
            df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
            df_merge = df_merge[['val', 'count']]
            df_merge.to_pickle(network_hist)
        else:
            print(f'[{self.layer_idx}] Create network_hist ({network_hist})')
            df_hist.to_pickle(network_hist)

        # output_real = F.linear(input, qweight, bias=None)
        # import pdb; pdb.set_trace()

        return output
    
    def _ADC_clamp_value(self):
        # get ADC clipping value for hist [Layer or Network hist]
        if self.psum_mode == 'sigma':
            if self.pclipmode == 'layer':
                phist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
                # phist = f'./hist/layer{self.layer_idx}_hist.pkl'
            elif self.pclipmode == 'network':
                phist = f'{self.checkpoint}/hist/network_hist.pkl'

            if os.path.isfile(phist):
                # print(f'Load ADC_hist ({phist})')
                df_hist = pd.read_pickle(phist)
                mean, std, min, max = get_statistics_from_hist(df_hist)
            else:
                if self.pbits != 32:
                    assert False, "Error: Don't have ADC hist file"
                else:
                    mean, std, min, max = 0, 0, 0, 0
                
            # Why abs(mean) is used not mean?? => Asymmetric quantizaion is occured
            if self.pbits == 32:
                maxVal = 1
                minVal = 0
            else:
                if self.pclip == 'max':
                    maxVal = max
                    minVal = min
                else:
                    maxVal =  (abs(mean) + self.psigma*std).round() 
                    minVal = (abs(mean) - self.psigma*std).round() 
                    if (self.mapping_mode == 'two_com') or (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
                        minVal = min if minVal < 0 else minVal
        else:
            if self.mapping_mode == 'two_com':
                minVal = 0
                if self.pclip == 'max':
                    maxVal = self.arraySize - 1
                elif self.pclip == 'half':
                    maxVal = int(self.arraySize / 2) - 1
                elif self.pclip == 'quarter':
                    maxVal = int(self.arraySize / 4) - 1
                else:
                    assert False, 'Do not support this clip range {}'.format(self.pclip)
            else:
                assert False, 'Psum mode fix only support two_com mapping mode'

        midVal = (maxVal + minVal) / 2
        
        if self.info_print:
            if self.psum_mode == 'sigma':
                write_file = f'{self.checkpoint}/Layer_clipping_range.txt'
                if os.path.isfile(write_file) and (self.layer_idx == 0):
                    option = 'w'
                else:
                    option = 'a'
                with open(write_file, option) as file:
                    if self.layer_idx == 0:
                        file.write(f'{self.pclipmode}-wise Mode Psum quantization \n')
                        file.write(f'Layer_information  Mean    Std     Min Max Clip_Min    Clip_Max    Mid \n')
                    file.write(f'Layer{self.layer_idx}  {mean}  {std}   {min}   {max}   {minVal}    {maxVal}    {midVal}\n')
            
            print(f'{self.pclipmode}-wise Mode Psum quantization')
            if self.pbits == 32:
                print(f'Layer{self.layer_idx} information | pbits {self.pbits}')
            else:
                if self.psum_mode == 'sigma':
                    print(f'Layer{self.layer_idx} information | pbits {self.pbits} | Mean: {mean} | Std: {std} | Min: {min} | Max: {max} | Clip Min: {minVal} | Clip Max: {maxVal} | Mid: {midVal}')
                elif self.psum_mode == 'fix':
                    print(f'Layer{self.layer_idx} information | pbits {self.pbits} | Clip Min: {minVal} | Clip Max: {maxVal} | Mid: {midVal}')
            self.info_print = False

        return minVal, maxVal, midVal

    def _bitserial_comp_forward(self, input):

        # get quantization parameter and input bitserial 
        bits = self.bits
        w_serial = True if self.cbits==1 and self.mapping_mode == 'two_com' else False
        qweight = self.quant_func(self.weight, self.training, serial=w_serial)

        if self.wbit_serial:
            with torch.no_grad():
                w_scale = self.quant_func.get_alpha()
                sinput, a_scale = self.bitserial_split(input)

                psum_scale = w_scale * a_scale

                ### in-mem computation mimic (split conv & psum quant/merge)
                input_chunk = torch.chunk(sinput, bits, dim=1)
                if w_serial:
                    sweight, _ = self.bitserial_split(qweight/w_scale, act=False) 
                    sweight = torch.where(sweight>0, 1., 0.)
                    if self.hwnoise:
                        sweight = self.quant_func.noise_cell(sweight)
                        std_offset = self.quant_func.noise_cell.get_offset()
                    wsplit_num = int(self.bits / self.cbits)
                else:
                    sweight = qweight/w_scale
                    wsplit_num = 1
                weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)
                if w_serial:
                    w_one = torch.ones(size=weight_chunk[0].size(), device=weight_chunk[0].device)
                
                if self.info_print:
                    self.print_ratio(bits, input_chunk, weight_chunk)

                if self.psum_mode == 'sigma' or 'fix':
                    minVal, maxVal, midVal = self._ADC_clamp_value()
                    self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
                elif self.psum_mode == 'scan':
                    if self.info_print:
                        self.info_print = False
                    pass
                else:
                    assert False, 'This script does not support {self.psum_mode}'

                # to compare output data
                for abit, input_s in enumerate(input_chunk):
                    a_mag = 2**(abit)
                    if w_serial and self.hwnoise:
                        out_one = (std_offset) * self._split_forward(input_s, w_one, ignore_bias=True, infer_only=True, merge_group=True)
                    if w_serial and self.accurate:
                        cnt_input = self._split_forward(input_s/a_mag, w_one, ignore_bias=True, cat_output=False, infer_only=True)
                    
                    for wbit, weight_s in enumerate(weight_chunk):
                        out_adc = None
                        out_tmp = self._split_forward(input_s, weight_s, ignore_bias=True, cat_output=False, infer_only=True)

                        if not self.accurate:
                            out_adc = psum_quant_merge(out_adc, out_tmp,
                                                    pbits=self.pbits, step=self.pstep, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    pbound=self.pbound, center=self.center, weight=a_mag,
                                                    groups=self.split_groups, pzero=self.pzero)
                        else:
                            # accurate mode (SRAM)
                            out_quant = None
                            out_quant = psum_quant(out_quant, out_tmp,
                                                    pbits=self.pbits, step=self.pstep, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    pbound=self.pbound, center=self.center, weight=a_mag,
                                                    groups=self.split_groups, pzero=self.pzero)
                            
                            for g in range(0, self.split_groups):
                                accin_addr = torch.where(cnt_input[g] <= MAXThres)
                                out_quant[g][accin_addr] = out_tmp[g][accin_addr]
                                if g==0:
                                    out_adc = out_quant[g]
                                else:
                                    out_adc += out_quant[g]

                        if w_serial:
                            out_adc = 2**(wbit) * (out_adc - out_one) if self.hwnoise else 2**(wbit) * out_adc
                            if wsplit_num == wbit+1:
                                out_wsum -= out_adc
                            else:
                                out_wsum = out_adc if wbit == 0 else out_wsum + out_adc
                        else:
                            out_wsum = out_adc

                    output = out_wsum if abit == 0 else output+out_wsum

                # restore output's scale
                output = output * psum_scale
        else:
            # no serial compuatation with psum computation 
            self.pbits = 32
            output = self._split_forward(input, qweight, ignore_bias=True, merge_group=True)

        # add bias
        if self.bias is not None:
            output += self.bias
        
        # output_real = F.linear(input, qweight, bias=None)
        # import pdb; pdb.set_trace()

        return output

    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
        
        if self.bitserial_log:
            return self._bitserial_log_forward(x)
        else:
            return self._bitserial_comp_forward(x)

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        s =  'in_features={}, out_features={}, bias={}, bits={}, wbit_serial={}, split_groups={}, '\
            'mapping_mode={}, cbits={}, psum_mode={}, pbits={}, pbound={}, '\
            'bitserial_log={}, layer_idx={}'\
            .format(self.in_features, self.out_features, self.bias is not None, self.bits, self.wbit_serial,
            self.split_groups, self.mapping_mode, self.cbits, self.psum_mode, self.pbits, self.pbound, 
            self.bitserial_log, self.layer_idx)
        return s

def get_statistics_from_hist(df_hist):
    num_elements = df_hist['count'].sum()
    # min/max
    min_val = df_hist['val'].min()
    max_val = df_hist['val'].max()

    # mean
    df_hist['sum'] = df_hist['val'] * df_hist['count']
    mean_val = df_hist['sum'].sum() / num_elements

    # std
    df_hist['centered'] = df_hist['val'] - mean_val
    df_hist['var_sum'] = (df_hist['centered'] * df_hist['centered']) * df_hist['count']
    var_val = df_hist['var_sum'].sum() / num_elements
    std_val = math.sqrt(var_val)

    return [mean_val, std_val, min_val, max_val] 

def psum_initialize(model, act=True, weight=True, fixed_bit=-1, cbits=4, arraySize=128, mapping_mode='2T2R', psum_mode='sigma',
                    wbit_serial=False, pbits=32, pclipmode='layer', pclip='sigma', psigma=3, pbound=None, center=None,
                    accurate=False, checkpoint=None, info_print=False, log_file=None):
    counter=0
    for name, module in model.named_modules():
        if isinstance(module, (QuantActs.ReLU, QuantActs.HSwish, QuantActs.Sym)) and act:
            module.quant = True

            module.quant_func.noise = False
            module.quant_func.is_stochastic = True
            module.quant_func.is_discretize = True

            if fixed_bit != -1 :
                bit = ( fixed_bit+0.00001 -2 ) / 12
                bit = np.log(bit/(1-bit))
                module.quant_func.bit.data.fill_(bit)
                module.quant_func.bit.requires_grad = False
            
            #module.bit.data.fill_(-2)

        if isinstance(module, (Psum_QConv2d, Psum_QLinear))and weight:
            module.quant_func.noise = False
            module.quant_func.is_stochastic = True
            module.quant_func.is_discretize = True

            if fixed_bit != -1 :
                bit = ( fixed_bit -2 ) / 12
                bit = np.log(bit/(1-bit))
                module.quant_func.bit.data.fill_(bit)
                module.quant_func.bit.requires_grad = False
                module.bits = fixed_bit
            
            module.cbits = cbits
            module.wbit_serial = wbit_serial
            module.model_split_groups(arraySize)
            module.mapping_mode = mapping_mode
            module.pbits = pbits
            module.pclipmode = pclipmode.lower()
            module.psum_mode = psum_mode
            if psum_mode == 'sigma':
                module.pclip = pclip
                module.psigma = psigma
            elif psum_mode == 'fix':
                module.pclip = pclip
            elif psum_mode == 'scan':
                module.setting_pquant_func(pbits, center, pbound)
            else:
                assert False, "Only two options [sigma, scan]"
            
            module.accurate = accurate

            module.bitserial_log = log_file
            module.layer_idx = counter 
            module.checkpoint = checkpoint
            module.info_print = info_print
            counter += 1

def unset_bitserial_log(model):
    print("start unsetting Bitserial layers log bitplane info")
    counter = 0
    for name, module in model.named_modules():
        if isinstance(module, (Psum_QConv2d, Psum_QLinear)):
            module.bitserial_log = False
            print("Finish log unsetting {}, idx: {}".format(name.replace("module.", ""), counter))
            counter += 1

def set_bitserial_layer(model, pquant_idx, wbit_serial=None, pbits=32, center=[]):
    ## set block for bit serial computation
    print("start setting bitserial layer")
    counter = 0
    for name, module in model.named_modules():
        if isinstance(module, (Psum_QConv2d, Psum_QLinear)):
            if counter == pquant_idx:
                module.reset_layer(wbit_serial=wbit_serial, pbits=pbits, center=center)
            counter += 1
    print("finish setting bitserial layer ")

def set_bound_layer(model, pquant_idx, pbits, pbound, center):
    counter = 0
    for name, module in model.named_modules():
        if isinstance(module, (Psum_QConv2d, Psum_QLinear)):
            if counter == pquant_idx:
                module.setting_pquant_func(pbits, center, pbound)
            counter += 1

def hwnoise_initilaize(model, weight=False, hwnoise=True, cbits=4, mapping_mode=None, co_noise=0.01, noise_type='prop', res_val='rel', max_epoch=-1):
    for name, module in model.named_modules():
        if isinstance(module, (Psum_QConv2d, Psum_QLinear)) and weight and hwnoise:
            module.quant_func.hwnoise = True
            module.hwnoise = True

            if noise_type == 'grad':
                assert max_epoch != -1, "Enter max_epoch in hwnoise_initialize function"
            if hwnoise:
                module.quant_func.hwnoise_init(cbits=cbits, mapping_mode=mapping_mode, co_noise=co_noise, noise_type=noise_type, res_val=res_val, max_epoch=max_epoch)

class PsumQuantOps(object):
    psum_initialize = psum_initialize
    set_bitserial_layer = set_bitserial_layer
    set_bound_layer = set_bound_layer
    hwnoise_initilaize = hwnoise_initilaize
    unset_bitserial_log = unset_bitserial_log
    Conv2d = Psum_QConv2d
    Linear = Psum_QLinear