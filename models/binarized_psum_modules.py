import numpy as np
import torch
import math
import os
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter 
import pandas as pd
from .noise_cell import Noise_cell
import utils.padding as Pad
from .quantized_lsq_modules import *
from .quantized_basic_modules import psum_quant_merge
from .bitserial_modules import *
from .split_modules import *
# custom kernel
import conv_sweight_cuda

# split convolution across input channel
def split_conv(weight, nWL):
    nIC = weight.shape[1]
    nWH = weight.shape[2]*weight.shape[3]
    nMem = int(math.ceil(nIC/math.floor(nWL/nWH)))
    nIC_list = [int(math.floor(nIC/nMem)) for _ in range(nMem)]
    for idx in range((nIC-nIC_list[0]*nMem)):
        nIC_list[idx] += 1

    return nIC_list

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

class PsumBinConv(SplitConv):
    """
        Quant(LSQ)Conv + Psum quantization
    """
    def __init__(self, in_channels, out_channels, wbits=32, kernel_size=3, stride=1, padding=0, groups=1, bias=False, padding_mode='zeros', 
                weight_clip=0, weight_scale=True, arraySize=128, mapping_mode='none', psum_mode='sigma', cbits=None,
                is_noise=False, noise_type=None):
        super(PsumBinConv, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, groups=groups, bias=bias)
        # for Qconv
        self.wbits = wbits
        if self.wbits == 32: # ignore 1-padding and weight clipping when using full precision weight
            # self.padding_value = 0
            self.weight_clip = 0
            self.weight_scale = False
        else:
            self.weight_clip = weight_clip
            self.weight_scale = weight_scale
        self.padding = padding
        self.stride = stride
        self.padding_mode = padding_mode
        
        if self.padding_mode == 'zeros':
            self.padding_value = 0
        elif self.padding_mode == 'ones':
            self.padding_value = 1
        elif self.padding_mode == 'alter':
            self.padding_value = 0
        elif self.padding_mode == 'neg':
            self.padding_value = None

        # for split
        # self.split_nIC = split_conv(self.weight, arraySize)
        self.split_groups = calculate_groups(arraySize, self.fan_in)
        if self.fan_in % self.split_groups != 0:
            raise ValueError('fan_in must be divisible by groups')
        self.group_fan_in = int(self.fan_in / self.split_groups)
        self.group_in_channels = int(np.ceil(in_channels / self.split_groups))
        residual = self.group_fan_in % self.kSpatial
        if residual != 0:
            if self.kSpatial % residual != 0:
                self.group_in_channels += 1
        ## log move group for masking & group convolution
        self.group_move_in_channels = torch.zeros(self.split_groups-1, dtype=torch.int)
        group_in_offset = torch.zeros(self.split_groups, dtype=torch.int)
        self.register_buffer('group_in_offset', group_in_offset)
        ## get group conv info
        self._group_move_offset()
        
        # sweight
        sweight = torch.Tensor(self.out_channels*self.split_groups, self.group_in_channels, kernel_size, kernel_size)
        self.register_buffer('sweight', sweight)

        # for psum quantization
        self.mapping_mode = mapping_mode # Array mapping method [none, 2T2R, two_com, ref_d]]
        self.arraySize = arraySize
        self.cbits = cbits # Cell bits [multi, binary]
        self.psum_mode = psum_mode
        self.pclipmode = 'Layer'
        self.pbits = 32
        # for scan version
        self.pstep = None
        self.pzero = None # contain zero value (True)
        self.center = None
        self.pbound = arraySize if arraySize > 0 else self.fan_in
        # for sigma version
        self.pclip = 'sigma'
        self.psigma = 3

        # for noise option
        self.is_noise = is_noise
        self.noise_type = noise_type
        self.co_noise = 0
        self.w_format = 'weight'

        # for logging
        self.bitserial_log = False
        self.layer_idx = -1
        self.checkpoint = None
        self.info_print = True

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

    def reset_layer(self, groups=None, pbits=None, pbound=None, center=[]):
        self.setting_pquant_func(pbits=pbits, center=center, pbound=pbound)

    def _weight_bitserial(self, weight, w_scale, cbits=4):
        weight_uint = weight / w_scale
        if self.mapping_mode == "two_com":
            signed_w = (2**(self.wbits-1))*torch.where(weight_uint<0, 1.0, 0.0)
            value_w = torch.where(weight_uint<0, 2**(self.wbits-1) - abs(weight_uint), weight_uint)
            if cbits == 1:
                weight_serial = bitserial_func(value_w, (self.wbits-1))
                output = torch.cat((weight_serial, signed_w), 1)
                split_num = self.wbits
            elif cbits > 1:
                output = torch.cat((value_w, signed_w), 1)
                split_num = 2
            else:
                assert False, "Please select cell state mode"
        elif self.mapping_mode == "ref_d":
            if cbits > 1:
                shift_v = 2**(self.wbits-1)
                shift_w = shift_v*torch.ones(weight.size()).cuda()
                value_w = torch.add(weight_uint, shift_v)
                output = torch.cat((value_w, shift_w), 1)
                split_num = 2
            else:
                assert False, "Pleas select multi cell state for reference digital mapping mode"
        elif self.mapping_mode == "2T2R":
            if self.is_noise and self.w_format=='state':
                zeros = torch.zeros(weight_uint.size(), device=weight_uint.device)
                pos_w = torch.where(weight_uint>0, weight_uint, zeros)
                neg_w = torch.where(weight_uint<0, abs(weight_uint), zeros)
                output = torch.cat((pos_w, neg_w), 1)# 9 level cell bits 
                split_num=2
            else:
                # range = 2 ** (self.wbits - 1) - 1
                # output = torch.clamp(weight_uint, -range, range) # 8 level cell bits 
                output = weight_uint # 9 level cell bits 
                split_num = 1
        elif self.mapping_mode == "ref_a":
            if self.is_noise and self.w_format=='state':
                shift_v = 2**(self.wbits-1)
                shift_w = shift_v*torch.ones(weight.size()).cuda()
                value_w = torch.add(weight_uint, shift_v)
                output = torch.cat((value_w, shift_w), 1)
                split_num = 2
            else:
                output = weight_uint # 9 level cell bits 
                split_num=1
        elif self.mapping_mode == "PN":
            if cbits > 1:
                zeros = torch.zeros(weight_uint.size(), device=weight_uint.device)
                pos_w = torch.where(weight_uint>0, weight_uint, zeros)
                neg_w = torch.where(weight_uint<0, abs(weight_uint), zeros)
                output = torch.cat((pos_w, neg_w), 1)# 9 level cell bits 
                split_num=2
            else:
                assert False, "Pleas select multi cell state for reference digital mapping mode"
        else:
            output = weight_uint
            split_num = 1

        return output.round_(), split_num 
    
    def _cell_noise_init(self, cbits, mapping_mode, wsym=False, co_noise=0.01, noise_type='prop', res_val='rel', w_format="weight", max_epoch=-1):
        self.w_format = 'state' if res_val == 'abs' or noise_type == 'meas' else 'weight'
        # wbits = Parameter(torch.Tensor(1).fill_(self.wbits), requires_grad=False).round().squeeze()
        # for inference 
        if not (noise_type == 'prop' or 'interp'):
            noise_type = 'prop'
        self.noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, wsym, co_noise, noise_type, res_val=res_val, w_format=self.w_format)
    
    def _bitserial_log_forward(self, input):
        print(f'[layer{self.layer_idx}]: bitserial mac log')
        # delete padding_shpe & additional padding operation by matching padding/stride format with nn.Conv2d
        if self.padding > 0:
            if self.padding_mode == 'neg':
                self.padding_value = float(input.min()) 
            padding_shape = (self.padding, self.padding, self.padding, self.padding)
            input = Pad.pad(input, padding_shape, self.padding_mode, self.padding_value)

        # local parameter setting
        bitplane_idx = 0

        with torch.no_grad():
            # get quantization parameter and input bitserial 
            qweight = fw(self.weight, self.wbits, self.weight_clip, self.weight_scale, self.fan_in)

            if self.is_noise and self.w_format=="weight":
                import pdb; pdb.set_trace()
                qweight = self.noise_cell(qweight)


            ## get dataframe
            logger = f'{self.checkpoint}/layer{self.layer_idx}_mac_static.pkl'
            df = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

            # logger_scaled = f'{self.checkpoint}/layer{self.layer_idx}_wabitplane_mac_static_scaled_lastbatch.pkl'
            # df_scaled = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

            layer_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
            network_hist = f'{self.checkpoint}/hist/network_hist.pkl'

            ### Cell noise injection + Cell conductance value change
            if self.is_noise:
                bweight, wsplit_num = self._weight_bitserial(qweight, 1, cbits=self.cbits)
                bweight = self.noise_cell(bweight)
                
                weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
                
                ## make the same weight size as qweight
                if (self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a'):
                    bweight = weight_chunk[0] - weight_chunk[1]
                    wsplit_num = 1
                else:
                    ## [TODO] two_com split weight format check
                    delta_G, G_min = self.noise_cell.get_deltaG(G_min=True)
                    w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

                self.sweight = conv_sweight_cuda.forward(self.sweight, bweight, self.group_in_offset, self.split_groups)
                weight_chunk = torch.chunk(self.sweight, wsplit_num, dim=1)
            else:
                self.sweight = conv_sweight_cuda.forward(self.sweight, qweight, self.group_in_offset, self.split_groups)
                sweight, wsplit_num = self._weight_bitserial(self.sweight, 1, cbits=self.cbits)
                weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)

            # in-memory computing parameter computing
            out_tmp = None
            layer_hist_dict = {}
            for wbit, weight_s in enumerate(weight_chunk):
                wabitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_w:{wbit}__hist.pkl'
                wa_hist_dict = {}
                out_tmp = self._split_forward(input, weight_s, padded=True, ignore_bias=True,
                                                weight_is_split=True, infer_only=True)
                # out_tmp = F.conv2d(input_s[:,nIC_cnt:nIC_cnt+self.split_nIC[idx],:,:], weight_s, bias=self.bias,
                #             stride=self.stride, dilation=self.dilation, groups=self.groups)

                out_array = out_tmp.round()
                ## NOTE
                df.loc[bitplane_idx] = [wbit, 0,
                                                float(out_array.mean()), 
                                                float(out_array.std()), 
                                                float(out_array.min()), 
                                                float(out_array.max())] 

                # out_tmp_scale = out_tmp / self.pquant_bitplane[bitplane_idx]
                out_min = out_array.min()
                out_max = out_array.max()
                # df_scaled.loc[bitplane_idx] = [wbit, abit,
                #                                 float(out_tmp_scale.mean()), 
                #                                 float(out_tmp_scale.std()), 
                #                                 float(out_min), 
                #                                 float(out_max)] 

                # update hist
                for val in range(int(out_min), int(out_max)+1):
                    count = out_array.eq(val).sum().item()
                    # get wa_hist
                    wa_hist_dict[val] = count

                # save wabitplane_hist
                df_hist = pd.DataFrame(list(wa_hist_dict.items()), columns = ['val', 'count'])
                # wabitplane hist
                if os.path.isfile(wabitplane_hist):
                    print(f'[{self.layer_idx}]Update wabitplane_hist for w:{wbit} ({wabitplane_hist})')
                    df_wabitplane_hist = pd.read_pickle(wabitplane_hist) 
                    df_merge = pd.merge(df_wabitplane_hist, df_hist, how="outer", on="val")
                    df_merge = df_merge.replace(np.nan, 0)
                    df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                    df_merge = df_merge[['val', 'count']]
                    df_merge.to_pickle(wabitplane_hist)
                else:
                    print(f'[{self.layer_idx}]Create wabitplane_hist for w:{wbit} ({wabitplane_hist})')
                    df_hist.to_pickle(wabitplane_hist)

                # split output merge
                output_chunk = out_tmp.chunk(self.split_groups, dim=1) 
                for g in range(0, self.split_groups):
                    if g==0:
                        out_tmp = output_chunk[g]
                    else:
                        out_tmp += output_chunk[g]

                # weight output summation
                if self.mapping_mode == 'two_com':
                    if wsplit_num == wbit+1:
                        out_wsum -= out_tmp
                    else:
                        out_wsum = out_tmp if wbit == 0 else out_wsum + out_tmp
                elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
                    out_wsum = out_tmp if wbit == 0 else out_wsum - out_tmp
                else:
                        out_wsum = out_tmp if wbit == 0 else out_wsum + out_tmp

                bitplane_idx += 1

            # update layer hist
            for val, count in wa_hist_dict.items():
                if val in layer_hist_dict.keys():
                    layer_hist_dict[val] += count
                else:
                    layer_hist_dict[val] = count

            if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                out_one = (-G_min/delta_G) * self._split_forward(input, w_one, padded=True, ignore_bias=True,
                                                weight_is_split=True, infer_only=True, merge_group=True)
                out_wsum -= out_one
            output = out_wsum 

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
            print(f'[{self.layer_idx}]Update network_hist ({network_hist})')
            df_network_hist = pd.read_pickle(network_hist) 
            df_merge = pd.merge(df_network_hist, df_hist, how="outer", on="val")
            df_merge = df_merge.replace(np.nan, 0)
            df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
            df_merge = df_merge[['val', 'count']]
            df_merge.to_pickle(network_hist)
        else:
            print(f'[{self.layer_idx}] Create network_hist ({network_hist})')
            df_hist.to_pickle(network_hist)

        # if self.concat:
        #     output_real = F.conv2d(input, qweight, bias=self.bias,
        #                         stride=self.stride, dilation=self.dilation, groups=self.groups)
        #     import pdb; pdb.set_trace()
        #     print('split', output[0][0][0])
        #     print('real', output_real[0][0][0])

        return output

    def _ADC_clamp_value(self):
        # get ADC clipping value for hist [Layer or Network hist]
        if self.pclipmode == 'Layer':
            phist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
            # phist = f'./hist/layer{self.layer_idx}_hist.pkl'
        elif self.pclipmode == 'Network':
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
        
        midVal = (maxVal + minVal) / 2

        if self.info_print:
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
                print(f'Layer{self.layer_idx} information | pbits {self.pbits} | Mean: {mean} | Std: {std} | Min: {min} | Max: {max} | Clip Min: {minVal} | Clip Max: {maxVal} | Mid: {midVal}')
            self.info_print = False

        return minVal, maxVal, midVal 

    def _bitserial_comp_forward(self, input):
        # delete padding_shpe & additional padding operation by matching padding/stride format with nn.Conv2d
        if self.padding > 0:
            if self.padding_mode == 'neg':
                self.padding_value = float(input.min()) 
            padding_shape = (self.padding, self.padding, self.padding, self.padding)
            input = Pad.pad(input, padding_shape, self.padding_mode, self.padding_value)

        qweight = fw(self.weight, self.wbits, self.weight_clip, self.weight_scale, self.fan_in)

        if self.pbits == 1:
            self.pstep = 1
            self.center = 0 
            assert self.mapping_mode == '2T2R', "Only support 1bit (SA) when mapping mode is 2T2R"
        else:
            if self.psum_mode == 'sigma':
                minVal, maxVal, midVal = self._ADC_clamp_value()
                self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
            elif self.psum_mode == 'scan':
                pass
            else:
                assert False, 'This script does not support {self.psum_mode}'

        ### Cell noise injection + Cell conductance value change
        ### in-mem computation programming (weight constant-noise)
        if self.is_noise:
            bweight, wsplit_num = self._weight_bitserial(qweight, 1, cbits=self.cbits)
            bweight = self.noise_cell(bweight)
            
            weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
            
            ## make the same weight size as qweight
            if (self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a'):
                bweight = weight_chunk[0] - weight_chunk[1]
                wsplit_num = 1
            else:
                ## [TODO] two_com split weight format check
                delta_G, G_min = self.noise_cell.get_deltaG(G_min=True)
                w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

            self.sweight = conv_sweight_cuda.forward(self.sweight, bweight, self.group_in_offset, self.split_groups)
            weight_chunk = torch.chunk(self.sweight, wsplit_num, dim=1)
        else:
            self.sweight = conv_sweight_cuda.forward(self.sweight, qweight, self.group_in_offset, self.split_groups)
            sweight, wsplit_num = self._weight_bitserial(self.sweight, 1, cbits=self.cbits)
            weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)
        
        self.weight_chunk = weight_chunk

        out_adc = None
        for wbit, weight_s in enumerate(weight_chunk):
            out_tmp = self._split_forward(input, weight_s, padded=True, ignore_bias=True, cat_output=False,
                                    weight_is_split=True, infer_only=True)

            out_adc = psum_quant_merge(out_adc, out_tmp,
                                        pbits=self.pbits, step=self.pstep, 
                                        half_num_levels=self.phalf_num_levels, 
                                        pbound=self.pbound, center=self.center, weight=1,
                                        groups=self.split_groups, pzero=self.pzero)

            # weight output summation
            if self.mapping_mode == 'two_com':
                if wsplit_num == wbit+1:
                    output -= out_adc
                else:
                    output = out_adc if wbit == 0 else output + out_adc
            elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
                output = out_adc if wbit == 0 else output - out_adc
            else:
                output = out_adc if wbit == 0 else output + out_adc
            out_adc = None

        if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
            out_one = (-G_min/delta_G) * self._split_forward(input, w_one, padded=True, ignore_bias=True, cat_output=False,
                                    weight_is_split=True, infer_only=True, merge_group=True)
            output -= out_one
        # output_real = F.conv2d(input_s, qweight, bias=self.bias,
        #                         stride=self.stride, dilation=self.dilation, groups=self.groups)
        # import pdb; pdb.set_trace()

        # add bias
        if self.bias is not None:
            output += self.bias

        # output_real = F.conv2d(input, qweight, bias=self.bias,
        #                         stride=self.stride, dilation=self.dilation, groups=self.groups)
        # print(output_real[0][0])
        # print(output[0][0])
        # if self.layer_idx == 4:
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, ax = plt.subplots(ncols=2, figsize=(20, 12))

        # sns.histplot((qweight/w_scale).detach().cpu().numpy().ravel(), ax=ax[0], alpha=0.5)
        # sns.histplot((bweight).detach().cpu().numpy().ravel(), ax=ax[1], alpha=0.5)
        # ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=14)
        # ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=14)
        # ax[0].set_title(f'Before Noise', loc='right', fontsize=16)
        # ax[1].set_title(f'After Noise', loc='right', fontsize=16)
        # ax[0].set_xlabel('Weight', fontsize=16)
        # ax[1].set_xlabel('Weight', fontsize=16)
        # ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=14)
        # ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=14)

        # plt.savefig(os.getcwd() +f"/graph/ReRAM/NAT_Layer{self.layer_idx}_weight.png", bbox_inches='tight')

        # import pdb; pdb.set_trace()

        return output

    def forward(self, input):
        if self.wbits == 32:
            return F.conv2d(input, self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        if self.bitserial_log:
            return self._bitserial_log_forward(input)
        else:
            return self._bitserial_comp_forward(input)

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
        s += ', split_groups={split_groups}, mapping_mode={mapping_mode}, cbits={cbits}'
        s += ', psum_mode={psum_mode}, pbits={pbits}, pbound={pbound}'
        s += ', noise={is_noise}'
        s += ', bitserial_log={bitserial_log}, layer_idx={layer_idx}'            
        return s.format(**self.__dict__)

class PsumBinLinear(SplitLinear):
    """
        Quant(LSQ)Linear + Psum quantization
    """
    def __init__(self, in_features, out_features, wbits, symmetric=False, bias=False,
                weight_clip=0, weight_scale=True, arraySize=128, mapping_mode='none', psum_mode='sigma', cbits=None,
                is_noise=False, noise_type=None):
        super(PsumBinLinear, self).__init__(in_features, out_features, bias=bias)
        # for QLinear
        self.wbits = wbits
        if self.wbits == 32: # ignore 1-padding and weight clipping when using full precision weight
            # self.padding_value = 0
            self.weight_clip = 0
            self.weight_scale = False
        else:
            self.weight_clip = weight_clip
            self.weight_scale = weight_scale

        # for split
        # self.split_nIF = split_linear(self.weight, arraySize)
        self.split_groups = calculate_groups(arraySize, in_features)
        if in_features % self.split_groups != 0:
            raise ValueError('in_features must be divisible by groups')
        self.fan_in =  int(in_features / self.split_groups)
        self.group_in_features = int(in_features / self.split_groups)

        # for psum quantization
        self.mapping_mode = mapping_mode # Array mapping method [none, 2T2R, two_com, ref_d]
        self.cbits = cbits # Cell bits [multi, binary]
        self.psum_mode = psum_mode
        self.pclipmode = 'Layer'
        self.pbits = 32
        # for scan version
        self.pstep = None
        self.pzero = None # contain zero value (True)
        self.center = None
        self.pbound = arraySize if arraySize > 0 else self.fan_in
        # for sigma version
        self.pclip = 'sigma'
        self.psigma = 3

        # for noise option
        self.is_noise = is_noise
        self.w_format = 'weight'

        # for logging
        self.bitserial_log = False
        self.layer_idx = -1
        self.checkpoint = None
        self.info_print = True
    
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
    
    def reset_layer(self, groups=None, pbits=None, pbound=None, center=[]):
        self.setting_pquant_func(pbits=pbits, center=center, pbound=pbound)
    
    def _weight_bitserial(self, weight, w_scale, cbits=4):
        weight_uint = weight / w_scale
        if self.mapping_mode == "two_com":
            signed_w = (2**(self.wbits-1))*torch.where(weight_uint<0, 1.0, 0.0)
            value_w = torch.where(weight_uint<0, 2**(self.wbits-1) - abs(weight_uint), weight_uint)
            if cbits == 1:
                weight_serial = bitserial_func(value_w, (self.wbits-1))
                output = torch.cat((weight_serial, signed_w), 1)
                split_num = self.wbits
            elif cbits > 1:
                output = torch.cat((value_w, signed_w), 1)
                split_num = 2
            else:
                assert None, "Please select cell state mode"
        elif self.mapping_mode == "ref_d":
            if cbits > 1:
                shift_v = 2**(self.wbits-1)
                shift_w = shift_v*torch.ones(weight.size()).cuda()
                value_w = torch.add(weight_uint, shift_v)
                output = torch.cat((value_w, shift_w), 1)
                split_num = 2
            else:
                assert False, "Pleas select multi cell state for reference digital mapping mode"
        elif self.mapping_mode == "2T2R":
            if self.is_noise and self.w_format=='state':
                zeros = torch.zeros(weight_uint.size(), device=weight_uint.device)
                pos_w = torch.where(weight_uint>0, weight_uint, zeros)
                neg_w = torch.where(weight_uint<0, abs(weight_uint), zeros)
                output = torch.cat((pos_w, neg_w), 1)# 9 level cell bits 
                split_num=2
            else:
                # range = 2 ** (self.wbits - 1) - 1
                # output = torch.clamp(weight_uint, -range, range) # 8 level cell bits 
                output = weight_uint # 9 level cell bits 
                split_num=1
        elif self.mapping_mode == "PN":
            if cbits > 1:
                zeros = torch.zeros(weight_uint.size(), device=weight_uint.device)
                pos_w = torch.where(weight_uint>0, weight_uint, zeros)
                neg_w = torch.where(weight_uint<0, abs(weight_uint), zeros)
                output = torch.cat((pos_w, neg_w), 1)# 9 level cell bits 
                split_num=2
            else:
                assert False, "Pleas select multi cell state for reference digital mapping mode"
        elif self.mapping_mode == "ref_a":
            if self.is_noise and self.w_format=='state':
                shift_v = 2**(self.wbits-1)
                shift_w = shift_v*torch.ones(weight.size()).cuda()
                value_w = torch.add(weight_uint, shift_v)
                output = torch.cat((value_w, shift_w), 1)
                split_num = 2
            else:
                output = weight_uint # 9 level cell bits 
                split_num=1
        else:
            output = weight_uint
            split_num = 1

        return output.round_(), split_num 

    def _cell_noise_init(self, cbits, mapping_mode, wsym=False, co_noise=0.01, noise_type='prop', res_val='rel', w_format="weight", max_epoch=-1):
        self.w_format = 'state' if res_val == 'abs' or noise_type == 'meas' else 'weight'
        # wbits = Parameter(torch.Tensor(1).fill_(self.wbits), requires_grad=False).round().squeeze()
        # for inference 
        if not (noise_type == 'prop' or 'interp'):
            noise_type = 'prop'
        self.noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, wsym, co_noise, noise_type, res_val=res_val, w_format=self.w_format)
    
    def _bitserial_log_forward(self, input):
        print(f'[layer{self.layer_idx}]: bitserial mac log')

        # local parameter setting
        bitplane_idx = 0

        # get quantization parameter and input bitserial 
        qweight = fw(self.weight, self.wbits, self.weight_clip, self.weight_scale, self.fan_in)

        if self.is_noise and self.w_format=="weight":
            qweight = self.noise_cell(qweight)

        ## get dataframe
        logger = f'{self.checkpoint}/layer{self.layer_idx}_mac_static.pkl'
        df = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

        # logger_scaled = f'{self.checkpoint}/layer{self.layer_idx}_wabitplane_mac_static_scaled_lastbatch.pkl'
        # df_scaled = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

        layer_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
        network_hist = f'{self.checkpoint}/hist/network_hist.pkl'
        
        layer_hist_dict = {}

        ### in-mem computation mimic (split conv & psum quant/merge)
        bweight, wsplit_num = self._weight_bitserial(qweight, 1, cbits=self.cbits)

        ### Cell noise injection + Cell conductance value change
        if self.is_noise:
            bweight = self.noise_cell(bweight)
            weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
            
            if (self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a'):
                bweight = weight_chunk[0] - weight_chunk[1]
                wsplit_num = 1
            else:
                ## [TODO] two_com split weight format check
                delta_G, G_min = self.noise_cell.get_deltaG(G_min=True)
                w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

        weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)

        out_tmp = None
        for wbit, weight_s in enumerate(weight_chunk):
            wabitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_w:{wbit}_hist.pkl'
            wa_hist_dict = {}
            out_tmp = self._split_forward(input, weight_s, ignore_bias=True, infer_only=True)
            # out_tmp = F.linear(input_s[:,nIF_cnt:nIF_cnt+self.split_nIF[idx]], weight_s, bias=None)

            out_array = out_tmp.round()
                            
            ## NOTE
            df.loc[bitplane_idx] = [wbit, 0,
                                            float(out_array.mean()), 
                                            float(out_array.std()), 
                                            float(out_array.min()), 
                                            float(out_array.max())] 

            # out_tmp_scale = out_tmp / self.pquant_bitplane[bitplane_idx]
            out_min = out_array.min()
            out_max = out_array.max()
            # df_scaled.loc[bitplane_idx] = [wbit, abit,
            #                                 float(out_tmp_scale.mean()), 
            #                                 float(out_tmp_scale.std()), 
            #                                 float(out_min), 
            #                                 float(out_max)] 

            # update hist
            for val in range(int(out_min), int(out_max)+1):
                count = out_array.eq(val).sum().item()
                # get wa_hist
                wa_hist_dict[val] = count

            # save wabitplane_hist
            df_hist = pd.DataFrame(list(wa_hist_dict.items()), columns = ['val', 'count'])
            # wabitplane hist
            if os.path.isfile(wabitplane_hist):
                print(f'[{self.layer_idx}]Update wabitplane_hist for w:{wbit} ({wabitplane_hist})')
                df_wabitplane_hist = pd.read_pickle(wabitplane_hist) 
                df_merge = pd.merge(df_wabitplane_hist, df_hist, how="outer", on="val")
                df_merge = df_merge.replace(np.nan, 0)
                df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                df_merge = df_merge[['val', 'count']]
                df_merge.to_pickle(wabitplane_hist)
            else:
                print(f'[{self.layer_idx}]Create wabitplane_hist for w:{wbit} ({wabitplane_hist})')
                df_hist.to_pickle(wabitplane_hist)

            # split output merge
            output_chunk = out_tmp.chunk(self.split_groups, dim=1) 
            for g in range(0, self.split_groups):
                if g==0:
                    out_tmp = output_chunk[g]
                else:
                    out_tmp += output_chunk[g]

            # weight output summation
            if self.mapping_mode == 'two_com':
                if wsplit_num == wbit+1:
                    out_wsum -= out_tmp
                else:
                    out_wsum = out_tmp if wbit == 0 else out_wsum + out_tmp
            elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
                out_wsum = out_tmp if wbit == 0 else out_wsum - out_tmp
            else:
                out_wsum = out_tmp if wbit == 0 else out_wsum + out_tmp

            bitplane_idx += 1

        # update layer hist
        for val, count in wa_hist_dict.items():
            if val in layer_hist_dict.keys():
                layer_hist_dict[val] += count
            else:
                layer_hist_dict[val] = count

        if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
            out_one = (-G_min/delta_G) * self._split_forward(input, w_one, ignore_bias=True, infer_only=True, merge_group=True)
            out_wsum -= out_one
        output = out_wsum 

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
            print(f'[{self.layer_idx}]Update network_hist ({network_hist})')
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
        if self.pclipmode == 'Layer':
            phist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
            # phist = f'./hist/layer{self.layer_idx}_hist.pkl'
        elif self.pclipmode == 'Network':
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
        
        midVal = (maxVal + minVal) / 2
        
        if self.info_print:
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
                print(f'Layer{self.layer_idx} information | pbits {self.pbits} | Mean: {mean} | Std: {std} | Min: {min} | Max: {max} | Clip Min: {minVal} | Clip Max: {maxVal} | Mid: {midVal}')
            self.info_print = False

        return minVal, maxVal, midVal

    def _bitserial_comp_forward(self, input):

        # get quantization parameter and input bitserial 
        qweight = fw(self.weight, self.wbits, self.weight_clip, self.weight_scale, self.fan_in)

        with torch.no_grad():
            if self.pbits == 1:
                self.pstep = 1
                self.center = 0 
                assert self.mapping_mode != '2T2R', "Only support 1bit (SA) when mapping mode is 2T2R"
            else:
                if self.psum_mode == 'sigma':
                    minVal, maxVal, midVal = self._ADC_clamp_value()
                    self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
                elif self.psum_mode == 'scan':
                    pass
                else:
                    assert False, 'This script does not support {self.psum_mode}'

            ### in-mem computation mimic (split conv & psum quant/merge)
            bweight, wsplit_num = self._weight_bitserial(qweight, 1, cbits=self.cbits)

            ### Cell noise injection + Cell conductance value change
            if self.is_noise:
                bweight = self.noise_cell(bweight)
                weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
                
                if (self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a'):
                    bweight = weight_chunk[0] - weight_chunk[1]
                    wsplit_num = 1
                else:
                    ## [TODO] two_com split weight format check
                    delta_G, G_min = self.noise_cell.get_deltaG(G_min=True)
                    w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

            weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
            self.weight_chunk = weight_chunk

            out_adc = None
            for wbit, weight_s in enumerate(weight_chunk):
                out_tmp = self._split_forward(input, weight_s, ignore_bias=True, cat_output=False, infer_only=True)
                # out_tmp = F.linear(input_s[:,nIF_cnt:nIF_cnt+self.split_nIF[idx]], weight_s, bias=None)

                out_adc = psum_quant_merge(out_adc, out_tmp,
                                            pbits=self.pbits, step=self.pstep, 
                                            half_num_levels=self.phalf_num_levels, 
                                            pbound=self.pbound, center=self.center, weight=1,
                                            groups=self.split_groups, pzero=self.pzero)

                # weight output summation
                if self.mapping_mode == 'two_com':
                    if wsplit_num == wbit+1:
                        output -= out_adc
                    else:
                        output = out_adc if wbit == 0 else output + out_adc
                elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
                    output = out_adc if wbit == 0 else output - out_adc
                else:
                    output = out_adc if wbit == 0 else output + out_adc
                out_adc = None

            if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                out_one = (-G_min/delta_G) * self._split_forward(input, w_one, ignore_bias=True, infer_only=True, merge_group=True)
                output -= out_one
      
        # add bias
        if self.bias is not None:
            output += self.bias
        
        # output_real = F.linear(input, qweight, bias=None)
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, ax = plt.subplots(ncols=2, figsize=(20, 12))

        # sns.histplot((qweight/w_scale).detach().cpu().numpy().ravel(), ax=ax[0], alpha=0.5)
        # sns.histplot((bweight).detach().cpu().numpy().ravel(), ax=ax[1], alpha=0.5)
        # ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=14)
        # ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=14)
        # ax[0].set_title(f'Before Noise', loc='right', fontsize=16)
        # ax[1].set_title(f'After Noise', loc='right', fontsize=16)
        # ax[0].set_xlabel('Weight', fontsize=16)
        # ax[1].set_xlabel('Weight', fontsize=16)
        # ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=14)
        # ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=14)

        # plt.savefig(os.getcwd() +f"/graph/ReRAM/NAT_Layer{self.layer_idx}_weight.png")

        # import pdb; pdb.set_trace()

        return output

    def forward(self, input):

        if self.wbits == 32:
            return F.linear(input, self.weight, bias=self.bias)
        
        if self.bitserial_log:
            return self._bitserial_log_forward(input)
        else:
            return self._bitserial_comp_forward(input)

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        s =  'in_features={}, out_features={}, bias={}, wbits={}, wbit_serial={}, split_groups={}, '\
            'mapping_mode={}, cbits={}, psum_mode={}, pbits={}, pbound={}, '\
            'noise={}, bitserial_log={}, layer_idx={}'\
            .format(self.in_features, self.out_features, self.bias is not None, self.wbits, self.wbit_serial,
            self.split_groups, self.mapping_mode, self.cbits, self.psum_mode, self.pbits, self.pbound, 
            self.is_noise, self.bitserial_log, self.layer_idx)
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

def set_PsumBinary_log(model, pbits, pclipmode, pclip=None, psigma=None, checkpoint=None, pquant_idx=None, pbound=None, center=None, log_file=False):
    print("start setting Bitserial layers log bitplane info")
    counter = 0
    for m in model.modules():
        if type(m).__name__ in ['PsumBinConv' , 'PsumBinLinear']:
            m.layer_idx = counter
            if (pquant_idx is None) or (counter == pquant_idx):
                m.bitserial_log = log_file
                m.checkpoint = checkpoint
                m.pclipmode = pclipmode
                m.setting_pquant_func(pbits, center, pbound)
                m.pclip = pclip
                m.psigma = psigma
                print("finish setting {}, idx: {}".format(type(m).__name__, counter))
            else:
                print(f"pass {m} with counter {counter}")
            counter += 1

def unset_PsumBinary_log(model):
    print("start unsetting Bitserial layers log bitplane info")
    counter = 0
    for m in model.modules():
        if type(m).__name__ in ['PsumBinConv' , 'PsumBinLinear']:
            m.bitserial_log = False
            print("finish log unsetting {}, idx: {}".format(type(m).__name__, counter))
            counter += 1

def set_PsumBinary_layer(model, pquant_idx, pbits=32, center=[]):
    ## set block for bit serial computation
    print("start setting conv/fc bitserial layer")
    counter = 0
    for m in model.modules():
        if type(m).__name__ in ['PsumBinConv' , 'PsumBinLinear']:
            if counter == pquant_idx:
                m.reset_layer(pbits=pbits, center=center)
            counter += 1
    print("finish setting conv/fc bitserial layer ")

def set_Qact_bitserial(model, pquant_idx, abit_serial=True):
    ## set quantact bitserial
    print("start setting quantact bitserial")
    counter = 0
    for m in model.modules():
        if type(m).__name__ is ['Q_act', 'QLeakyReLU']:
            if counter == pquant_idx:
                m.bitserial = abit_serial
            counter += 1
    print("finish setting quantact bitserial ")

def set_PsumBin_Noise_injection(model, weight=False, wsym=False, hwnoise=True, cbits=4, mapping_mode=None, co_noise=0.01, noise_type='prop', res_val='rel', w_format='weight', max_epoch=-1):
    for name, module in model.named_modules():
        if isinstance(module, (PsumBinConv, PsumBinLinear)) and weight and hwnoise:
            if module.wbits != 32:
                module.is_noise = True

                if noise_type == 'grad':
                    assert max_epoch != -1, "Enter max_epoch in hwnoise_initialize function"
                if hwnoise:
                    module._cell_noise_init(cbits=cbits, mapping_mode=mapping_mode, wsym=wsym, co_noise=co_noise, noise_type=noise_type, res_val=res_val, w_format=w_format, max_epoch=max_epoch)

def count_ArrayMaxV(wbits, cbits, mapping_mode, arraySize):
    if mapping_mode == '2T2R':
        if cbits >= (wbits-1):
            aMaxV = (2**(wbits-1)-1) * arraySize
        else:
            assert False, 'This file does not support this case cbits_{} < wbits_{}'.format(cbits, wbits)
    elif mapping_mode == 'two_com':
        if cbits >= (wbits-1):
            aMaxV = (2**(wbits-1)-1) * arraySize
        else:
            assert False, 'This file does not support this case cbits_{} < wbits_{}'.format(cbits, wbits)
    elif mapping_mode == 'ref_d':
        if cbits >= wbits:
            aMaxV = (2**(wbits)-1) * arraySize
        else:
            assert False, 'This file does not support this case cbits_{} < wbits_{}'.format(cbits, wbits)
    else:
        assert False, 'This file does not support the mapping_mode {}'.format(mapping_mode)
    
    return aMaxV