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
from .quantized_basic_modules import psum_quant_merge, psum_quant
from .bitserial_modules import *
from .split_modules import *

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

def calculate_groups_channels(arraySize, channels, kernel):
    if arraySize > 0:
        # groups
        groups = int(np.ceil(channels / np.floor(arraySize / (kernel**2))))
    else:
        groups = 1

    return groups

class TPsumQConv(SplitConv):
    """
        Quant(LSQ)Conv + Psum quantization
    """
    def __init__(self, in_channels, out_channels, wbits=32, kernel_size=3, stride=1, padding=0, groups=1, symmetric=False, bias=False, padding_mode='zeros', 
                arraySize=128, wbit_serial=False, mapping_mode='none', psum_mode='sigma', pclipmode='Layer', cbits=None):
        super(TPsumQConv, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, groups=groups, bias=bias)
        # for Qconv
        self.wbits = wbits
        self.wbit_serial = wbit_serial
        self.abit_serial = True if self.wbit_serial else False
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

        self.quan_w_fn = LSQReturnScale(bit=self.wbits, half_range=False, symmetric=symmetric, per_channel=False)

        # for split
        # self.split_nIC = split_conv(self.weight, arraySize)
        # channel-split 
        self.split_groups = calculate_groups_channels(arraySize, self.in_channels, self.kernel_size[0])
        self.nIC_list = split_conv(self.weight, arraySize)
        self.group_in_channels = self.nIC_list[0]
        self.group_move_in_channels = self.nIC_list

        # sweight
        # sweight = torch.Tensor(self.out_channels*self.split_groups, self.group_in_channels, kernel_size, kernel_size)
        # self.register_buffer('sweight', sweight)

        # for psum quantization
        self.mapping_mode = mapping_mode # Array mapping method [none, 2T2R, two_com, ref_d]]
        self.arraySize = arraySize
        self.cbits = cbits # Cell bits [multi, binary]
        self.psum_mode = psum_mode
        self.model_mode = None
        self.pclipmode = pclipmode
        self.pbits = 32
        # for scan version
        self.pstep = None
        self.pzero = None  # contain zero value (True)
        self.center = None
        self.pbound = None
        # for sigma version
        self.pclip = 'sigma'
        self.prange = 3
        self.weight_chunk = None
        # for pseudo-nosie training
        if (psum_mode == 'retrain') or (psum_mode == 'fix'):
            self.setting_alpha(pclipmode=pclipmode)
        self.noise_comb = False

        # for noise option
        self.is_noise = False
        self.co_noise = 0
        self.ratio = 100
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
            if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'ref_a'):
                self.pstep = self.pbound / ((2.**(self.pbits - 1)) - 1)
            else:
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
    
    def setting_alpha(self, pclipmode='Layer'):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.register_buffer('init_state', torch.zeros(1, device=device))
        if pclipmode == 'Layer':
            self.alpha = nn.Parameter(torch.Tensor(1)[0].to(device))
        elif pclipmode == 'Array':
            self.alpha = nn.Parameter(torch.zeros((self.split_groups, 1, 1, 1, 1), device=device))
        else:
            assert False, "pclipmode check, this pclipmode is {}".format(pclipmode)

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
            if self.is_noise and ((not self.training) or (self.training and self.w_format == 'state')):
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
            if self.is_noise and ((not self.training) or (self.training and self.w_format == 'state')):
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

        out_round = output.round()
        return (out_round - output).detach() + output, split_num 

    def setting_fix_range(self):
        if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'ref_a'):
            minPsum = self.arraySize * 2 * (self.wbits-1)
            minC = -(minPsum / self.prange) # Actually, level of negative range is small then positive 
            midC = 0
        elif (self.mapping_mode == 'ref_d'):
            midPsum = self.arraySize * 2 * (self.wbits-1)
            minC = 0
            midC = midPsum/self.prange
        else:
            assert False, "Not designed {} mode".format(self.mapping_mode)
        
        if self.info_print:
            print(f'Layer{self.layer_idx} information | pbits {self.pbits} | prange {self.prange} | Clip Min: {minC} | Mid: {midC} |')
            self.info_print = False

        return minC, midC
    
    # store weight magnitude for in-mem computing mimic 
    ## Assume that cell bits are enough
    def _output_magnitude(self, abit, wbit, split_num):
        multi_scale = 1
        if self.mapping_mode == "two_com":
            w_mag = 2**(self.wbits-1) if (wbit+1)==split_num else 2**wbit
            if self.cbits > 1:
                multi_scale = 2**(self.wbits-1)-1 if (wbit+1)==split_num else 1
        else:
            w_mag = 1

        out_mag = int(w_mag * (2**abit))
        return out_mag, multi_scale
    
    def _cell_noise_init(self, cbits, mapping_mode, co_noise=0.01, noise_type='prop', res_val='rel', shrink=None, retention=False, deltaG=None, w_format="weight", max_epoch=-1):
        self.w_format = 'state' if res_val == 'abs' or noise_type == 'meas' else 'weight'
        # wbits = Parameter(torch.Tensor(1).fill_(self.wbits), requires_grad=False).round().squeeze()
        # for inference 
        if not (noise_type == 'prop' or 'interp'):
            noise_type = 'prop'
        self.noise_cell_log = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type, res_val=res_val, shrink=shrink, retention=False, w_format=self.w_format)
        self.noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type, res_val=res_val, shrink=shrink, retention=retention, set_deltaG=deltaG, w_format=self.w_format)
        self.noise_cell_inf = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type, res_val='abs', shrink=shrink, retention=retention, set_deltaG=deltaG, w_format="state")

    def init_form(self, x, half_levels, psum_scale = 1):
        if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'PN') or (self.mapping_mode == 'ref_a'):
            # self.alpha.data.fill_((x.detach().abs().mean())+x.detach().std())
            # alpha = ((x.detach().abs().mean() * 2 / (half_levels ** 0.5)))
            if self.pclipmode == 'Layer':
                # alpha = ((x.detach().abs().mean() * 2 / (half_levels ** 0.5)))
                alpha = (x.detach().abs().std() * 3 / (half_levels))
                self.alpha.data.fill_(np.log(np.exp(alpha.item())-1))   # softplus initialization 
            else: 
                alpha = x.detach().abs().std(dim=list(range(1, x.dim())), keepdim=True) * 3 / (half_levels)
                self.alpha.data.copy_(torch.log(torch.exp(alpha)-1))

            with torch.no_grad():
                real_alpha = F.softplus(self.alpha) / psum_scale
                print("{} alpha: {} {}".format(self.layer_idx, self.alpha.view(-1), real_alpha.view(-1)))

        else:
            assert False, "{} mode is not considered in training clipping parameter".format(self.mapping_mode)


    def _bitserial_log_forward(self, input, weight=None, short_path=None):
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
            if weight is None:
                weight = self.weight
            qweight, w_scale = self.quan_w_fn(weight)

            if short_path is None:
                sinput, a_scale, abits = Bitserial.bitserial_act(input, debug=False) # short_path parameter does not exist
                a_shift = None
            else:
                sinput, a_scale, abits = Bitserial.bitserial_act(input, debug=False) # short_path parameter does not exist

            if self.is_noise and self.w_format=="weight":
                qweight = self.noise_cell_log(qweight/w_scale)


            ## get dataframe
            logger = f'{self.checkpoint}/layer{self.layer_idx}_mac_static.pkl'
            df = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

            # logger_scaled = f'{self.checkpoint}/layer{self.layer_idx}_wabitplane_mac_static_scaled_lastbatch.pkl'
            # df_scaled = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

            layer_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
            network_hist = f'{self.checkpoint}/hist/network_hist.pkl'

            #plane hist
            ### in-mem computation mimic (split conv & psum quant/merge)
            input_chunk = torch.chunk(sinput, abits, dim=1)

            ### Cell noise injection + Cell conductance value change
            if self.is_noise:
                bweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)
                bweight = self.noise_cell_log(bweight)
                
                weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
                
                ## make the same weight size as qweight
                if ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')) and (self.w_format=='state'):
                    bweight = weight_chunk[0] - weight_chunk[1]
                    wsplit_num = 1
                elif self.mapping_mode == 'two_com':
                    ## [TODO] two_com split weight format check
                    delta_G, G_min = self.noise_cell_log.get_deltaG(G_min=True)
                    w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

                self.sweight = conv_sweight_cuda.forward(self.sweight, bweight, self.group_in_offset, self.split_groups)
                weight_chunk = torch.chunk(self.sweight, wsplit_num, dim=1)
            else:
                self.sweight = conv_sweight_cuda.forward(self.sweight, qweight, self.group_in_offset, self.split_groups)
                sweight, wsplit_num = self._weight_bitserial(self.sweight, w_scale, cbits=self.cbits)
                weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)

            # in-memory computing parameter computing
            psum_scale = w_scale * a_scale 
            out_tmp = None
            layer_hist_dict = {}
            for abit, input_s in enumerate(input_chunk):
                abitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_a:{abit}_hist.pkl'
                a_hist_dict = {}
                for wbit, weight_s in enumerate(weight_chunk):
                    wabitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_w:{wbit}_a:{abit}_hist.pkl'
                    wa_hist_dict = {}
                    out_tmp = self._split_forward(input_s, weight_s, padded=True, ignore_bias=True,
                                                    weight_is_split=True, infer_only=True)
                    # out_tmp = F.conv2d(input_s[:,nIC_cnt:nIC_cnt+self.split_nIC[idx],:,:], weight_s, bias=self.bias,
                    #             stride=self.stride, dilation=self.dilation, groups=self.groups)

                    out_mag, multi_scale = self._output_magnitude(abit, wbit, wsplit_num)
                    out_array = out_tmp.round()*multi_scale/out_mag
                    ## NOTE
                    df.loc[bitplane_idx] = [wbit, abit,
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
                        # get w_hist
                        if val in a_hist_dict.keys():
                            a_hist_dict[val] += count
                        else:
                            a_hist_dict[val] = count

                    # save wabitplane_hist
                    df_hist = pd.DataFrame(list(wa_hist_dict.items()), columns = ['val', 'count'])
                    # wabitplane hist
                    if os.path.isfile(wabitplane_hist):
                        print(f'[{self.layer_idx}]Update wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
                        df_wabitplane_hist = pd.read_pickle(wabitplane_hist) 
                        df_merge = pd.merge(df_wabitplane_hist, df_hist, how="outer", on="val")
                        df_merge = df_merge.replace(np.nan, 0)
                        df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                        df_merge = df_merge[['val', 'count']]
                        df_merge.to_pickle(wabitplane_hist)
                    else:
                        print(f'[{self.layer_idx}]Create wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
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

                # save abitplane_hist
                df_hist = pd.DataFrame(list(a_hist_dict.items()), columns = ['val', 'count'])
                # wbitplane hist
                if os.path.isfile(abitplane_hist):
                    print(f'[{self.layer_idx}]Update abitplane_hist for a:{abit} ({abitplane_hist})')
                    df_abitplane_hist = pd.read_pickle(abitplane_hist) 
                    df_merge = pd.merge(df_abitplane_hist, df_hist, how="outer", on="val")
                    df_merge = df_merge.replace(np.nan, 0)
                    df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                    df_merge = df_merge[['val', 'count']]
                    df_merge.to_pickle(abitplane_hist)
                else:
                    print(f'[{self.layer_idx}]Create abitplane_hist for a:{abit} ({abitplane_hist})')
                    df_hist.to_pickle(abitplane_hist)

                # update layer hist
                for val, count in a_hist_dict.items():
                    if val in layer_hist_dict.keys():
                        layer_hist_dict[val] += count
                    else:
                        layer_hist_dict[val] = count

                if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                    out_one = (-G_min/delta_G) * self._split_forward(input_s, w_one, padded=True, ignore_bias=True,
                                                    weight_is_split=True, infer_only=True, merge_group=True)
                    out_wsum -= out_one
                output = out_wsum if abit ==0 else output+out_wsum

            # restore output's scale
            output = output * psum_scale

            # restore output's shift
            if a_shift is not None:
                weight_sum = torch.sum(qweight, (3, 2, 1))
                shift = weight_sum * a_shift
                psum_shift = torch.unsqueeze(shift, 0)
                psum_shift = torch.unsqueeze(psum_shift, 2)
                psum_shift = torch.unsqueeze(psum_shift, 2)

                output = output + psum_shift

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
                maxVal =  (abs(mean) + self.prange*std).round() 
                minVal = (abs(mean) - self.prange*std).round()
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

        # get quantization parameter and input bitserial
        qweight, w_scale = self.quan_w_fn(self.weight)

        if self.wbit_serial:
            with torch.no_grad():
                if self.abit_serial:
                    sinput, a_scale, abits = Bitserial.bitserial_act(input, debug=False) # short_path parameter does not exist
                else:
                    abits, a_scale = Bitserial.get_abit_scale()
                    sinput = input / a_scale
                    abits = 1

                input_chunk = torch.chunk(sinput, abits, dim=1)

                psum_scale = w_scale * a_scale

                if self.psum_mode == 'sigma':
                    minVal, _, midVal = self._ADC_clamp_value()
                    self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
                elif self.psum_mode == 'fix':
                    minVal, midVal = self.setting_fix_range() # half range
                    self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
                    self.fix_alpha = F.softplus(self.alpha)
                    self.pstep = self.fix_alpha * self.pstep
                elif self.psum_mode == 'retrain':
                    # self.alpha.data = self.alpha.data.round()
                    # self.pstep = self.alpha
                    self.pstep = F.softplus(self.alpha) /psum_scale

                else:
                    assert False, 'This script does not support {self.psum_mode}'

                ### in-mem computation mimic (split conv & psum quant/merge)

                ### Cell noise injection + Cell conductance value change
                ### in-mem computation programming (weight constant-noise)
                if (self.weight_chunk is None):
                    if self.is_noise:
                        sweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)
                        sweight = self.noise_cell_inf(sweight)
                        
                        weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)
                        
                        ## make the same weight size as qweight
                        if ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                            sweight = weight_chunk[0] - weight_chunk[1]
                            wsplit_num = 1
                        elif self.mapping_mode == 'two_com':
                            ## [TODO] two_com split weight format check
                            delta_G, G_min = self.noise_cell_inf.get_deltaG(G_min=True)
                            w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

                        weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)
                    else:
                        sweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)
                        weight_chunk = torch.chunk(sweight, wsplit_num, dim=1)
                    
                    self.weight_chunk = weight_chunk
                else:
                    weight_chunk = self.weight_chunk
                    wsplit_num = 1

                # parameter computing

                out_adc = None
                for abit, input_s in enumerate(input_chunk):
                    for wbit, weight_s in enumerate(weight_chunk):
                        out_tmp = self._split_forward(input_s, weight_s, padded=True, ignore_bias=True, cat_output=False,
                                                weight_is_split=True, infer_only=True, channel=True)

                        out_mag, multi_scale = self._output_magnitude(abit, wbit, wsplit_num)
                        out_adc = psum_quant_merge(out_adc, out_tmp,
                                                    pbits=self.pbits, step=self.pstep, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    pbound=self.pbound, center=self.center, weight=out_mag/multi_scale,
                                                    groups=self.split_groups, pzero=self.pzero, psum_mode=self.psum_mode)

                        # weight output summation
                        if self.mapping_mode == 'two_com':
                            if wsplit_num == wbit+1:
                                out_wsum -= out_adc
                            else:
                                out_wsum = out_adc if wbit == 0 else out_wsum + out_adc
                        elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
                            out_wsum = out_adc if wbit == 0 else out_wsum - out_adc
                        else:
                            out_wsum = out_adc if wbit == 0 else out_wsum + out_adc
                        out_adc = None

                    if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                        out_one = (-G_min/delta_G) * self._split_forward(input_s, w_one, padded=True, ignore_bias=True, cat_output=False,
                                                weight_is_split=True, infer_only=True, merge_group=True)
                        out_wsum -= out_one
                    output = out_wsum if abit == 0 else output+out_wsum

                # restore output's scale
                output = output * psum_scale
        else:
            if not self.abit_serial:
                # in-mem computation mimic (split conv & psum quant/merge)
                self.pbits = 32
                output = self._split_forward(input, qweight, padded=True, ignore_bias=True,  merge_group=True)
            else:
                assert False, "we do not support act serial only model"

        # add bias
        if self.bias is not None:
            output += self.bias

        # output_real = F.conv2d(input, qweight, bias=self.bias,
        #                         stride=self.stride, dilation=self.dilation, groups=self.groups)

        # import pdb; pdb.set_trace()

        return output
  
    
    def _bitserial_retrain_forward(self, input):
        # delete padding_shpe & additional padding operation by matching padding/stride format with nn.Conv2d

        if self.padding > 0:
            if self.padding_mode == 'neg':
                self.padding_value = float(input.min()) 
            padding_shape = (self.padding, self.padding, self.padding, self.padding)
            input = Pad.pad(input, padding_shape, self.padding_mode, self.padding_value)

        # get quantization parameter and input bitserial
        qweight, w_scale = self.quan_w_fn(self.weight)
        
        if self.abit_serial:
            sinput, a_scale, split_abit = Bitserial.bitserial_act(input, training=True) # short_path parameter does not exist
        else:
            abits, a_scale = Bitserial.get_abit_scale()
            sinput = input / a_scale
            split_abit = 1

        input_chunk = torch.chunk(sinput, split_abit, dim=1)

        psum_scale = w_scale * a_scale

        if self.psum_mode == 'sigma':
            minVal, _, midVal = self._ADC_clamp_value()
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
            step_train = False
        elif self.psum_mode == 'fix':
            minVal, midVal = self.setting_fix_range() # half range
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
            step_train = True 
            self.fix_alpha = F.softplus(self.alpha)
        elif self.psum_mode == 'retrain':
            # if self.init_state == 1:
            self.pstep = F.softplus(self.alpha) /psum_scale
                # self.pstep = (self.pstep - self.pstep/psum_scale).detach()+self.pstep/psum_scale
            # self.pstep = self.alpha
            step_train = True
        else:
            assert False, 'This script does not support {self.psum_mode}'

        ### Cell noise injection + Cell conductance value change
        ### in-mem computation programming (weight constant-noise)

        qweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)

        with torch.no_grad():
            if self.is_noise:
                nweight = self.noise_cell(qweight)
                
                weight_chunk = torch.chunk(nweight, wsplit_num, dim=1)
                
                ## make the same weight size as qweight
                if ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')) and (self.w_format=='state'):
                    nweight = weight_chunk[0] - weight_chunk[1]
                    wsplit_num = 1

                qweight.copy_(nweight)

        weight_chunk = torch.chunk(qweight, wsplit_num, dim=1)

        # parameter computing

        out_adc = None
        for abit, input_s in enumerate(input_chunk):
            for wbit, weight_s in enumerate(weight_chunk):
                # partial sum output before ADC 
                out_tmp = self._split_forward(input_s, weight_s, padded=True, ignore_bias=True, cat_output=True,
                                        weight_is_split=True, split_train=True, channel=True)
                if (self.psum_mode == 'retrain') and (self.init_state == 0):
                    # self.init_form(sum(out_tmp)/self.split_groups, self.phalf_num_levels)
                    with torch.no_grad():
                        out_init = torch.stack(torch.chunk(out_tmp, self.split_groups, dim=1))
                    self.init_form(out_init*psum_scale, self.phalf_num_levels, psum_scale)
                    self.init_state.fill_(1)
                    self.pstep = F.softplus(self.alpha) /psum_scale

                    # self.pstep = self.alpha

                out_mag, multi_scale = self._output_magnitude(abit, wbit, wsplit_num)
                if self.psum_mode == 'fix':
                    out_adc =  psum_quant_merge_train(out_adc, out_tmp / self.pstep,
                                                    pbits=self.pbits, step=self.fix_alpha, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    center=self.center, weight=out_mag/multi_scale,
                                                    groups=self.split_groups, pzero=self.pzero, step_train=step_train) * self.pstep
                else:
                    out_adc =  psum_quant_merge_train(out_adc, out_tmp,
                                                    pbits=self.pbits, step=self.pstep, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    center=self.center, weight=out_mag/multi_scale,
                                                    groups=self.split_groups, pzero=self.pzero, step_train=step_train)
                if wbit == 0:
                    out_wsum = out_adc
                else:
                    out_wsum += out_adc 

                out_adc = None

            if abit == 0:
                output = out_wsum
            else:
                output += out_wsum

        # restore output's scale
        output = output * psum_scale

        # add bias
        if self.bias is not None:
            output += self.bias

        # output_real = F.conv2d(input, qweight, bias=self.bias,
        #                         stride=self.stride, dilation=self.dilation, groups=self.groups)

        return output

    def _bitserial_pnq_forward(self, input):

        # delete padding_shpe & additional padding operation by matching padding/stride format with nn.Conv2d
        if self.padding > 0:
            if self.padding_mode == 'neg':
                self.padding_value = float(input.min()) 
            padding_shape = (self.padding, self.padding, self.padding, self.padding)
            input = Pad.pad(input, padding_shape, self.padding_mode, self.padding_value)

        # get quantization parameter and input bitserial 
        qweight, w_scale = self.quan_w_fn(self.weight)
        # weight_g = qweight.detach().clone()

        if self.abit_serial and (not self.noise_comb):
            sinput, a_scale, abits = Bitserial.bitserial_act(input, training=True) # short_path parameter does not exist
            split_abit = abits
        else:
            abits, a_scale = Bitserial.get_abit_scale()
            sinput = input / a_scale
            split_abit = 1 
        
        input_chunk = torch.chunk(sinput, split_abit, dim=1)

        psum_scale = w_scale * a_scale

        if self.psum_mode == 'sigma':
            minVal, _, midVal = self._ADC_clamp_value()
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
        elif self.psum_mode == 'fix':
            minVal, midVal = self.setting_fix_range() # half range
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
        elif self.psum_mode == 'retrain':
            self.pstep = F.softplus(self.alpha) /psum_scale
        else:
            assert False, 'This script does not support {self.psum_mode}'
        
        ### Cell noise injection + Cell conductance value change
        qweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)
        
        with torch.no_grad():
            if self.is_noise:
                nweight = self.noise_cell(qweight)
                weight_chunk = torch.chunk(nweight, wsplit_num, dim=1)
                
                if ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')) and (self.w_format=='state'):
                    nweight = weight_chunk[0] - weight_chunk[1]
                    wsplit_num = 1

                qweight.copy_(nweight)

        weight_chunk = torch.chunk(qweight, wsplit_num, dim=1)

        ### in-mem computation mimic (split conv & psum quant/merge)
        for abit, input_s in enumerate(input_chunk):
            for wbit, weight_s in enumerate(weight_chunk):
                out_tmp = torch.stack(self._split_forward(input_s, weight_s, padded=True, ignore_bias=True, cat_output=False,
                                        weight_is_split=True, split_train=True, channel=True), dim=0)
                
                # import seaborn as sns
                # import matplotlib.pyplot as plt
                # group_mean = out_tmp.mean((1, 2, 3, 4))
                # group_std = out_tmp.std((1, 2, 3, 4))
                # abit_mean.append(out_tmp.mean())
                # abit_std.append(out_tmp.std())
                # out_graph = out_tmp.view(self.split_groups, -1)
                # df = pd.DataFrame(out_graph.detach().cpu()).transpose()
                # df_data = pd.DataFrame(out_graph.detach().cpu().view(-1))
                # if abit == 0:
                #     df_abit = df_data
                # else:
                #     df_abit= pd.concat([df_abit, df_data], axis=1)

                # fig, ax = plt.subplots(figsize=(15, 10))
                # g = sns.catplot(data=df, ax=ax, errorbar=("pi", 100), aspect=1.2, kind="boxen", linewidth=0.6)
                # # df.add_prefix('Group_')
                # g.set_axis_labels('Group index', 'Psum Value')
                # g.fig.suptitle(f'Layer {self.layer_idx} abit: {abit}', color='gray', y=1.02)
                # plt.savefig(os.getcwd() +"/graph/ReRAM/Layer{}_abit{}_out_dist.png".format(self.layer_idx, abit), bbox_inches='tight')
                # plt.clf()                


                if (self.psum_mode == 'retrain') and (self.init_state == 0):
                    if not self.noise_comb:
                        self.init_form(out_tmp*psum_scale, self.phalf_num_levels, psum_scale)
                    else:
                        with torch.no_grad():
                            input_init = input_s.bool().to(input_s.dtype)
                            out_init = torch.stack(self._split_forward(input_init, weight_s, padded=True, ignore_bias=True, cat_output=False,
                                        weight_is_split=True, split_train=True, channel=True), dim=0)
                        self.init_form(out_init*psum_scale, self.phalf_num_levels, psum_scale)
                        del input_init, out_init
                        torch.cuda.empty_cache()
                    self.init_state.fill_(1)
                    self.pstep = F.softplus(self.alpha) / psum_scale
                elif (self.psum_mode == 'fix') and (self.init_state == 0):
                    alpha = 1 / (abs(minVal)/(3*out_tmp.std()))
                    self.alpha.data.fill_(np.log(np.exp(alpha.item())-1)) 
                    self.fix_alpha = F.softplus(self.alpha)
                    self.init_state.fill_(1)
                    print("[Layer {}] Set fix alpha parameter {}, softplus{}".format(self.layer_idx, 1/alpha, self.fix_alpha))

                # pseudo-noise generation 
                with torch.no_grad():
                    Qp = self.phalf_num_levels 
                    Qn = 1 - self.phalf_num_levels

                    if not self.noise_comb:
                        noise = (torch.rand_like(out_tmp) - 0.5)
                        # noise = (2**abit * sigma)**2 * (torch.rand_like(out_tmp) - 0.5)
                    else: #4-bit at once time
                        sum_sigma = 0
                        sigma = math.sqrt(1/12)
                        for a in range(abits):
                            sum_sigma += (2**a * sigma)**2
                        noise = sum_sigma * torch.randn_like(out_tmp) #normal distribution approximation
                        Qp = (2**(abits)-1) * Qp
                        Qn = (2**(abits)-1) * Qn
                        # self.pstep  = (2**(abits)-1) * self.pstep
                if self.psum_mode == 'fix':
                    out_tmp /= self.fix_alpha
                
                out_tmp = out_tmp / self.pstep
                
                c1 = out_tmp >= Qp
                c2 = out_tmp <= Qn

                # out_tmp = torch.where(c1, Qp, torch.where(c2, Qn, out_tmp+noise))
                out_tmp = torch.where(c1, Qp, torch.where(c2, Qn, out_tmp+noise))*self.pstep
                
                if self.psum_mode == 'fix':
                    out_tmp *= self.fix_alpha

                if wbit == 0:
                    out_wsum = out_tmp.sum(dim=0)
                else: 
                    out_wsum += out_tmp.sum(dim=0)
            if abit == 0:
                output = out_wsum * (2**abit)
            else:
                output += out_wsum * (2**abit)

        # df_abit.columns = ['0', '1', '2', '3']
        # fig, ax = plt.subplots(figsize=(15, 10))
        # g = sns.catplot(data=df_abit, ax=ax, errorbar=("pi", 100), aspect=1.2, kind="boxen", linewidth=0.6)
        # # df.add_prefix('Group_')
        # g.set_axis_labels('Abit bit position', 'Psum Value')
        # g.fig.suptitle(f'Layer {self.layer_idx}', color='gray', y=1.02)
        # plt.savefig(os.getcwd() +"/graph/ReRAM/Layer{}_out_dist.png".format(self.layer_idx, abit), bbox_inches='tight')
        # plt.clf()  

        # restore output's scale
        output = output * psum_scale

        # output_real = F.conv2d(input, weight_g, bias=self.bias,
        #                         stride=self.stride, dilation=self.dilation, groups=self.groups)
        
        # import pdb; pdb.set_trace()
        # add bias
        if self.bias is not None:
            output += self.bias

        return output

    def forward(self, input):
        if self.wbits == 32:
            return F.conv2d(input, self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        if self.training:
            if self.model_mode == 'lsq_pst':
                return self._bitserial_retrain_forward(input)
            elif self.model_mode =='pnq_pst':
                return self._bitserial_pnq_forward(input)
            else:
                assert False, 'Only lsq_pst or pnq_pst mode can train partial sum, Check your model_mode {}'.format(self.model_mode)
        else:
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
        s += ', wbits={wbits}, wbit_serial={wbit_serial}'
        s += ', split_groups={split_groups}, mapping_mode={mapping_mode}, cbits={cbits}'
        s += ', model_mode={model_mode}, psum_mode={psum_mode}, pbits={pbits}, pbound={pbound}'
        s += ', noise={is_noise}'
        s += ', bitserial_log={bitserial_log}, layer_idx={layer_idx}'            
        return s.format(**self.__dict__)

class TPsumQLinear(SplitLinear):
    """
        Quant(LSQ)Linear + Psum quantization
    """
    def __init__(self, in_features, out_features, wbits, symmetric=False, bias=False,
                arraySize=128, wbit_serial=False, mapping_mode='none', psum_mode='sigma', pclipmode='Layer', cbits=None):
        super(TPsumQLinear, self).__init__(in_features, out_features, bias=bias)
        # for QLinear
        self.wbits = wbits
        self.wbit_serial = wbit_serial
        self.abit_serial = True if wbit_serial else False

        self.quan_w_fn = LSQReturnScale(bit=self.wbits, half_range=False, symmetric=symmetric, per_channel=False)

        # for split
        # self.split_nIF = split_linear(self.weight, arraySize)
        self.split_groups = calculate_groups(arraySize, in_features)
        if in_features % self.split_groups != 0:
            raise ValueError('in_features must be divisible by groups')
        self.group_in_features = int(in_features / self.split_groups)

        # for psum quantization
        self.mapping_mode = mapping_mode # Array mapping method [none, 2T2R, two_com, ref_d]
        self.arraySize = arraySize
        self.cbits = cbits # Cell bits [multi, binary]
        self.psum_mode = psum_mode
        self.pclipmode = pclipmode
        self.pbits = 32
        # for scan version
        self.pstep = None
        self.pzero = None # contain zero value (True)
        self.center = None
        self.pbound = arraySize if arraySize > 0 else self.fan_in
        # for sigma version
        self.pclip = 'sigma'
        self.prange = 3
        self.weight_chunk = None
        # for retrain version
        self.model_mode = None
        if (psum_mode == 'retrain') or (psum_mode == 'fix'):
            self.setting_alpha(pclipmode=pclipmode)
        # for pseudo-nosie training
        self.noise_comb = False

        # for noise option
        self.is_noise = False
        self.w_format = 'weight'

        # for logging
        self.bitserial_log = False
        self.layer_idx = -1
        self.checkpoint = None
        self.info_print = True
    
    def   setting_pquant_func(self, pbits=None, center=[], pbound=None):
        # setting options for pquant func
        if pbits is not None:
            self.pbits = pbits
        if pbound is not None:
            self.pbound = pbound
            # get pquant step size
            if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'ref_a'):
                self.pstep = self.pbound / ((2.**(self.pbits - 1)) - 1)
            else:
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

    def setting_alpha(self, pclipmode='Layer'):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.register_buffer('init_state', torch.zeros(1, device=device))
        if pclipmode == 'Layer':
            self.alpha = nn.Parameter(torch.Tensor(1)[0].to(device))
            # self.alpha = nn.Parameter(torch.zeros(1))
        elif pclipmode == 'Array':
            self.alpha = nn.Parameter(torch.zeros((self.split_groups, 1, 1), device=device))
        else:
            assert False, "pclipmode check, this pclipmode is {}".format(pclipmode)
    
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
            if self.is_noise and ((not self.training) or (self.training and self.w_format == 'state')):
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
            if self.is_noise and ((not self.training) or (self.training and self.w_format == 'state')):
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

        out_round = output.round()
        return (out_round - output).detach() + output, split_num 
    
    def setting_fix_range(self):

        if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'ref_a'):
            minPsum = self.arraySize * 2 * (self.wbits-1)
            minC = -(minPsum / self.prange)
            # minC = -(minPsum / self.prange)
            midC = 0
        elif (self.mapping_mode == 'ref_d'):
            midPsum = self.arraySize * 2 * (self.wbits-1)
            minC = 0
            midC = midPsum/self.prange
        else:
            assert False, "Not designed {} mode".format(self.mapping_mode)
        
        if self.info_print:
            print(f'Layer{self.layer_idx} information | pbits {self.pbits} | prange {self.prange} | Clip Min: {minC} | Mid: {midC} |')
            # print(f'Layer{self.layer_idx} information | pbits {self.pbits} | prange {self.prange} | Clip Min: {minC} | Mid: {midC} |')
            self.info_print = False

        return minC, midC
    
    # store weight magnitude for in-mem computing mimic 
    ## Assume that cell bits are enough
    def _output_magnitude(self, abit, wbit, split_num):
        multi_scale = 1
        if self.mapping_mode == "two_com":
            w_mag = 2**(self.wbits-1) if (wbit+1)==split_num else 2**wbit
            if self.cbits > 1:
                multi_scale = 2**(self.wbits-1)-1 if (wbit+1)==split_num else 1
        else:
            w_mag = 1

        out_mag = int(w_mag * (2**abit))
        return out_mag, multi_scale

    def _cell_noise_init(self, cbits, mapping_mode, co_noise=0.01, noise_type='prop', res_val='rel', shrink=None, retention=False, deltaG=None, max_epoch=-1):
        self.w_format = 'state' if res_val == 'abs' or noise_type == 'meas' else 'weight'
        # wbits = Parameter(torch.Tensor(1).fill_(self.wbits), requires_grad=False).round().squeeze()
        if self.psum_mode == 'sigma':
            self.noise_cell_log = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type, res_val=res_val, shrink=shrink, retention=False, w_format=self.w_format)
        self.noise_cell = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type, res_val=res_val, shrink=shrink, retention=retention, set_deltaG=deltaG, w_format=self.w_format)
        self.noise_cell_inf = Noise_cell(self.wbits, cbits, mapping_mode, co_noise, noise_type, res_val="abs", shrink=shrink, retention=retention, set_deltaG=deltaG, w_format="state")
    
    def init_form(self, x, half_levels, psum_scale=1):
        if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'PN') or (self.mapping_mode == 'ref_a'):
            # self.alpha.data.fill_((x.detach().abs().std()*3).ceil())
            # alpha = ((x.detach().abs().mean() * 2 / (half_levels ** 0.5)))
            if self.pclipmode == 'Layer':
                alpha = (x.abs().std() * 3 / (half_levels))
                # alpha = ((x.detach().abs().mean() * 2 / (half_levels ** 0.5)))
                self.alpha.data.fill_(np.log(np.exp(alpha.item())-1))
            else: 
                alpha = x.detach().abs().std(dim=list(range(1, x.dim())), keepdim=True) * 3 / (half_levels)
                self.alpha.data.copy_(torch.log(torch.exp(alpha)-1))
            
            with torch.no_grad():
                real_alpha = F.softplus(self.alpha) / psum_scale
                print("{} alpha: {} {}".format(self.layer_idx, self.alpha.view(-1), real_alpha.view(-1)))
        else:
            assert False, "{} mode is not considered in training clipping parameter"

    def _bitserial_log_forward(self, input):
        print(f'[layer{self.layer_idx}]: bitserial mac log')

        # local parameter setting
        bitplane_idx = 0

        # get quantization parameter and input bitserial 
        qweight, w_scale = self.quan_w_fn(self.weight)
        sinput, a_scale, abits = Bitserial.bitserial_act(input, debug=False)
        psum_scale = w_scale * a_scale

        if self.is_noise and self.w_format=="weight":
            qweight = self.noise_cell_log(qweight/w_scale)

        ## get dataframe
        logger = f'{self.checkpoint}/layer{self.layer_idx}_mac_static.pkl'
        df = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

        # logger_scaled = f'{self.checkpoint}/layer{self.layer_idx}_wabitplane_mac_static_scaled_lastbatch.pkl'
        # df_scaled = pd.DataFrame(columns=['wbits', 'abits', 'mean', 'std', 'min', 'max'])

        layer_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_hist.pkl'
        network_hist = f'{self.checkpoint}/hist/network_hist.pkl'
        
        layer_hist_dict = {}
        ### in-mem computation mimic (split conv & psum quant/merge)
        input_chunk = torch.chunk(sinput, abits, dim=1)

        bweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)

        ### Cell noise injection + Cell conductance value change
        if self.is_noise:
            bweight = self.noise_cell_log(bweight)
            weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
            
            if (self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a'):
                bweight = weight_chunk[0] - weight_chunk[1]
                wsplit_num = 1
            else:
                ## [TODO] two_com split weight format check
                delta_G, G_min = self.noise_cell_log.get_deltaG(G_min=True)
                w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

        weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)

        out_tmp = None
        for abit, input_s in enumerate(input_chunk):
            abitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_a:{abit}_hist.pkl'
            a_hist_dict = {}
            for wbit, weight_s in enumerate(weight_chunk):
                wabitplane_hist = f'{self.checkpoint}/hist/layer{self.layer_idx}_w:{wbit}_a:{abit}_hist.pkl'
                wa_hist_dict = {}
                out_tmp = self._split_forward(input_s, weight_s, ignore_bias=True, infer_only=True)
                # out_tmp = F.linear(input_s[:,nIF_cnt:nIF_cnt+self.split_nIF[idx]], weight_s, bias=None)

                out_mag, multi_scale = self._output_magnitude(abit, wbit, wsplit_num)
                out_array = out_tmp.round()*multi_scale/out_mag
                                
                ## NOTE
                df.loc[bitplane_idx] = [wbit, abit,
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
                    # get w_hist
                    if val in a_hist_dict.keys():
                        a_hist_dict[val] += count
                    else:
                        a_hist_dict[val] = count

                # save wabitplane_hist
                df_hist = pd.DataFrame(list(wa_hist_dict.items()), columns = ['val', 'count'])
                # wabitplane hist
                if os.path.isfile(wabitplane_hist):
                    print(f'[{self.layer_idx}]Update wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
                    df_wabitplane_hist = pd.read_pickle(wabitplane_hist) 
                    df_merge = pd.merge(df_wabitplane_hist, df_hist, how="outer", on="val")
                    df_merge = df_merge.replace(np.nan, 0)
                    df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                    df_merge = df_merge[['val', 'count']]
                    df_merge.to_pickle(wabitplane_hist)
                else:
                    print(f'[{self.layer_idx}]Create wabitplane_hist for w:{wbit}/a:{abit} ({wabitplane_hist})')
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

            # save abitplane_hist
            df_hist = pd.DataFrame(list(a_hist_dict.items()), columns = ['val', 'count'])
            # wbitplane hist
            if os.path.isfile(abitplane_hist):
                print(f'[{self.layer_idx}]Update abitplane_hist for a:{abit} ({abitplane_hist})')
                df_abitplane_hist = pd.read_pickle(abitplane_hist) 
                df_merge = pd.merge(df_abitplane_hist, df_hist, how="outer", on="val")
                df_merge = df_merge.replace(np.nan, 0)
                df_merge['count'] = df_merge['count_x'] + df_merge['count_y']
                df_merge = df_merge[['val', 'count']]
                df_merge.to_pickle(abitplane_hist)
            else:
                print(f'[{self.layer_idx}]Create abitplane_hist for a:{abit} ({abitplane_hist})')
                df_hist.to_pickle(abitplane_hist)

            # update layer hist
            for val, count in a_hist_dict.items():
                if val in layer_hist_dict.keys():
                    layer_hist_dict[val] += count
                else:
                    layer_hist_dict[val] = count

            if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                out_one = (-G_min/delta_G) * self._split_forward(input_s, w_one, ignore_bias=True, infer_only=True, merge_group=True)
                out_wsum -= out_one
            output = out_wsum if abit ==0 else output+out_wsum

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
                maxVal =  (abs(mean) + self.prange*std).round() 
                minVal = (abs(mean) - self.prange*std).round() 
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
        qweight, w_scale = self.quan_w_fn(self.weight)

        if self.wbit_serial:
            with torch.no_grad():
                if self.abit_serial:
                    sinput, a_scale, abits = Bitserial.bitserial_act(input, debug=False) # short_path parameter does not exist
                else:
                    abits, a_scale = Bitserial.get_abit_scale()
                    sinput = input / a_scale
                    abits = 1

                psum_scale = w_scale * a_scale

                if self.psum_mode == 'sigma':
                    minVal, maxVal, midVal = self._ADC_clamp_value()
                    self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
                elif self.psum_mode == 'fix':
                    minVal, midVal = self.setting_fix_range() # half range
                    self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
                    self.fix_alpha = F.softplus(self.alpha)
                    self.pstep = self.fix_alpha * self.pstep
                elif self.psum_mode == 'retrain':
                    # self.alpha.data = self.alpha.data.round()
                    self.pstep = F.softplus(self.alpha) /psum_scale
                    # self.pstep = self.alpha
                elif self.psum_mode == 'scan':
                    pass
                else:
                    assert False, 'This script does not support {self.psum_mode}'

                ### in-mem computation mimic (split conv & psum quant/merge)
                input_chunk = torch.chunk(sinput, abits, dim=1)

                bweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)
                ### Cell noise injection + Cell conductance value change
                if (self.weight_chunk is None) or self.training:
                    if self.is_noise:
                        bweight = self.noise_cell_inf(bweight)
                        weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
                        
                        if ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                            bweight = weight_chunk[0] - weight_chunk[1]
                            wsplit_num = 1
                        else:
                            ## [TODO] two_com split weight format check
                            delta_G, G_min = self.noise_cell_inf.get_deltaG(G_min=True)
                            w_one = torch.ones(size=weight_chunk[0].size()).to(weight_chunk[0].device)

                    weight_chunk = torch.chunk(bweight, wsplit_num, dim=1)
                    self.weight_chunk = weight_chunk
                else:
                    weight_chunk = self.weight_chunk

                out_adc = None
                for abit, input_s in enumerate(input_chunk):
                    for wbit, weight_s in enumerate(weight_chunk):
                        out_tmp = self._split_forward(input_s, weight_s, ignore_bias=True, cat_output=False, infer_only=True)
                        # out_tmp = F.linear(input_s[:,nIF_cnt:nIF_cnt+self.split_nIF[idx]], weight_s, bias=None)

                        out_mag, multi_scale = self._output_magnitude(abit, wbit, wsplit_num)

                        out_adc = psum_quant_merge(out_adc, out_tmp,
                                                    pbits=self.pbits, step=self.pstep, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    pbound=self.pbound, center=self.center, weight=out_mag/multi_scale,
                                                    groups=self.split_groups, pzero=self.pzero, psum_mode=self.psum_mode)

                        # weight output summation
                        if self.mapping_mode == 'two_com':
                            if wsplit_num == wbit+1:
                                out_wsum -= out_adc
                            else:
                                out_wsum = out_adc if wbit == 0 else out_wsum + out_adc
                        elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'PN'):
                            out_wsum = out_adc if wbit == 0 else out_wsum - out_adc
                        else:
                            out_wsum = out_adc if wbit == 0 else out_wsum + out_adc
                        out_adc = None

                    if self.is_noise and not ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')):
                        out_one = (-G_min/delta_G) * self._split_forward(input_s, w_one, ignore_bias=True, infer_only=True, merge_group=True)
                        out_wsum -= out_one
                    output = out_wsum if abit == 0 else output+out_wsum

                # restore output's scale
                output = output * psum_scale
        else:
            abit_serial = Bitserial.abit_serial()
            if not abit_serial:
                # in-mem computation mimic (split linear & psum quant/merge)
                self.pbits = 32
                output = self._split_forward(input, qweight, ignore_bias=True, merge_group=True)
            else:
                assert False, "we do not support act serial only model"

        # add bias
        if self.bias is not None:
            output += self.bias
        
        # output_real = F.linear(input, qweight, bias=None)
        # import pdb; pdb.set_trace()

        return output
    
    def _bitserial_retrain_forward(self, input):

        # get quantization parameter and input bitserial 
        qweight, w_scale = self.quan_w_fn(self.weight)

        if self.abit_serial:
            sinput, a_scale, abits = Bitserial.bitserial_act(input, training=True) # short_path parameter does not exist
        else:
            abits, a_scale = Bitserial.get_abit_scale()
            sinput = input / a_scale
            abits = 1

        psum_scale = w_scale * a_scale

        if self.psum_mode == 'sigma':
            minVal, maxVal, midVal = self._ADC_clamp_value()
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
            step_train = False
        elif self.psum_mode == 'fix':
            minVal, midVal = self.setting_fix_range() # half range
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
            step_train = True 
            self.fix_alpha = F.softplus(self.alpha)
        elif self.psum_mode == 'retrain':
            self.pstep = F.softplus(self.alpha) /psum_scale
            step_train = True
        else:
            assert False, 'This script does not support {self.psum_mode}'
        ### in-mem computation mimic (split conv & psum quant/merge)
        input_chunk = torch.chunk(sinput, abits, dim=1)

        ### Cell noise injection + Cell conductance value change

        qweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)

        with torch.no_grad():
            if self.is_noise:
                nweight = self.noise_cell(qweight)
                weight_chunk = torch.chunk(nweight, wsplit_num, dim=1)
                
                if ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')) and (self.w_format=='state'):
                    nweight = weight_chunk[0] - weight_chunk[1]
                    wsplit_num = 1

                qweight.copy_(nweight)

        weight_chunk = torch.chunk(qweight, wsplit_num, dim=1)

        out_adc = None
        for abit, input_s in enumerate(input_chunk):
            for wbit, weight_s in enumerate(weight_chunk):

                out_tmp = self._split_forward(input_s, weight_s, ignore_bias=True, cat_output=True, split_train=True)
                # out_tmp = F.linear(input_s[:,nIF_cnt:nIF_cnt+self.split_nIF[idx]], weight_s, bias=None)
                
                if (self.psum_mode == 'retrain') and (self.init_state == 0):
                    # self.init_form(sum(out_tmp)/self.split_groups, self.phalf_num_levels)
                    with torch.no_grad():
                        out_init = torch.stack(torch.chunk(out_tmp, self.split_groups, dim=1))
                    self.init_form(out_init*psum_scale, self.phalf_num_levels, psum_scale)
                    self.init_state.fill_(1)
                    self.pstep = F.softplus(self.alpha) /psum_scale

                out_mag, multi_scale = self._output_magnitude(abit, wbit, wsplit_num)

                if self.psum_mode == 'fix':
                    out_adc =  psum_quant_merge_train(out_adc, out_tmp / self.pstep,
                                                    pbits=self.pbits, step=self.fix_alpha, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    center=self.center, weight=out_mag/multi_scale,
                                                    groups=self.split_groups, pzero=self.pzero, step_train=step_train) * self.pstep
                else:
                    out_adc = psum_quant_merge_train(out_adc, out_tmp,
                                                    pbits=self.pbits, step=self.pstep, 
                                                    half_num_levels=self.phalf_num_levels, 
                                                    center=self.center, weight=out_mag/multi_scale,
                                                    groups=self.split_groups, pzero=self.pzero, step_train=step_train)

                if wbit == 0:
                    out_wsum = out_adc
                else:
                    out_wsum += out_adc 

                out_adc = None

            if abit == 0:
                output = out_wsum
            else:
                output += out_wsum

        # restore output's scale
        output = output * psum_scale

        # add bias
        if self.bias is not None:
            output += self.bias
        
        # output_real = F.linear(input, qweight, bias=None)

        return output
    
    def _bitserial_pnq_forward(self, input):

        # get quantization parameter and input bitserial 
        qweight, w_scale = self.quan_w_fn(self.weight)

        if self.abit_serial and (not self.noise_comb):
            sinput, a_scale, abits = Bitserial.bitserial_act(input, training=True) # short_path parameter does not exist
            split_abit = abits
        else:
            abits, a_scale = Bitserial.get_abit_scale()
            sinput = input / a_scale
            split_abit = 1 
        
        input_chunk = torch.chunk(sinput, split_abit, dim=1)

        psum_scale = w_scale * a_scale

        if self.psum_mode == 'sigma':
            minVal, _, midVal = self._ADC_clamp_value()
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
        elif self.psum_mode == 'fix':
            minVal, midVal = self.setting_fix_range() # half range
            self.setting_pquant_func(pbits=self.pbits, center=minVal, pbound=midVal-minVal)
        elif self.psum_mode == 'retrain':
            self.pstep = F.softplus(self.alpha) /psum_scale
        else:
            assert False, 'This script does not support {self.psum_mode}'
        
        ### Cell noise injection + Cell conductance value change
        qweight, wsplit_num = self._weight_bitserial(qweight, w_scale, cbits=self.cbits)

        with torch.no_grad():
            if self.is_noise:
                nweight = self.noise_cell(qweight)
                weight_chunk = torch.chunk(nweight, wsplit_num, dim=1)
                
                if ((self.mapping_mode=='2T2R') or (self.mapping_mode=='ref_a')) and (self.w_format=='state'):
                    nweight = weight_chunk[0] - weight_chunk[1]
                    wsplit_num = 1

                qweight.copy_(nweight)

        weight_chunk = torch.chunk(qweight, wsplit_num, dim=1)

        ### in-mem computation mimic (split conv & psum quant/merge)
        for abit, input_s in enumerate(input_chunk):
            for wbit, weight_s in enumerate(weight_chunk):
                out_tmp = torch.stack(self._split_forward(input_s, weight_s, ignore_bias=True, cat_output=False, split_train=True), dim=0)

                # import seaborn as sns
                # import matplotlib.pyplot as plt
                # group_mean = out_tmp.mean((1, 2, 3, 4))
                # group_std = out_tmp.std((1, 2, 3, 4))
                # abit_mean.append(out_tmp.mean())
                # abit_std.append(out_tmp.std())
                # out_graph = out_tmp.view(self.split_groups, -1)
                # df = pd.DataFrame(out_graph.detach().cpu()).transpose()
                # df_data = pd.DataFrame(out_graph.detach().cpu().view(-1))
                # if abit == 0:
                #     df_abit = df_data
                # else:
                #     df_abit= pd.concat([df_abit, df_data], axis=1)

                # fig, ax = plt.subplots(figsize=(15, 10))
                # g = sns.catplot(data=df, ax=ax, errorbar=("pi", 100), aspect=1.2, kind="boxen", linewidth=0.6)
                # # df.add_prefix('Group_')
                # g.set_axis_labels('Group index', 'Psum Value')
                # g.fig.suptitle(f'Layer {self.layer_idx} abit: {abit}', color='gray', y=1.02)
                # plt.savefig(os.getcwd() +"/graph/ReRAM/Layer{}_abit{}_out_dist.png".format(self.layer_idx, abit), bbox_inches='tight')
                # plt.clf() 

                if (self.psum_mode == 'retrain') and (self.init_state == 0):
                    if not self.noise_comb:
                        self.init_form(out_tmp*psum_scale, self.phalf_num_levels, psum_scale)
                    else:
                        with torch.no_grad():
                            input_init = input_s.bool().to(input_s.dtype)
                            out_init = torch.stack(self._split_forward(input_init, weight_s, ignore_bias=True, cat_output=False, split_train=True), dim=0)
                        self.init_form(out_init*psum_scale, self.phalf_num_levels, psum_scale)
                        del input_init, out_init
                        torch.cuda.empty_cache()
                    self.init_state.fill_(1)
                    self.pstep = F.softplus(self.alpha) /psum_scale
                elif (self.psum_mode == 'fix') and (self.init_state == 0):
                    alpha = 1 / (abs(minVal)/(3*out_tmp.std()))
                    self.alpha.data.fill_(np.log(np.exp(alpha.item())-1)) 
                    self.fix_alpha = F.softplus(self.alpha)
                    self.init_state.fill_(1)
                    print("[Layer {}] Set fix alpha parameter {}, softplus{}".format(self.layer_idx, 1/alpha, self.fix_alpha))

                # pseudo-noise generation
                with torch.no_grad():
                    Qp = self.phalf_num_levels
                    Qn = 1 - self.phalf_num_levels

                    if not self.noise_comb:
                        noise = torch.rand_like(out_tmp) - 0.5
                    else: #4-bit at once time
                        sum_sigma = 0
                        sigma = math.sqrt(1/12)
                        for a in range(abits):
                            sum_sigma += (2**a * sigma)**2
                        noise = sum_sigma * torch.randn_like(out_tmp) #normal distribution approximation 
                        Qp = (2**(abits)-1) * Qp
                        Qn = (2**(abits)-1) * Qn
                        # self.pstep  = (2**(abits)-1) * self.pstep
                
                if self.psum_mode == 'fix':
                    out_tmp /= self.fix_alpha

                out_tmp = out_tmp / self.pstep

                c1 = out_tmp >= Qp
                c2 = out_tmp <= Qn

                out_tmp = torch.where(c1, Qp, torch.where(c2, Qn, out_tmp+noise))*self.pstep
                # out_tmp = torch.where(c1, self.pstep, torch.where(c2, -self.pstep, out_tmp+noise*delta))
                
                if self.psum_mode == 'fix':
                    out_tmp *= self.fix_alpha

                if wbit == 0:
                    out_wsum = out_tmp.sum(dim=0)
                else: 
                    out_wsum += out_tmp.sum(dim=0)

            # df_abit.columns = ['0', '1', '2', '3']
            # fig, ax = plt.subplots(figsize=(15, 10))
            # g = sns.catplot(data=df_abit, ax=ax, errorbar=("pi", 100), aspect=1.2, kind="boxen", linewidth=0.6)
            # # df.add_prefix('Group_')
            # g.set_axis_labels('Abit bit position', 'Psum Value')
            # g.fig.suptitle(f'Layer {self.layer_idx}', color='gray', y=1.02)
            # plt.savefig(os.getcwd() +"/graph/ReRAM/Layer{}_out_dist.png".format(self.layer_idx, abit), bbox_inches='tight')
            # plt.clf()
            # if self.layer_idx == 7:
            #     exit()

            if abit == 0:
                output = out_wsum * (2**abit)
            else:
                output += out_wsum * (2**abit)

        # restore output's scale
        output = output * psum_scale

        # add bias
        if self.bias is not None:
            output += self.bias
        
        # output_real = F.linear(input, qweight, bias=None)
        
        return output

    def forward(self, input):

        if self.wbits == 32:
            return F.linear(input, self.weight, bias=self.bias)
        
        if self.training:
            if self.model_mode == 'lsq_pst':
                return self._bitserial_retrain_forward(input)
            elif self.model_mode =='pnq_pst':
                return self._bitserial_pnq_forward(input)
            else:
                assert False, 'Only lsq_pst or pnq_pst mode can train partial sum, Check your model_mode {}'.format(self.model_mode)
        else:
            if self.bitserial_log:
                return self._bitserial_log_forward(input)
            else:
                return self._bitserial_comp_forward(input)

    def extra_repr(self):
        """Provides layer information, including wbits, when print(model) is called."""
        s =  'in_features={}, out_features={}, bias={}, wbits={}, wbit_serial={}, split_groups={}, '\
            'mapping_mode={}, cbits={}, model_mode={}, psum_mode={}, pbits={}, pbound={}, '\
            'noise={}, bitserial_log={}, layer_idx={}'\
            .format(self.in_features, self.out_features, self.bias is not None, self.wbits, self.wbit_serial,
            self.split_groups, self.mapping_mode, self.cbits, self.model_mode, self.psum_mode, self.pbits, self.pbound, 
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

def set_TBitSerial_log(model, pbits, pclipmode, model_mode, abit_serial=None, pclip=None, prange=None, checkpoint=None, pquant_idx=None, pbound=None, center=None, noise_comb=False, log_file=False):
    print("start setting Bitserial layers log bitplane info")
    counter = 0
    for m in model.modules():
        if type(m).__name__ in ['TPsumQConv' , 'TPsumQLinear']:
            m.layer_idx = counter
            if (pquant_idx is None) or (counter == pquant_idx):
                m.bitserial_log = log_file
                m.checkpoint = checkpoint
                m.model_mode = model_mode
                m.pclipmode = pclipmode
                m.abit_serial = abit_serial
                m.setting_pquant_func(pbits, center, pbound)
                m.pclip = pclip
                m.prange = prange
                if model_mode == 'pnq_pst':
                    m.noise_comb = noise_comb
                print("finish setting {}, idx: {}".format(type(m).__name__, counter))
            else:
                print(f"pass {m} with counter {counter}")
            counter += 1

def unset_TBitSerial_log(model):
    print("start unsetting Bitserial layers log bitplane info")
    counter = 0
    for m in model.modules():
        if type(m).__name__ in ['TPsumQConv' , 'TPsumQLinear']:
            m.bitserial_log = False
            print("finish log unsetting {}, idx: {}".format(type(m).__name__, counter))
            counter += 1

def set_tbitserial_layer(model, pquant_idx, wbit_serial=None, pbits=32, center=[]):
    ## set block for bit serial computation
    print("start setting conv/fc bitserial layer")
    counter = 0
    for m in model.modules():
        if type(m).__name__ in ['TPsumQConv' , 'TPsumQLinear']:
            if counter == pquant_idx:
                m.reset_layer(wbit_serial=wbit_serial, pbits=pbits, center=center)
            counter += 1
    print("finish setting conv/fc bitserial layer ")

def set_TNoise_injection(model, weight=False, hwnoise=True, cbits=4, mapping_mode=None, co_noise=0.01, noise_type='prop', res_val='rel', shrink=None, max_epoch=-1,
                        deltaG=None, retention=False, reten_kind='linear', reten_type='percent', reten_value=0):
    for name, module in model.named_modules():
        if isinstance(module, (TPsumQConv, TPsumQLinear)) and weight and hwnoise:
            if module.wbits != 32:
                module.is_noise = True

                if noise_type == 'grad':
                    assert max_epoch != -1, "Enter max_epoch in hwnoise_initialize function"
                if hwnoise:
                    module._cell_noise_init(cbits=cbits, mapping_mode=mapping_mode, co_noise=co_noise, noise_type=noise_type, res_val=res_val, shrink=shrink, retention=retention, deltaG=deltaG, max_epoch=max_epoch)
                    if retention:
                        module.noise_cell.retention_init(kind=reten_kind, type=reten_type, value=reten_value)
