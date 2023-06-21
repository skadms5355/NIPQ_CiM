import torch
import torch.nn as nn
import math
import time
import numpy as np
from torch.autograd.function import Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pandas as pd

def data_probability(x, adc_out, data_prob):
    """ Select data with noise condition """
    x_min = int(x.min())
    x_max = int(x.max())+1
    # start_time=time.time()
    for val in range(x_min, x_max):
        index = torch.where(x==val)
        num_replace = x[index].size()
        val_rep = np.random.choice(adc_out, num_replace, p=data_prob.loc[val].to_numpy())
        val_rep = torch.from_numpy(val_rep).type(x.dtype).cuda()
        x[index] = val_rep 
    # print(f'x_min {x_min} | x_max {x_max}| measure time {time.time()-start_time}')
    return x

def chip_quant_merge(x, adc_out, data_prob, groups, scale):
    """ Select data with noise condition """
    # input list shape 
    for g in range(0, groups):
        x[g].div_(scale)
        out_tmp = data_probability(x[g], adc_out, data_prob)
        out_tmp.mul_(scale)
        output = out_tmp if g==0 else output + out_tmp
    return output

def get_data(path, pbits, pqunat=None, maxVal=256, minVal=0):
    """ Acquire chip data and Scale data to step size """
    df_chip = pd.read_csv(path)
    df_chip = df_chip.set_index('MAC_result')
    df_chip = df_chip.drop(['Unnamed: 3842'], axis=1) # inhwan's csv file (for removing space)
    df_chip = df_chip.apply(pd.to_numeric)
    if pqunat:
        df_chip.columns = range(int(2**(pbits))) ## obtain adc_out value
        step = (maxVal - minVal) / (2.**pbits-1)
    else:
        step = 1
    # data_prob = df_chip.to_numpy() # shape is [MAC_value, ADC output]
    data_prob = df_chip # shape is [MAC_value, ADC output]

    adc_out = (df_chip.columns.to_numpy()*step).astype(int)
    return data_prob, adc_out
