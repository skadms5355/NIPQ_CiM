import torch
import torch.nn as nn

from .binarized_psum_modules import *
from .binarized_modules import nonlinear


class PConv_BN_Merge(nn.Module):
    def __init__(self, inplane, outplane, wbits, weight_clip, weight_scale, padding_mode, arraySize, mapping_mode, is_noise, noise_type,
                 abits, binary_mode, ste, offset, width, maxpool=False):
        super(PConv_BN_Merge, self).__init__()
        self.pconv = PsumBinConv(inplane, outplane, wbits=wbits, weight_clip=weight_clip, weight_scale=weight_scale, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode,
                        arraySize=arraySize, mapping_mode=mapping_mode, cbits=1, is_noise=is_noise, noise_type=noise_type)
        self.split_groups = self.pconv.split_groups
        self.BN = nn.BatchNorm2d(outplane*self.pconv.split_groups)
        self.Maxpooling = None
        if maxpool:
            self.Maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.qrelu = nonlinear(abits=abits, mode=binary_mode, ste=ste, offset=offset, width=width)

    def merge_output(self, x):
        tmp = torch.chunk(x, self.split_groups, dim=1)

        output = tmp[0]
        for i in range(1, self.split_groups):
            output += tmp[i]

        return output

    def forward(self, x):

        x = self.pconv(x)
        if self.Maxpooling is not None:
            x = self.Maxpooling(x)
        x = self.BN(x)
        x = self.merge_output(x)
        x = self.qrelu(x)

        return x 
    
class PLinear_BN_Merge(nn.Module):
    def __init__(self, infeatures, outfeatures, wbits, weight_clip, weight_scale, arraySize, mapping_mode, is_noise, noise_type,
                 abits, binary_mode, ste, offset, width):
        super(PConv_BN_Merge, self).__init__()
        self.plinear = PsumBinConv(infeatures, outfeatures, wbits=wbits, weight_clip=weight_clip, weight_scale=weight_scale,
                        arraySize=arraySize, mapping_mode=mapping_mode, cbits=1, is_noise=is_noise, noise_type=noise_type)
        self.split_groups = self.plinear.split_groups
        self.BN = nn.BatchNorm1d(outfeatures*self.plinear.split_groups)
        self.qrelu = nonlinear(abits=abits, mode=binary_mode, ste=ste, offset=offset, width=width)

    def merge_output(self, x):
        tmp = torch.chunk(x, self.split_groups, dim=1)

        output = tmp[0]
        for i in range(1, self.split_groups):
            output += tmp[i]

        return output

    def forward(self, x):

        x = self.plinear(x)
        x = self.BN(x)
        x = self.merge_output(x)
        x = self.qrelu(x)

        return x 
