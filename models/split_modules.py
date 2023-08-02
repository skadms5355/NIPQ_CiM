"""Custom network modules for convolution operations with convolution split

This module contains custom network modules for convolution split for computing-in-memory.
"""
import itertools
from .quantized_basic_modules import *
import math
import conv_sweight_cuda


class SplitConv(nn.Conv2d):
    """Split Conv module. It provide input split on nn.Conv2d

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        groups (int, optional): Number of split groups. Default: 1

    Shape:
        - Input: (N, C_in, H_in, W_in)
        - Output: (N, C_out, H_out, W_out)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                padding=0, padding_mode='zeros', groups=1, bias=False):
        super(SplitConv, self).__init__(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, padding_mode=padding_mode,\
                              bias=bias)
        self.split_groups = groups
        # define in/out channels for group
        self.kSpatial = kernel_size**2
        self.fan_in = in_channels * self.kSpatial
        if self.fan_in % self.split_groups != 0:
            raise ValueError('fan_in must be divisible by groups')
        self.group_fan_in = int(self.fan_in / self.split_groups)
        self.group_in_channels = int(np.ceil(in_channels / self.split_groups))
        residual = self.group_fan_in % self.kSpatial
        if residual != 0:
            if self.kSpatial % residual != 0:
                self.group_in_channels += 1

        # log move group for masking & group convolution
        self.group_move_in_channels = torch.zeros(self.split_groups-1, dtype=torch.int)
        group_in_offset = torch.zeros(self.split_groups, dtype=torch.int)
        self.register_buffer('group_in_offset', group_in_offset)

    def reset_groups(self, groups):
        if self.split_groups == groups:
            return
        assert self.split_groups != 1, 'reset group is not supported when groups is not 1'
        self.out_channels = int(self.out_channels / self.split_groups * groups)
        if self.fan_in % groups != 0:
            raise ValueError('fan_in must be divisible by groups')
        self.group_fan_in = int(self.fan_in * self.split_groups / groups)
        self.group_in_channels = int(np.ceil(self.in_channels * self.split_groups / groups))
        residual = self.group_fan_in % self.kSpatial
        if residual != 0:
            if self.kSpatial % residual != 0:
                self.group_in_channels += 1
        # add one more channels for the group if spatial is not divisible by the residual
        self.group_out_channels = int(self.out_channels * self.split_groups / groups)

        # sweight
        self.sweight.resize_(self.out_channels*groups, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        # reset split_groups
        self.split_groups = groups

        # log move group for masking & group convolution
        self.group_move_in_channels.resize_(groups-1).fill_(0)
        self.group_in_offset.resize_(groups)
        # get group conv info
        self._group_move_offset()

    def _group_move_offset(self):
        print("_group_mode_offset function")
        # required information
        kSpatial = self.kernel_size[0] * self.kernel_size[1]
        # offset of the input in the start channel dim for each group
        in_offset = 0
        end_in_channels = 0
        # initialize mask!
        for i in range(0, self.split_groups):
            # update end_in_channels
            if i == 0 :
                end_in_channels = self.group_in_channels
            else:
                end_in_channels += self.group_move_in_channels[i-1]
            # update start mask & fix group_move_in_channels 
            # if end_in_channels exceed self.in_channels
            if end_in_channels > self.in_channels:
                assert end_in_channels - self.in_channels == 1, \
                       'The maximum difference between end_in_channels & self.in_channels should be 1.'
                if self.fan_in % kSpatial != 0:
                    assert kSpatial % (self.fan_in % kSpatial) != 0, \
                       'end_in_channels can only exceed when kSpatial is not divisible by remain.'
                # move one channel back
                in_offset = in_offset + kSpatial
                self.group_move_in_channels[i-1] -= 1
                end_in_channels -= 1
            self.group_in_offset[i] = in_offset
            # update group_move_in_channels & next in_offset
            if i < self.split_groups - 1:
                # compute fully used channels
                in_residue = kSpatial - in_offset
                remain = self.group_fan_in - in_residue
                fulled_in_channels = np.floor(remain / kSpatial)
                self.group_move_in_channels[i] = fulled_in_channels + 1 # 1 is for in_residue channel
                # update offset
                in_offset = remain % kSpatial

        # check if we scanned all input channels
        assert end_in_channels == self.in_channels, \
               'end_in_channels and self.in_channels should be the same'


    def _split_forward(self, input, weight, padded=False, ignore_bias=False, cat_output=True, weight_is_split=False, infer_only=False, merge_group=False, binary=False):
        """Before this opertaion, weights should be masked!"""
        # pad the input if it is not padded 
        if padded is False:
            if self.padding_mode != 'zeros':
                input = F.pad(input, self._reversed_padding_repeated_twice,\
                              mode=self.padding_mode)
                padded = True
        if padded is True:
            padding = 0
        else:
            padding = self.padding

        if not binary:
            output = None

            if self.training and (not infer_only):
                output = F.conv2d(input, weight, bias=None,
                            stride=self.stride, padding=padding, dilation=self.dilation)

            with torch.no_grad():
                # split weight
                if weight_is_split:
                    split_weight = weight.chunk(self.split_groups, dim=0)
                else:
                    self.sweight = conv_sweight_cuda.forward(self.sweight, weight, self.group_in_offset, self.split_groups)
                    split_weight = self.sweight.chunk(self.split_groups, dim=0)

                input_channel = 0
                out_tmp = []
                for i in range(0, self.split_groups):
                    split_input = input.narrow(1, input_channel, self.group_in_channels)
                    out_tmp.append(F.conv2d(split_input, split_weight[i], bias=None,\
                                stride=self.stride, padding=padding, dilation=self.dilation))
                    # prepare for the next stage
                    if i < self.split_groups - 1:
                        input_channel += self.group_move_in_channels[i]

                # concat output or merge group
                if merge_group:
                    tmp = out_tmp
                    out_tmp = tmp[0]
                    for i in range(1, self.split_groups):
                        out_tmp += tmp[i]
                elif cat_output:
                    out_tmp = torch.cat(out_tmp, 1)

                if self.training and (not infer_only):
                    output.copy_(out_tmp)
                else:
                    output = out_tmp

            # add bias
            if (ignore_bias is False) and (self.bias is not None):
                output += self.bias
        else:
            # weight_split = torch.chunk(weight.view(weight.shape[0], -1), self.split_groups, dim=1)
            # sweight_split = torch.chunk(self.sweight.view(weight.shape[0], -1).to(weight.device), self.split_groups, dim=1)
            if weight_is_split:
                split_weight = weight.chunk(self.split_groups, dim=0)
            else:
                self.sweight = conv_sweight_cuda.forward(self.sweight, weight, self.group_in_offset, self.split_groups)
                split_weight = self.sweight.chunk(self.split_groups, dim=0)
            # with torch.no_grad():
                # masking = torch.zeros(self.split_groups, self.group_in_channels, self.kernel_size[0], self.kernel_size[1], device=weight.device)
                # one_w = torch.ones(1, self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], device=weight.device)
                # masking = conv_sweight_cuda.forward(masking, one_w, self.group_in_offset, self.split_groups)
                # split_masking = masking.chunk(self.split_groups, dim=0)

            input_channel = 0
            out_tmp = []
            for i in range(0, self.split_groups):
                split_input = input.narrow(1, input_channel, self.group_in_channels)
                mask = torch.zeros_like(split_weight[0], dtype=torch.bool, device=weight.device)
                mask.view(split_weight[0].shape[0], -1)[:,self.group_in_offset[i]:self.group_in_offset[i] + self.group_fan_in] = 1 
                # split_weight = weight.narrow(1, input_channel, self.group_in_channels) * split_masking[i]
                out_tmp.append(F.conv2d(split_input, torch.mul(mask, split_weight[i]), bias=None,\
                            stride=self.stride, padding=padding, dilation=self.dilation))
                # prepare for the next stage
                if i < self.split_groups - 1:
                    input_channel += self.group_move_in_channels[i]

            output = torch.cat(out_tmp, 1)

        return output
 

    def forward(self, input):
        # self._mask_weight()
        return self._split_forward(input, self.weight, padded=False, cat_output=True)

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
        if self.split_groups != 1:
            s += ', split_groups={split_groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', wbits={wbits}'
        return s.format(**self.__dict__)


class SplitLinear(nn.Linear):
    """Split Linear module. It provide input split on nn.Linear

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        groups (int, optional): Number of split groups. Default: 1

    Shape:
        - Input: (N, in_features)
        - Output: (N, out_feature)
    """
    def __init__(self, in_features, out_features, bias=False, groups=1):
        super(SplitLinear, self).__init__(in_features, out_features, bias=bias)
        # define in/out features for group
        self.split_groups = groups
        if in_features % self.split_groups != 0:
            raise ValueError('in_features must be divisible by groups')
        self.group_in_features = int(in_features / self.split_groups)

    def reset_groups(self, groups):
        if self.split_groups == groups:
            return
        assert self.groups != 1, 'reset group is not supported when groups is not 1'
        # reset in/out_feature information
        self.group_in_features = int(self.in_features / groups)
        # reset groups
        self.split_groups = groups

    def _split_forward(self, input, weight, ignore_bias=False, cat_output=True, infer_only=False, merge_group=False, binary=False):
        
        if not binary:
            output = None

            # operation for backward
            if self.training and (not infer_only):
                output = F.linear(input, weight, bias=None)

            # split fc
            with torch.no_grad():
                # split input & weight
                split_input = input.chunk(self.split_groups, dim=-1)
                split_weight = weight.chunk(self.split_groups, dim=1)

                out_tmp = []
                for i in range(0, self.split_groups):
                    out_tmp.append(F.linear(split_input[i], split_weight[i], bias=None))

                # concat output or merge group
                if merge_group:
                    tmp = out_tmp
                    out_tmp = tmp[0]
                    for i in range(1, self.split_groups):
                        out_tmp += tmp[i]
                elif cat_output:
                    out_tmp = torch.cat(out_tmp, 1)

                if self.training and (not infer_only):
                    output.copy_(out_tmp)
                else:
                    output = out_tmp

            # add bias
            if (ignore_bias is False) and (self.bias is not None):
                output += self.bias
        else:
            split_input = input.chunk(self.split_groups, dim=-1)
            split_weight = weight.chunk(self.split_groups, dim=1)

            out_tmp = []
            for i in range(0, self.split_groups):
                out_tmp.append(F.linear(split_input[i], split_weight[i], bias=None))

            out_tmp = torch.cat(out_tmp, 1)
            output = out_tmp

        return output

    def forward(self, input):
        return self._split_forward(input, self.weight, cat_output=True)

    def extra_repr(self):
        """"Provides layer information, including groups. when print(model) is called."""
        return 'in_features={}, out_features={}, bias={}, groups={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.groups
        )