import torch
import torch.nn.functional as F

__all__ = ['pad']

def alter_pad_1d(input, padding_shape):
    padding = padding_shape[0]

    lr_padding = torch.ones([input.size(0), input.size(1), input.size(2), padding], dtype=torch.float32).cuda()

    lr_padding[:, :, 1::2, :] *= -1
    lr_padding[:, :, :, 1::2] *= -1

    input = torch.cat((lr_padding, input, -lr_padding), dim=3)
    return input

def alter_pad_2d(input, padding_shape):
    padding = padding_shape[0]
    input_size = input.size()

    lr_padding = torch.ones([input_size[0], input_size[1], input_size[2], padding], dtype=torch.float32).cuda()
    l_flip = 1 if (padding % 2 == 1) else -1
    r_flip = 1 if (input_size[3] % 2 == 1) else -1

    lr_padding[:, :, 1::2, :] *= -1
    lr_padding[:, :, :, 1::2] *= -1

    input = torch.cat((l_flip * lr_padding, input, r_flip * lr_padding), dim=3)

    ud_padding = torch.ones([input_size[0], input_size[1], padding, input_size[3] + 2 * padding], dtype=torch.float32).cuda()
    d_flip = 1 if ((input_size[2] + padding) % 2 == 0) else -1

    ud_padding[:, :, 0::2, :] *= -1
    ud_padding[:, :, :, 1::2] *= -1

    input = torch.cat((ud_padding, input, d_flip * ud_padding), dim=2)

    return input

def alter_pad(input, padding_shape):
    if(len(padding_shape) == 2):
        output = alter_pad_1d(input, padding_shape)
    elif(len(padding_shape) == 4):
        output = alter_pad_2d(input, padding_shape)
    else:
        assert False, 'Dimension of alternative padding should be 1 or 2'

    return output

def pad(input, padding_shape, padding_mode, padding_value):
    if padding_mode != 'alter':
        output = F.pad(input, padding_shape, 'constant', padding_value)
    else:
        output = alter_pad(input, padding_shape)
    return output
