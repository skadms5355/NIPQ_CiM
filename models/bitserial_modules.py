import torch

def bitserial_func(input, bits):
    output_dtype = input.round_().dtype
    output_uint8= input.to(torch.uint8)
    bitserial_step = 1 / (2.**(bits - 1.))

    output = output_uint8 & 1
    for i in range(1, bits):
        out_tmp = output_uint8 & (1 << i)
        output = torch.cat((output, out_tmp), 1)
    output = output.to(output_dtype)
    # output.mul_(bitserial_step) ## for preventing overflow

    return output.round_()

# transform float value to bitserial value (int)
class Bitserial():
    """
        Change floating values to integer values and represent bit serial 2's complement
        weight: multiply scale vector
        activation: multiply scale vector and round fuction (no linearity)
    """
    @classmethod
    def get_value(cls, bits, scale, bitserial=False):
        cls.bits = bits
        cls.scale = scale
        cls.bitserial = bitserial

    @classmethod
    def bitserial_act(cls, input, debug=False):
        """
            input: [batch, channel, H, W]
            output: [batch, abits * channel, H, W]
        """
        output = input / cls.scale  # remove remainder value ex).9999 

        output_bit = bitserial_func(output, cls.bits)

        if debug: 
            print('ascale: ', cls.scale)
            print('input: ', sorted(set(input.cpu().detach().numpy().ravel())))
            print('input_step_round', set(output.cpu().detach().numpy().ravel()))
            print('input_step_round', set(output_bit.cpu().detach().numpy().ravel()))

        return output_bit, cls.scale , cls.bits
    
    @classmethod
    def abit_serial(cls):
        return cls.bitserial