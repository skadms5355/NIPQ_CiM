import torch

def bitserial_func(input, bits, training=False):
    if training:
        return Bitseiral_train.apply(input, bits)
    else:
        output_dtype = input.round_().dtype
        output_uint8= input.to(torch.uint8)

        output = output_uint8 & 1
        for i in range(1, bits):
            out_tmp = output_uint8 & (1 << i)
            output = torch.cat((output, out_tmp), 1)
        output = output.to(output_dtype)

        return output

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
    def bitserial_act(cls, input, training=False, debug=False):
        """
            input: [batch, channel, H, W]
            output: [batch, abits * channel, H, W]
        """
        output = input / cls.scale  # remove remainder value ex).9999 
        output = bitserial_func(output, cls.bits, training=training)

        if debug: 
            print('ascale: ', cls.scale)
            print('input: ', sorted(set(input.cpu().detach().numpy().ravel())))
            print('input_step_round', set(output.cpu().detach().numpy().ravel()))

        return output, cls.scale , cls.bits
    
    @classmethod
    def get_abit_scale(cls):
        return cls.bits, cls.scale
    
    @classmethod
    def abit_serial(cls):
        return cls.bitserial
    
class Bitseiral_train(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bits):
        ctx.other = bits
        output_dtype = input.round_().dtype
        output_uint8= input.to(torch.uint8)

        output = output_uint8 & 1
        for i in range(1, bits):
            out_tmp = output_uint8 & (1 << i)
            output = torch.cat((output, out_tmp), 1)
        output = output.bool().to(output_dtype)

        return output

    def backward(ctx, grad_output):
        bits = ctx.other
        split_grad = torch.chunk(grad_output, bits, dim=1)
        for b in range(bits):
            if b == 0:
                grad_input = split_grad[0]
            else:
                grad_input += (split_grad[b]/2**b)
        grad_input = grad_input/bits 
        
        return grad_input, None
    