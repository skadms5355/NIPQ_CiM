import torch
import torch.nn as nn

class NonlinearQuant(torch.autograd.Function):
    """
        This module performs nonlinear quantization.
        Each interval sets to random noise value from HSPICE simulation
    """
    @staticmethod
    def forward(ctx, input, step, weight, pbits, group, half_num_levels, std):
        output = []
        intervals = int(2**(pbits)-1)
        ref = torch.tensor([(1-half_num_levels) + 0.5 + i for i in range(intervals)], device=input[0].device)
        scaled_step = step * weight
        for g in range(group):
            output.append(torch.zeros_like(input[g]))
            noise_Q = (std ** 2) * torch.randn_like(ref, device=input[g].device)
            noise_Q = torch.clamp(noise_Q, -0.5, 0.5) 

            input[g] = input[g] / scaled_step
            noise_ref = noise_Q + ref
            # noise_ref = ref
            # print('g {}: noise_ref {}'.format(g, noise_ref))
            for i in range(intervals+1):
                if i == 0 :
                    mask = (input[g] < noise_ref[i])
                    output[g][mask] = (1-half_num_levels)
                elif i == intervals:
                    mask = (input[g] >= noise_ref[i-1])
                    output[g][mask] = half_num_levels
                else:
                    mask = (input[g] >= noise_ref[i-1]) & (input[g] < noise_ref[i])
                    output[g][mask] = (ref[i-1] + ref[i]) / 2

            output[g] *= scaled_step

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Yet, Value gradients out of range don't consider 
        return grad_output, None, None, None, None
