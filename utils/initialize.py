from __future__ import division

import math
import warnings
import torch

__all__ = ['init_weight']

# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and re-usable.
def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)


def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)


def uniform_(tensor, a=0., b=1.):
    # type: (Tensor, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor, mean=0., std=1.):
    # type: (Tensor, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    return _no_grad_normal_(tensor, mean, std)



def _find_fan(layer):
    dimensions = layer.weight.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    fan_in = layer.weight.size(1)
    fan_out = layer.weight.size(0)

    # receptive field size for in and out
    rf_in = 1
    rf_out = 1
    if dimensions > 2:
        if isinstance(layer.stride, tuple):
            rf_out = math.ceil(layer.weight.size(2)/layer.stride[0])*math.ceil(layer.weight.size(3)/layer.stride[1])
        elif isinstance(layer.stride, int):
            rf_out = math.ceil(layer.weight.size(2)/layer.stride)*math.ceil(layer.weight.size(3)/layer.stride)
        else:
            assert False, "Something wrong about layer's stride."

        rf_in = layer.weight.size(2)*layer.weight.size(3)

    fan_in = rf_in * fan_in
    fan_out = rf_out * fan_out

    return fan_in, fan_out



def init_weight(layer, method='xavier', dist='normal', mode='fan_both'):
    r"""
    Initialize the weight of layer following different methods.

    It is good to note that the implementation differs from the Pytorch implementation,
    https://pytorch.org/docs/stable/nn.init.html. We provide initialization in the most
    general case. For xavier, we assume gain=1.0, and for kaiming, we assume nonlinearity='relu'

    Args:
        method: method to follow for initialization. Xavier, kaiming, etc.
        dist: distribution to use. uniform, normal, etc.
        mode: which pass to optimize. forward, backward or both.
    """
    if hasattr(layer, 'weight'):
        fan_in, fan_out = _find_fan(layer)
    else:
        raise ValueError("Layer {} has no weight to initialize.".format(type(layer).__name__))

    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_both']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    n = fan_in if mode == 'fan_in' else fan_out if mode == 'fan_out' else (float(fan_in + fan_out) / 2.0)

    method = method.lower()
    valid_methods = ['xavier', 'kaiming', 'uniform']
    if method not in valid_methods:
        raise ValueError("Method {} not supported, please use one of {}".format(method, valid_methods))

    if method == 'uniform':
        return uniform_(layer.weight, a=-1., b=1.)

    dist = dist.lower()
    valid_dists = ['normal', 'uniform']
    if dist not in valid_dists:
        raise ValueError("Distribution {} not supported, please use one of {}".format(dist, valid_dists))

    if dist == 'normal':
        if method == 'xavier':
            std = math.sqrt(1.0 / n)
        elif method == 'kaiming':
            std = math.sqrt(2.0 / n)

        return _no_grad_normal_(layer.weight, 0., std)

    elif dist == 'uniform':
        if method == 'xavier':
            bound = math.sqrt(3.0 / n)
        elif method == 'kaiming':
            bound = math.sqrt(6.0 / n)

        return _no_grad_uniform_(layer.weight, -bound, bound)



