'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from distutils.dir_util import copy_tree
import errno
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
import math
import pdb
import torch.nn as nn
import torch.nn.init as init

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_now', 'AverageMeter', 'ForkedPdb']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_now(prefix, timestamp=None):
    '''make dir by current time if not exist, append one to time if it already exists'''
    if timestamp is None:
        timestamp = datetime.now()
    dir_path = os.path.join(prefix, timestamp.strftime("%Y-%b-%d-%H-%M-%S"))
    try:
        os.makedirs(dir_path)
        return dir_path
    except FileExistsError:
        timestamp = timestamp + timedelta(seconds=1)
        return mkdir_now(prefix, timestamp)
        pass

def copy_folder(src, dst):
    '''copy saved at local checkpoint folde to NAS server '''
    copy_foldname=os.path.basename(os.path.normpath(src))
    dst_fold=os.path.join(dst, copy_foldname)
    try:
        shutil.copytree(src, dst_fold)
    except FileExistsError:
        copy_tree(src, dst_fold)


def lndir_p(src, dst):
    '''link checkpoint if link is not None'''
    if dst is not None:
        try:
            assert src is not None, 'src is not set.'
            rel_path_src = os.path.relpath(src, os.path.dirname(dst))
            os.symlink(src=rel_path_src , dst=dst)
        except OSError:  # Python >2.5
            raise
    else:
        pass


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

