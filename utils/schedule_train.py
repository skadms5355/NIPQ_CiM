from functools import partial
import torch
import torch.optim as optim
import math
from .warmup import warmup
import warnings

__all__ = ['set_optimizer', 'set_scheduler']


# generate generator for dictionary of parameter and corresponding learning rates
def param_and_lr_with_glr(model, lr):
    for idx, module in enumerate(model.modules()):
        for key, param in module._parameters.items():
            if param is not None:
                param_lr = lr
                if module._get_name() in ['BinConv', 'BinLinear']:
                    param_lr = lr * module.glr
                yield {'params': param, 'lr': param_lr}

def set_optimizer(model, other_parameters, weight_parameters, args):
    # make list of dictionary of parameter and corresponding learning rates
    if args.glr:
        param_and_lr_list = list(param_and_lr_with_glr(model, args.lr))
        print(f"It is not sure if weight decay is applied to weights only when glr is used. \
                If glr is used, the model is probably binary model and weight decay is not used.")
    else:
        param_and_lr_list = [{'params': other_parameters, 'weight_decay': 0, 'lr': args.lr}, {'params': weight_parameters, 'weight_decay': args.weight_decay, 'lr': args.lr}]

    # set optimzier
    # (TODO) VINN: we need momentum correction to use weight_decay for sgd properly.
    # Refer to Section 3 of "Accurate, Large Minibatch SGD - Learning Imagenet in 1 Hour"
    
    # We don't need to pass the args.weight_decay to optimizer since we specified per-parameter weight_decay value already.
    # However, we still pass it for the case when glr is used. Again, it is not clear WD correctly works with glr.
    # The argument passed through optimizer does not override the per-parameter weight_decay values (https://pytorch.org/docs/stable/optim.html#per-parameter-options).
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_and_lr_list,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            param_and_lr_list,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(
            param_and_lr_list,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_and_lr_list,
            weight_decay=args.weight_decay
        )
    return optimizer

def linear_decay(epochs, step):
    return (1.0 - step/epochs)

def set_scheduler(optimizer, args):
    # define learning rate scheduling methods.
    schedule = args.schedule
    if args.lr_method == 'lr_step':
        if args.warmup > 0:
            schedule = [i - args.warmup for i in schedule]
        after_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, schedule, args.gamma)
    elif args.lr_method == 'lr_linear':
        after_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(linear_decay, args.epochs - args.warmup))
    elif args.lr_method == 'lr_exp':
        after_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.lr_method == 'lr_cosineannealwr':
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.T0, args.T_mult, eta_min=args.eta_min)
    elif args.lr_method == 'lr_cosineanneal':
        after_scheduler = CosineAnnealing(optimizer, args.T0, args.T_mult, eta_min=args.eta_min)
    else :
        assert False, 'learning rate decay method should be one of step, linear, exp, cosineanneal'

    # Warmup scheduler
    if args.warmup > 0 and args.rank == 0:
        print('==> Using warmup scheduler..')
    scheduler = warmup(optimizer, multiplier=args.multiplier, warmup_start_multiplier=args.warmup_start_multiplier, total_epoch=args.warmup, after_scheduler=after_scheduler)

    return scheduler, after_scheduler

class CosineAnnealing(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0.0, last_epoch: int = -1, verbose: bool = False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min  # Final LR multiplier of cosine annealing
        self.T_cur = last_epoch
        super(CosineAnnealing, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
            
        lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
