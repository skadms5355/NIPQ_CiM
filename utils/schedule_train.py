from functools import partial
import torch
import torch.optim as optim

from .warmup import warmup

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
    elif args.lr_method == 'lr_cosineanneal':
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.T0, args.T_mult)
    else :
        assert False, 'learning rate decay method should be one of step, linear, exp, cosineanneal'

    # Warmup scheduler
    if args.warmup > 0 and args.rank == 0:
        print('==> Using warmup scheduler..')
    scheduler = warmup(optimizer, multiplier=args.multiplier, total_epoch=args.warmup, after_scheduler=after_scheduler)

    return scheduler, after_scheduler
