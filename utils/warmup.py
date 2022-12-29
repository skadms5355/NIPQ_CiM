from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class warmup(_LRScheduler):
    """
        Reference: https://github.com/ildoonet/pytorch-gradual-warmup-lr


    	Gradually warm-up(increasing) learning rate in optimizer.
    	Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(warmup, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
                return temp
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            # return [base_lr * (float(self.last_epoch + 1) / self.total_epoch) for base_lr in self.base_lrs]
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.multiplier* (float(self.last_epoch + 1) / self.total_epoch) for base_lr in self.base_lrs]

    def get_last_lr(self):
        if self.last_epoch >= self.total_epoch:
            return self.after_scheduler.get_last_lr()
        else:
            return self._last_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if (key != 'optimizer' and key != 'after_scheduler')}

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(warmup, self).step(epoch)
