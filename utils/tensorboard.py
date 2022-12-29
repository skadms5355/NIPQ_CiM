"""This module provides pytorch utilities for Tensorboard.

The module defines a Tensorboard_writer class, which can be used for generating graph, logging scalars
such as loss and accuracy, printing histograms of tensors and gradients in each layer.

"""

import warnings
import torch
warnings.simplefilter("ignore", category=FutureWarning)
try:
    from torch.utils.tensorboard import SummaryWriter
    # from tensorboardX import SummaryWriter
except ImportError:
    pass

__all__ = ['Tensorboard_writer']

class Tensorboard_writer():
    """Creates a Tensorboard Summary Writer. Its functions work if args.tensorboard > 0.

    Attributes:
        use_tensorboard (int): Tensorboard log level.
            If 0, the object is not created.
            If 1, only graph, loss and accuracies are logged.
            If 2, parameters and thier gradients are logged.
            If 3, tensor inputs and outputs, and their gradients are logged.
        writer: A SummaryWriter with its save_path given in its creation.
        foward_logger_handle_list: A list of forward logger handle.
        backward_logger_handle_list: A list of backward logger handle.

    """

    def __init__(self, save_path, use_tensorboard, rank):
        self.use_tensorboard = use_tensorboard
        self.rank = rank
        if use_tensorboard > 0 and rank == 0:
            self.writer = SummaryWriter(save_path)
            self.forward_logger_handle_list = []
            self.backward_logger_handle_list = []


    def _forward_logger(self, writer, module_name, idx, epoch, tensorboard_log_batch_size=50):
        """A hook for retrieving layer inputs and outputs. """

        def hook(self, input, output):
            writer.add_histogram(f"forward/{idx}/{module_name}/input",
                                 input[0][:tensorboard_log_batch_size], epoch) #input: tuple
            writer.add_histogram(f"forward/{idx}/{module_name}/output",
                                 output[:tensorboard_log_batch_size], epoch) #output: tensor
            writer.flush() #need for write events to disk
        return hook


    def _backward_logger(self, writer, module_name, idx, epoch, tensorboard_log_batch_size=50):
        """A hook for retrieving gradients of layer inputs and outputs. """

        def hook(self, grad_input, grad_output):
            if grad_input[0] is not None:
                writer.add_histogram(f"backward/{idx}/{module_name}/grad_input",
                                     grad_input[0][:tensorboard_log_batch_size], epoch) #grad_input: tuple
            writer.add_histogram(f"backward/{idx}/{module_name}/grad_output",
                                 grad_output[0][:tensorboard_log_batch_size], epoch) #grad_output: tuple
            writer.flush() #need for write events to disk
        return hook


    def add_hooks(self, model, epoch, tensorboard_log_batch_size=50, inspect_modules=None):
        """Add hookers for tensorboard logging of layer input/output of forward/backward. """

        if self.use_tensorboard >= 3 and self.rank == 0:
            assert inspect_modules is None, "inspect_modules will be ignored!"

            # will observe output distribution and mean of these layers
            inspect_modules = ['Conv2d', 'BinConv', 'BinLinear', 'Linear', 'BatchNorm2d', 'BatchNorm1d', 'MaxPool2d', 'AvgPool2d']
            for idx, module in enumerate(model.modules()):
                module_name = module._get_name()
                if module_name in inspect_modules:
                    # add forward hooker for tensorboard logging
                    forward_handle = module.register_forward_hook(
                        self._forward_logger(self.writer, module_name, idx, epoch, tensorboard_log_batch_size))
                    self.forward_logger_handle_list.append(forward_handle)
                    # add backward hooker for tensorboard logging
                    backward_handle = module.register_backward_hook(
                        self._backward_logger(self.writer, module_name, idx, epoch, tensorboard_log_batch_size))
                    self.backward_logger_handle_list.append(backward_handle)


    def remove_hooks(self):
        """Remove hooks for tensorboard logging of layer input/output of forward/backward. """

        if self.use_tensorboard >= 3 and self.rank == 0:
            while self.forward_logger_handle_list:
                handle = self.forward_logger_handle_list.pop()
                handle.remove()
            while self.backward_logger_handle_list:
                handle = self.backward_logger_handle_list.pop()
                handle.remove()


    def add_graph(self, model, inputs):
        """Add a graph to the tensorboard. """

        if self.use_tensorboard >= 1 and self.rank == 0:
            self.writer.add_graph(model, inputs)
            self.writer.flush() #need for write events to disk


    def log_param(self, model, epoch, inspect_modules=None, inspect_keys=None):
        """Add histograms of parameters (weights, BN params, etc) to the tensorboard. """

        if self.use_tensorboard >= 2 and self.rank == 0:
            if inspect_modules is None:
                # inspect_modules = ['Conv2d', 'BinConv', 'BinLinear', 'Linear', 'BatchNorm2d', 'BatchNorm1d']
                inspect_modules = ['BatchNorm2d', 'BatchNorm1d']
            if inspect_keys is None:
                inspect_keys = ['weight', 'bias']
            for idx, module in enumerate(model.modules()):
                module_name = module._get_name()
                if module_name in inspect_modules:
                    for key in module.state_dict().keys():
                        # log parameters & buffers
                        if key in inspect_keys:
                            self.writer.add_histogram(f"variable/{idx}/{module_name}_{key}",
                                                      module.state_dict()[key].clone().cpu().data.numpy(), epoch)
            self.writer.flush() #need for write events to disk


    def log_grads(self, model, epoch, inspect_modules=None):
        """Add histograms of gradient of parameters (weights, BN params, etc) to the tensorboard. """

        if self.use_tensorboard >= 2 and self.rank == 0:
            if inspect_modules is None:
                inspect_modules = ['Conv2d', 'BinConv', 'BinLinear', 'Linear', 'BatchNorm2d', 'BatchNorm1d']
            for idx, module in enumerate(model.modules()):
                module_name = module._get_name()
                if module_name in inspect_modules:
                    for key, param in module._parameters.items():
                        # log gradient of parameters
                        if param is not None:
                            self.writer.add_histogram(f"grad/{idx}/{module_name}_{key}",
                                                      param.grad.clone().cpu().data.numpy(), epoch)
            self.writer.flush() #need for write events to disk


    def log_scalar(self, tag, scalar_value, epoch):
        """Add scalar values to the tensorboard. """

        if self.use_tensorboard >= 1 and self.rank == 0:
            self.writer.add_scalar(tag, scalar_value, epoch)
            self.writer.flush() #need for write events to disk


    def log_scalars(self, main_tag, tag_scalar_dict, epoch):
        """Add multiple scalar values given as a (tag, scalar) dictionary to the tensorboard. """

        # The behavior in naming tag is different from add_scalars!
        if self.use_tensorboard >= 1 and self.rank == 0:
            for tag, scalar in tag_scalar_dict.items():
                self.writer.add_scalar(f"{main_tag}/{tag}", scalar, epoch)
            self.writer.flush() #need for write events to disk


    def close(self):
        if self.use_tensorboard > 0 and self.rank == 0:
            self.writer.close()
