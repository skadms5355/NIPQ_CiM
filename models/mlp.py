import torch 
import torch.nn as nn
from .quantized_lsq_modules import *
from .quantized_basic_modules import *
from .binarized_modules import *

__all__ = ['lsq_mlp', 'binarynet_512mlp', 'quant_mlp']

class LSQ_mlp(nn.Module):
    def __init__(self, **kwargs):
        super(LSQ_mlp, self).__init__()

        self.fc1 = QLinear(784, 512, wbits=32)
        self.bn1 = nn.BatchNorm1d(512)
        self.qact = add_act(abits=kwargs['abits'])

        self.fc2 = QLinear(512, 512, kwargs['wbits'])
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = QLinear(512, 512, kwargs['wbits'])
        self.bn3 = nn.BatchNorm1d(512)
        self.act = add_act(abits=32)

        self.fc4 = QLinear(512, kwargs['num_classes'], wbits=32, bias=True)
        self.bn4 = nn.BatchNorm1d(kwargs['num_classes'])

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.qact(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.qact(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x

class Binarynet_MLP(nn.Module):
    def __init__(self, **kwargs):
        super(Binarynet_MLP, self).__init__()

        self.fc1 = BinLinear(784, 512, 32, kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn1 = nn.BatchNorm1d(512)
        self.act = nonlinear(kwargs['abits'], kwargs['binary_mode'], kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'])

        self.fc2 = BinLinear(512, 512, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = BinLinear(512, 512, kwargs['wbits'], kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn3 = nn.BatchNorm1d(512)
        self.n_act = nonlinear(32)

        self.fc4 = BinLinear(512, kwargs['num_classes'], 32, kwargs['weight_clip'], kwargs['weight_scale'])
        self.bn4 = nn.BatchNorm1d(kwargs['num_classes'])

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.n_act(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x

class Quant_mlp(nn.Module):
    def __init__(self, **kwargs):
        super(Quant_mlp, self).__init__()

        self.fc1 = QuantLinear(784, 512, wbits=32, weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'])
        self.bn1 = nn.BatchNorm1d(512)
        self.qact = nonlinear(abits=kwargs['abits'], ste=kwargs['ste'])

        self.fc2 = QuantLinear(512, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'])
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = QuantLinear(512, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'])
        self.bn3 = nn.BatchNorm1d(512)
        self.act = nonlinear(abits=32, ste=kwargs['ste'])

        self.fc4 = QuantLinear(512, kwargs['num_classes'], wbits=32 if kwargs['wbits']==32 else 32, weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], bias=True)
        self.bn4 = nn.BatchNorm1d(kwargs['num_classes'])

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.qact(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.qact(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x

def binarynet_512mlp(**kwargs):
    assert kwargs['dataset'] == 'mnist', f"binarynet mlp model only supports MNIST dataset."
    return Binarynet_MLP(**kwargs)

def lsq_mlp(**kwargs):
    assert kwargs['dataset'] == 'mnist', f"binarynet vgg model only supports CIFAR-10 dataset."

    return LSQ_mlp(**kwargs)

def quant_mlp(**kwargs):
    assert kwargs['dataset'] == 'mnist', f"binarynet vgg model only supports CIFAR-10 dataset."

    return Quant_mlp(**kwargs)
