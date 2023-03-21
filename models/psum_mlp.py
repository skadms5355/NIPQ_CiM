import torch 
import torch.nn as nn
from .quantized_lsq_modules import *
from .quantized_basic_modules import *
from .psum_modules import *

__all__ = ['psum_mlp']

class Psum_mlp(nn.Module):
    def __init__(self, **kwargs):
        super(Psum_mlp, self).__init__()

        self.fc1 = QLinear(784, 512, wbits=32)
        self.bn1 = nn.BatchNorm1d(512)
        self.qact = add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'])

        self.fc2 = PsumQLinear(512, 512, wbits=kwargs['wbits'], arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], psum_mode=kwargs['psum_mode'],
                            mapping_mode=kwargs['mapping_mode'], cbits=kwargs['cbits'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'])
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PsumQLinear(512, 512, wbits=kwargs['wbits'], arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], psum_mode=kwargs['psum_mode'],
                            mapping_mode=kwargs['mapping_mode'], cbits=kwargs['cbits'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'])
        self.bn3 = nn.BatchNorm1d(512)
        self.act = add_act(abits=32)

        self.fc4 = QLinear(512, kwargs['num_classes'], wbits=32)
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


def psum_mlp(**kwargs):
    assert kwargs['dataset'] == 'mnist', f"binarynet vgg model only supports CIFAR-10 dataset."

    return Psum_mlp(**kwargs)
