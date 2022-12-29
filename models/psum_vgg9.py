import torch 
import torch.nn as nn
from .quantized_lsq_modules import *
from .quantized_modules import *
from .psum_modules import *

__all__ = ['psum_vgg9']

class Psum_vgg9(nn.Module):
    def __init__(self, **kwargs):
        super(Psum_vgg9, self).__init__()

        self.features = nn.Sequential(
            QConv(3, 128, wbits=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial']),

            PsumQConv(128, 128, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial']),

            PsumQConv(128, 256, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm2d(256),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial']),

            PsumQConv(256, 256, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial']),

            PsumQConv(256, 512, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm2d(512),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial']),

            PsumQConv(512, 512, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial']),    
        )
        self.classifier = nn.Sequential(
            PsumQLinear(512 * 4 * 4, 1024, wbits=kwargs['wbits'], arraySize=kwargs['arraySize'], 
                        wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                        psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm1d(1024),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial']), 

            PsumQLinear(1024, 1024, wbits=kwargs['wbits'], arraySize=kwargs['arraySize'], 
                        wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                        psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm1d(1024),
            add_act(abits=32), 

            nn.Dropout(kwargs['drop']),
            QLinear(1024, kwargs['num_classes'], wbits=32, bias=True),
            nn.BatchNorm1d(kwargs['num_classes']),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def psum_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"binarynet vgg model only supports CIFAR-10 dataset."

    return Psum_vgg9(**kwargs)
