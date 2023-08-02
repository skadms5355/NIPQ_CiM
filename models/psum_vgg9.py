import torch 
import torch.nn as nn
from .nipq_quantization_module import QuantOps as Q
from .nipq_hwnoise_psum_module import PsumQuantOps as PQ
from .quantized_lsq_modules import *
from .quantized_basic_modules import *
from .psum_modules import *
from .binarized_psum_modules import *
from .binarized_modules import nonlinear

__all__ = ['psum_lsq_vgg9', 'psum_quant_vgg9', 'psum_nipq_vgg9', 'psum_binary_vgg9']

class PNIPQ_vgg9(nn.Module):
    def __init__(self, **kwargs):
        super(PNIPQ_vgg9, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            PQ.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            PQ.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            PQ.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            PQ.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            PQ.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            
        )
        self.classifier = nn.Sequential(
            PQ.Linear(512 * 4 * 4, 1024, act_func=Q.ReLU()),  
            nn.BatchNorm1d(1024), 
            nn.ReLU(inplace=True),

            PQ.Linear(1024, 1024, act_func=Q.ReLU()),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Dropout(kwargs['drop']),
            # Q.Linear(1024, kwargs['num_classes'], bias=True, act_func=Q.ReLU),
            nn.Linear(1024, kwargs['num_classes'], bias=True),
            nn.BatchNorm1d(kwargs['num_classes']),   
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class PLSQ_vgg9(nn.Module):
    def __init__(self, **kwargs):
        super(PLSQ_vgg9, self).__init__()

        self.features = nn.Sequential(
            # nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            QuantConv(3, 128, wbits=32 if kwargs['FL_quant']==32 else 32, weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], wsymmetric=kwargs['wsymmetric'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
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
            add_act(abits=kwargs['abits'] if kwargs['FL_quant'] else 32), 

            nn.Dropout(kwargs['drop']),
            QuantLinear(1024, kwargs['num_classes'], wbits=32 if kwargs['FL_quant']==32 else 32, weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], wsymmetric=kwargs['wsymmetric'], bias=True),
            nn.BatchNorm1d(kwargs['num_classes']),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class PQuant_vgg9(nn.Module):
    def __init__(self, **kwargs):
        super(PQuant_vgg9, self).__init__()

        self.features = nn.Sequential(
            # nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            QConv(3, 128, wbits=kwargs['wbits'] if kwargs['FL_quant'] else 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(128, 128, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(128, 256, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm2d(256),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(256, 256, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(256, 512, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm2d(512),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(512, 512, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False,
                    arraySize=kwargs['arraySize'], wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                    psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                    is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),    
        )
        self.classifier = nn.Sequential(
            PsumQLinear(512 * 4 * 4, 1024, wbits=kwargs['wbits'], arraySize=kwargs['arraySize'], 
                        wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                        psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm1d(1024),
            add_act(abits=kwargs['abits'], bitserial=kwargs['abit_serial'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']), 

            PsumQLinear(1024, 1024, wbits=kwargs['wbits'], arraySize=kwargs['arraySize'], 
                        wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                        psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm1d(1024),
            add_act(abits=kwargs['abits'] if kwargs['FL_quant'] else 32, mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']), 

            nn.Dropout(kwargs['drop']),
            PsumQLinear(1024, kwargs['num_classes'], wbits=kwargs['wbits'] if kwargs['FL_quant'] else 32, arraySize=kwargs['arraySize'], 
                        wbit_serial=kwargs['wbit_serial'], mapping_mode=kwargs['mapping_mode'], 
                        psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'], bias=True),
            # nn.Linear(1024, kwargs['num_classes'], bias=True),
            nn.BatchNorm1d(kwargs['num_classes']),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PBin_vgg(nn.Module):
    def __init__(self, **kwargs):
        super(PBin_vgg, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # BinConv(3, 128, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode']),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(128, 128, wbits=kwargs['wbits'], wbit_serial=kwargs['wbit_serial'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(128, 256, wbits=kwargs['wbits'], wbit_serial=kwargs['wbit_serial'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(256, 256, wbits=kwargs['wbits'], wbit_serial=kwargs['wbit_serial'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(256, 512, wbits=kwargs['wbits'], wbit_serial=kwargs['wbit_serial'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm2d(512),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQConv(512, 512, wbits=kwargs['wbits'], wbit_serial=kwargs['wbit_serial'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                        is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

        )
        self.classifier = nn.Sequential(
            PsumQLinear(512 * 4 * 4, 1024, wbits=kwargs['wbits'], wbit_serial=kwargs['wbit_serial'],
                            arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                            is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm1d(1024),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PsumQLinear(1024, 1024, wbits=kwargs['wbits'], wbit_serial=kwargs['wbit_serial'],
                            arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], psum_mode=kwargs['psum_mode'], cbits=kwargs['cbits'], 
                            is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type']),
            nn.BatchNorm1d(1024),
            # nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            nonlinear(abits=32, mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            nn.Linear(1024, kwargs['num_classes'], bias=True),
            # BinLinear(1024, kwargs['num_classes'], wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(kwargs['num_classes'])
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def psum_lsq_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"binarynet vgg model only supports CIFAR-10 dataset."

    return PLSQ_vgg9(**kwargs)

def psum_quant_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"binarynet vgg model only supports CIFAR-10 dataset."

    return PQuant_vgg9(**kwargs)

def psum_nipq_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"binarynet vgg model only supports CIFAR-10 dataset."

    return PNIPQ_vgg9(**kwargs)

def psum_binary_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"binarynet vgg model only supports CIFAR-10 dataset."

    return PBin_vgg(**kwargs)
