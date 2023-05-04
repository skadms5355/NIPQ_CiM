import torch 
import torch.nn as nn
from typing import Any
from .nipq_quantization_module import QuantOps as Q
from .quantized_lsq_modules import *
from .quantized_basic_modules import *

__all__ = ['nipq_vgg9', 'lsq_vgg9', 'quant_vgg9']

class NIPQ_vgg9(nn.Module):
    def __init__(self, **kwargs):
        super(NIPQ_vgg9, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            Q.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            Q.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            Q.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            Q.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            Q.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, act_func=Q.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            
        )
        self.classifier = nn.Sequential(
            Q.Linear(512 * 4 * 4, 1024, act_func=Q.ReLU()),  
            nn.BatchNorm1d(1024), 
            nn.ReLU(inplace=True),

            Q.Linear(1024, 1024, act_func=Q.ReLU()),
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

class LSQ_vgg9(nn.Module):
    def __init__(self, **kwargs):
        super(LSQ_vgg9, self).__init__()
        self.symmetric = False

        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            add_act(abits=kwargs['abits']),

            QConv(128, 128, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False, symmetric=self.symmetric),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            add_act(abits=kwargs['abits']),

            QConv(128, 256, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False, symmetric=self.symmetric),
            nn.BatchNorm2d(256),
            add_act(abits=kwargs['abits']),

            QConv(256, 256, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False, symmetric=self.symmetric),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            add_act(abits=kwargs['abits']),

            QConv(256, 512, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False, symmetric=self.symmetric),
            nn.BatchNorm2d(512),
            add_act(abits=kwargs['abits']),

            QConv(512, 512, wbits=kwargs['wbits'], kernel_size=3, stride=1, padding=1, bias=False, symmetric=self.symmetric),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            add_act(abits=kwargs['abits']),    
        )
        self.classifier = nn.Sequential(
            QLinear(512 * 4 * 4, 1024, wbits=kwargs['wbits'], symmetric=self.symmetric),  
            nn.BatchNorm1d(1024),
            add_act(abits=kwargs['abits']), 

            QLinear(1024, 1024, wbits=kwargs['wbits'], symmetric=self.symmetric),
            nn.BatchNorm1d(1024),
            add_act(abits=32 if kwargs['wbits']==32 else 32), 

            nn.Dropout(kwargs['drop']),
            nn.Linear(1024, kwargs['num_classes'], bias=True),
            nn.BatchNorm1d(kwargs['num_classes']),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Quant_vgg9(nn.Module):
    def __init__(self, **kwargs):
        super(Quant_vgg9, self).__init__()

        self.features = nn.Sequential(
            QuantConv(3, 128, wbits=32 if kwargs['wbits']==32 else 32, weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], ste=kwargs['ste']),

            QuantConv(128, 128, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits']),

            QuantConv(128, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits']),

            QuantConv(256, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nonlinear(abits=kwargs['abits']),

            QuantConv(256, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.BatchNorm2d(512),
            nonlinear(abits=kwargs['abits']),

            QuantConv(512, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], kernel_size=3, stride=1, padding=1, padding_mode=kwargs['padding_mode'], bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nonlinear(abits=kwargs['abits']),    
        )
        self.classifier = nn.Sequential(
            QuantLinear(512 * 4 * 4, 1024, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(1024),
            nonlinear(abits=kwargs['abits']), 

            QuantLinear(1024, 1024, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(1024),
            nonlinear(abits=32 if kwargs['wbits']==32 else 32), 

            nn.Dropout(kwargs['drop']),
            QuantLinear(1024, kwargs['num_classes'], wbits=32 if kwargs['wbits']==32 else 32, weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], bias=True),
            nn.BatchNorm1d(kwargs['num_classes']),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def nipq_vgg9(**kwargs: Any) -> NIPQ_vgg9:
    assert kwargs['dataset'] == 'cifar10', f"vgg model only supports CIFAR-10 dataset."

    return NIPQ_vgg9(**kwargs)
    
def lsq_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"vgg model only supports CIFAR-10 dataset."

    return LSQ_vgg9(**kwargs)

def quant_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"vgg model only supports CIFAR-10 dataset."

    return Quant_vgg9(**kwargs)