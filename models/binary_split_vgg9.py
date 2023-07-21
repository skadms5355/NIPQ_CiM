import torch
import torch.nn as nn

from .binarized_psum_modules import *
from .binarized_modules import nonlinear, BinarizedNeurons
__all__ = ['binary_split_vgg9']

class PConv_BN_Merge(nn.Module):
    def __init__(self, inplane, outplane, wbits, weight_clip, weight_scale, padding_mode, arraySize, mapping_mode, is_noise, noise_type,
                 abits, mode, ste, offset, width, pool=False):
        super(PConv_BN_Merge, self).__init__()
        self.pconv = PsumBinConv(inplane, outplane, wbits=wbits, weight_clip=weight_clip, weight_scale=weight_scale, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode,
                        arraySize=arraySize, mapping_mode=mapping_mode, pbits=1, cbits=1, is_noise=is_noise, noise_type=noise_type)
        self.pooling = None
        if pool:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
            # self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.split_groups = self.pconv.split_groups
        self.BN = nn.BatchNorm2d(outplane*self.pconv.split_groups)
        self.hardtanh = nn.Hardtanh(-1, 1, inplace=True)
        self.binarized = BinarizedNeurons(mode='signed')
        self.qrelu = nonlinear(abits=abits, mode=mode, ste=ste, offset=offset, width=width)

    def merge_output(self, x):
        tmp = torch.chunk(x, self.split_groups, dim=1)

        output = tmp[0]
        for i in range(1, self.split_groups):
            output = output.add(tmp[i])

        return output
    
    def forward(self, x):
        x = self.pconv(x)
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.BN(x)
        x = self.hardtanh(x)
        x = self.binarized(x)
        x = self.merge_output(x)
        x = self.qrelu(x)

        return x 
    
class PLinear_BN_Merge(nn.Module):
    def __init__(self, infeatures, outfeatures, wbits, weight_clip, weight_scale, arraySize, mapping_mode, is_noise, noise_type,
                 abits, mode, ste, offset, width):
        super(PLinear_BN_Merge, self).__init__()
        self.plinear = PsumBinLinear(infeatures, outfeatures, wbits=wbits, weight_clip=weight_clip, weight_scale=weight_scale,
                        arraySize=arraySize, mapping_mode=mapping_mode, pbits=1, cbits=1, is_noise=is_noise, noise_type=noise_type)
        self.split_groups = self.plinear.split_groups
        self.BN = nn.BatchNorm1d(outfeatures*self.plinear.split_groups)
        self.hardtanh = nn.Hardtanh(-1, 1, inplace=True)
        self.binarized = BinarizedNeurons(mode='signed')
        self.qrelu = nonlinear(abits=abits, mode=mode, ste=ste, offset=offset, width=width)

    def merge_output(self, x):
        tmp = torch.chunk(x, self.split_groups, dim=1)

        output = tmp[0]
        for i in range(1, self.split_groups):
            output = output.add(tmp[i])
        return output

    def forward(self, x):
        x = self.plinear(x)
        x = self.BN(x)
        x = self.hardtanh(x)
        x = self.binarized(x)
        x = self.merge_output(x)
        x = self.qrelu(x)

        return x 


class BinarySplit_vgg(nn.Module):
    def __init__(self, **kwargs):
        super(BinarySplit_vgg, self).__init__()
        
        self.features = nn.Sequential(        
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nonlinear(abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            PConv_BN_Merge(128, 128, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'],
                        abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'],
                        pool=True),

            PConv_BN_Merge(128, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'],
                        abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'],
                        pool=False),

            PConv_BN_Merge(256, 256, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'],
                        abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'],
                        pool=True),

            PConv_BN_Merge(256, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'],
                        abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'],
                        pool=False),

            PConv_BN_Merge(512, 512, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale'], padding_mode=kwargs['padding_mode'],
                        arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'],
                        abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width'],
                        pool=True)
        )
        self.classifier = nn.Sequential(
            PLinear_BN_Merge(512 * 4 * 4, 1024, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale = kwargs['weight_scale'],
                            arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'],
                            abits=kwargs['abits'], mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),
            
            PLinear_BN_Merge(1024, 1024, wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale = kwargs['weight_scale'],
                            arraySize=kwargs['arraySize'], mapping_mode=kwargs['mapping_mode'], is_noise=kwargs['is_noise'], noise_type=kwargs['noise_type'],
                            abits=32, mode=kwargs['binary_mode'], ste=kwargs['ste'], offset=kwargs['x_offset'], width=kwargs['width']),

            nn.Linear(1024, kwargs['num_classes'], bias=True),
            # BinLinear(1024, kwargs['num_classes'], wbits=kwargs['wbits'], weight_clip=kwargs['weight_clip'], weight_scale=kwargs['weight_scale']),
            nn.BatchNorm1d(kwargs['num_classes'])
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def binary_split_vgg9(**kwargs):
    assert kwargs['dataset'] == 'cifar10', f"binarynet vgg model only supports CIFAR-10 dataset."

    return BinarySplit_vgg(**kwargs)