from __future__ import absolute_import

# The models below are modified from the baselines in order
# The models below contains baselines such as models from
# torchvision, and authors' models
from .resnet_baseline import *

# The models below contains multi-bit quantization such as models from
from .vgg9 import *
from .mlp import *
from .alexnet import *
from .resnet18_lsq import *
from .resnet18_nipq import *

# for partial sum compuatation 
from .psum_vgg9 import *
from .psum_mlp import *
from .psum_alexnet import *
from .psum_resnet18 import *
from .psum_resnet18_nipq import *
