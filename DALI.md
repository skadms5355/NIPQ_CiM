# Installation and Usage of NVIDIA DALI

## What is DALI?

NVIDIA Data Loading Library (DALI) is a collection of highly optimized building blocks, and an execution engine, to accelerate the pre-processing of the input data for deep learning applications. For more information please refer to [official website](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html).

## Installation

Here we introduce how to install DALI on the environment described at [README.md](https://github.com/Hyungjun-K1m/BNN_AtoZ/tree/master/src/README.md).

### Prerequisites

1. Linux x64.
2. NVIDIA Driver >= 410.48 (CUDA 10.0 or later)
3. Pytorch >= 0.4

### Installation

1. `conda update pip`
2. `pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda100`
  OR `pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110` for CUDA 11

It is possible to compile DALI from source. Please refer to [here](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/compilation.html) for more detailed instruction.

## When to use?

We recommend to use DALI dataloader only if ImageNet dataset is used since speed gain is negligible with smaller datasets.

1. When using only 1 GPU, training speed (images/s) is almost constant. However, GPU/CPU usage differs.
 - Use torchvision when GPU memory is not enough.
 - Use DALI GPU by using `--dali true` argument when CPU becomes bottleneck.
 - You can also use DALI CPU by using `--dali true --dali-cpu true` arguments if you want middle way.

2. When using more than 1 GPU, DALI is faster than torchvision.
 - Since DALI CPU and DALI GPU have similar speed, you may choose one of them considering GPU/CPU usage balance.
 
