from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='conv_sweight',
    ext_modules=[
            CUDAExtension(
                name='conv_sweight_cuda',
                sources=['conv_sweight_cuda.cpp','conv_sweight_cuda_kernel.cu'],
                # Command parameters for compiling extension package(Extra compilation coptions)
                extra_compile_args={
                    'cxx': [],
                    'nvcc': [
                        '-gencode=arch=compute_37,code=sm_37',
                        '-gencode=arch=compute_50,code=sm_50',
                        '-gencode=arch=compute_60,code=sm_60',
                        '-gencode=arch=compute_61,code=sm_61',
                        '-gencode=arch=compute_70,code=sm_70',
                        '-gencode=arch=compute_75,code=sm_75',
                        '-gencode=arch=compute_80,code=sm_80',
                        '-gencode=arch=compute_37,code=compute_37'
                    ]
                }
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

