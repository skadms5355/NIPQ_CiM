#include <torch/extension.h>

#include <vector>

using namespace at;


// CUDA forward declarations
Tensor conv_sweight_cuda_forward(
    Tensor&                     sweight,
    const Tensor                weight,
    const Tensor                group_in_offset,
    const int                   groups);

// C++ interface

//#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor conv_sweight_forward(
    Tensor&                     sweight,
    const Tensor                weight,
    const Tensor                group_in_offset,
    const int                   groups) {

    CHECK_INPUT(sweight);
    CHECK_INPUT(weight);

    return conv_sweight_cuda_forward(sweight, weight, group_in_offset, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_sweight_forward, "conv_sweight forward (CUDA)");
//  m.def("backward", &pquant_backward, "pquant backward (CUDA)");
}

