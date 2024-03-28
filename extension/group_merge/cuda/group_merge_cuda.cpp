#include <torch/extension.h>

#include <vector>

using namespace at;


// CUDA forward declarations
Tensor group_merge_cuda_forward(
    Tensor&                     output,
    const std::vector<Tensor>   input,
    const int                   groups);

// C++ interface

//#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor group_merge_forward(
    Tensor&                     output,
    const std::vector<Tensor>   input,
    const int                   groups) {

    CHECK_INPUT(output);
    CHECK_INPUT(input[0]);

    return group_merge_cuda_forward(output, input, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_merge_forward, "group merge forward (CUDA)");
//  m.def("backward", &pquant_merge_backward, "pquant merge backward (CUDA)");
}

