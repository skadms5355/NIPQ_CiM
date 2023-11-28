#include <torch/extension.h>

#include <vector>

using namespace at;


// CUDA forward declarations
Tensor pquant_scale_group_merge_cuda_forward(
    Tensor&                     output,
    const std::vector<Tensor>   input,
    const float                 pbits,
    const std::vector<float>&   step,
    const float                 half_num_levels,
    const float                 weight,
    const float                 center,
    const int                   groups,
    const bool                  pzero);

// C++ interface

//#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor pquant_scale_group_merge_forward(
    Tensor&                     output,
    const std::vector<Tensor>   input,
    const float                 pbits,
    const std::vector<float>&   step,
    const float                 half_num_levels,
    const float                 weight,
    const float                 center,
    const int                   groups,
    const bool                  pzero) {

    CHECK_INPUT(output);
    CHECK_INPUT(input[0]);

    return pquant_scale_group_merge_cuda_forward(output, input, pbits, step, half_num_levels, weight, center, groups, pzero);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pquant_scale_group_merge_forward, "pquant group merge forward (CUDA)");
//  m.def("backward", &pquant_merge_backward, "pquant merge backward (CUDA)");
}

