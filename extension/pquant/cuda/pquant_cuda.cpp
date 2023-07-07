#include <torch/extension.h>

#include <vector>

using namespace at;


// CUDA forward declarations
std::vector<Tensor> pquant_cuda_forward(
    std::vector<Tensor>&        output,
    const std::vector<Tensor>   input,
    const float                 pbits,
    const float                 step,
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

std::vector<Tensor> pquant_forward(
    std::vector<Tensor>&        output,
    const std::vector<Tensor>   input,
    const float                 pbits,
    const float                 step,
    const float                 half_num_levels,
    const float                 weight,
    const float                 center,
    const int                   groups,
    const bool                  pzero) {

    CHECK_INPUT(output[0]);
    CHECK_INPUT(input[0]);

    return pquant_cuda_forward(output, input, pbits, step, half_num_levels, weight, center, groups, pzero);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pquant_forward, "pquant forward (CUDA)");
//  m.def("backward", &pquant_backward, "pquant backward (CUDA)");
}

