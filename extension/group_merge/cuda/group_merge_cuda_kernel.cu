//#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CEIL(x,y)   (((x) + (y) -1) / (y))

#define THREADS 512

using namespace at;

// group0(batch0, batch1,...), group1(batch0, batch1,...) 
template <typename scalar_t>
__global__ void merge_cuda_forward_kernel(
    scalar_t*__restrict__           output,
    scalar_t** __restrict__         input,
    int                             groups,
    size_t                          num_features_per_batch) {

    // get index for element
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    // break threads not in the range
    if (col >= num_features_per_batch) return;

    // get index & data
    uint idx = blockIdx.y * num_features_per_batch + col;

    //printf("col: %d, idx: %d\n", col, idx);

    // initialize output register
    scalar_t out_reg = 0;

    for ( int g = 0; g < groups; g++ ) {
        out_reg += input[g][idx];
        //printf("[col: %d, idx: %d] g: %d, input: %f\n", col, idx, g, input[g][idx]);
    }

    // update the computation result on the global memory
    output[idx] += out_reg;
}

template <typename data_t> /* data_t = scalar_t, but changed name to prevent crush with scalar_t on function call */
Tensor& merge_cuda(
    Tensor&                         output,
    std::vector<Tensor>             input,
    int                             groups,
    size_t                          batch_size,
    size_t                          num_features_per_batch) {

    // re-organize input information for GPU devices
    unsigned int inputPointersSize = groups * sizeof(data_t *);

    // get GPU storage for input
    auto d_inputs_storage = empty({inputPointersSize}, output.options().dtype(/*'at::'*/kByte));
    auto d_inputs = static_cast<data_t**>(d_inputs_storage.data_ptr()); 

    // get current GPU stream
    //at::cuda::CUDAStream stream = cuda::getCurrentCUDAStream();
    //auto stream = cuda::getCurrentCUDAStream();

    // copy input to pinned memory
    auto stackInputs_storage = empty({inputPointersSize}, 
                            output.options().dtype(/*'at::'*/kByte).device(/*'at::'*/kCPU).pinned_memory(true));
    auto stackInputs = static_cast<data_t**>(stackInputs_storage.data_ptr());
    for (int g = 0; g < groups; g++) {
        stackInputs[g] = input[g].data_ptr<data_t>();
    }
    
    // copy input information from CPU to GPU
    native::copy_(d_inputs_storage, stackInputs_storage, /* non_blocking= */ true);

    // launch kernel
    const dim3 grid( CEIL(num_features_per_batch, THREADS), batch_size);
    const dim3 blocks( THREADS, 1, 1 );
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "merge_forward_cuda", ([&] {
        merge_cuda_forward_kernel<scalar_t><<<grid, blocks>>>(
            output.data_ptr<scalar_t>(),
            (scalar_t**)d_inputs,
            groups,
            num_features_per_batch);
    }));

    return output;
}


// main forward function
Tensor group_merge_cuda_forward(
    Tensor&                     output,
    const std::vector<Tensor>   input,
    const int                   groups) {

    // get tensor information of input group
    const auto input_group = input[0];
    const auto batch_size = input_group.size(0);
    const auto num_features_per_batch = input_group.numel() / batch_size;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "merge_cuda", ([&] {
        merge_cuda<scalar_t>(
            output, input, groups, batch_size, num_features_per_batch);
        }));

    return output;
}

#undef CEIL
#undef THREADS
