//#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#define CEIL(x,y)   (((x) + (y) -1) / (y))

#define THREADS 512

using namespace at;

// quantization function
template <typename scalar_t>
inline __device__ scalar_t psum_quantization(scalar_t data, 
    float scaled_step, scalar_t half_num_levels, float center, float pbits, bool pzero) {

    // STEP1. centering data
    data = data - center;

    // STEP2. scale the data
    data = data / scaled_step;

    // STEP3. quantize the data
    if (pzero) {
        data = roundf( data );
        data = (data > half_num_levels)? half_num_levels : data;
        data = (data < -1 * half_num_levels)? -1 * half_num_levels : data;
    } else {
        if ( pbits == 1) {
            data = (data >= 0)? 1 : -1;
        } else {
            //data = data + 0.5;
            data = roundf( data );
            data = (data > half_num_levels)? half_num_levels : data;
            data = (data < 1 - half_num_levels)? 1 - half_num_levels : data;
            //data = data - 0.5;
        }
    }
    
    // STEP4. scale/center back the data
    data = data * scaled_step + center;

    return data;
}

// group0(batch0, batch1,...), group1(batch0, batch1,...) 
template <typename scalar_t>
__global__ void pquant_cuda_forward_kernel(
    scalar_t**__restrict__           output,
    scalar_t** __restrict__         input,
    float                           scaled_step,
    scalar_t                        half_num_levels,
    float                           center,
    int                             groups,
    float                           pbits,
    bool                            pzero,
    size_t                          num_features_per_batch) {

    // get index for element
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    // break threads not in the range
    if (col >= num_features_per_batch) return;

    // get index & data
    uint idx = blockIdx.y * num_features_per_batch + col;
    scalar_t data = input[0][idx];

    // define local registers for next step
    scalar_t data_next;

    for ( int g = 0; g < groups; g++ ) {
        // STEP1. prefetch the next iteration data & index
        if (g < groups - 1) {
            data_next = input[g+1][idx];
        }

        // STEP2. do the computation
        output[g][idx] = psum_quantization<scalar_t>(data, scaled_step, half_num_levels, center, pbits, pzero);

        // STEP3. update for the next stage
        data = data_next;
    }

}

template <typename data_t> /* data_t = scalar_t, but changed name to prevent crush with scalar_t on function call */
std::vector<Tensor>& pquant_cuda(
    std::vector<Tensor>&            output,
    std::vector<Tensor>             input,
    float                           scaled_step,
    float                           half_num_levels,
    float                           center,
    int                             groups,
    float                           pbits,
    bool                            pzero,
    size_t                          batch_size,
    size_t                          num_features_per_batch) {

    // re-organize input information for GPU devices
    unsigned int inputPointersSize = groups * sizeof(data_t *);

    /** copy vector pointers of input from cpu to gpu**/
    // get GPU storage for input
    auto d_inputs_storage = empty({inputPointersSize}, output[0].options().dtype(/*'at::'*/kByte));
    auto d_inputs = static_cast<data_t**>(d_inputs_storage.data_ptr()); 

    // copy input to pinned memory
    auto stackInputs_storage = empty({inputPointersSize}, 
                            output[0].options().dtype(/*'at::'*/kByte).device(/*'at::'*/kCPU).pinned_memory(true));
    auto stackInputs = static_cast<data_t**>(stackInputs_storage.data_ptr());
    for (int g = 0; g < groups; g++) {
        stackInputs[g] = input[g].data_ptr<data_t>();
    }
    
    // copy input information from CPU to GPU
    native::copy_(d_inputs_storage, stackInputs_storage, /* non_blocking= */ true);

    /** copy vector pointers of outpt from cpu to gpu**/
    // get GPU storage for input
    auto d_outputs_storage = empty({inputPointersSize}, output[0].options().dtype(/*'at::'*/kByte));
    auto d_outputs = static_cast<data_t**>(d_outputs_storage.data_ptr()); 

    // copy input to pinned memory
    auto stackOutputs_storage = empty({inputPointersSize}, 
                            output[0].options().dtype(/*'at::'*/kByte).device(/*'at::'*/kCPU).pinned_memory(true));
    auto stackOutputs = static_cast<data_t**>(stackOutputs_storage.data_ptr());
    for (int g = 0; g < groups; g++) {
        stackOutputs[g] = output[g].data_ptr<data_t>();
    }
    
    // copy input information from CPU to GPU
    native::copy_(d_outputs_storage, stackOutputs_storage, /* non_blocking= */ true);

    // launch kernel
    const dim3 grid( CEIL(num_features_per_batch, THREADS), batch_size);
    const dim3 blocks( THREADS, 1, 1 );
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output[0].scalar_type(), "pquant_forward_cuda", ([&] {
        pquant_cuda_forward_kernel<scalar_t><<<grid, blocks>>>(
            (scalar_t**)d_outputs,
            (scalar_t**)d_inputs,
            scaled_step,
            (scalar_t)half_num_levels,
            center,
            groups,
            pbits,
            pzero,
            num_features_per_batch);
    }));

    return output;
}



// main forward function
std::vector<Tensor> pquant_cuda_forward(
    std::vector<Tensor>&        output,
    const std::vector<Tensor>   input,
    const float                 pbits,
    const float                 step,
    const float                 half_num_levels,
    const float                 weight,
    const float                 center,
    const int                   groups,
    const bool                  pzero) {


    if ( pbits == 32 ) {
        output.assign(input.begin(), input.end());
    } else {
        // get tensor information of input group
        const auto input_group = input[0];
        const auto batch_size = input_group.size(0);
        const auto num_features_per_batch = input_group.numel() / batch_size;

        float scaled_step = step * weight; // scale step to bitplane

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(output[0].scalar_type(), "pquant_cuda", ([&] {
            pquant_cuda<scalar_t>(output, input, scaled_step,
                                half_num_levels, center,
                                groups, pbits, pzero, batch_size,
                                num_features_per_batch);
        }));
    }

    return output;
}

#undef CEIL
#undef THREADS
