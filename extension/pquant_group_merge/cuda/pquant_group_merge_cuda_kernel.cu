//#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

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

    scalar_t zero = 0;

    // STEP3. quantize the data
    if (pzero) {
        if ( pbits == 1) {
            data = (data >= 0)? 1 : -1;
        } else {
            //data = data + 0.5;
            data = roundf( data );
            data = (data > half_num_levels - 1)? half_num_levels - 1  : data;
            data = (data < - half_num_levels)? - half_num_levels : data;
            //data = data - 0.5;
        }
    } else {
        data = roundf( data );
        data = (data > 2 * half_num_levels - 1)? 2 * half_num_levels - 1 : data;
        data = (data < zero)? zero : data;
    }
    
    
    // STEP4. scale/center back the data
    data = data * scaled_step + center;

    return data;
}

// group0(batch0, batch1,...), group1(batch0, batch1,...) 
template <typename scalar_t>
__global__ void pquant_merge_cuda_forward_kernel(
    scalar_t*__restrict__           output,
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

    // initialize output register
    scalar_t out_reg = 0;

    for ( int g = 0; g < groups; g++ ) {
        // STEP1. prefetch the next iteration data & index
        if (g < groups - 1) {
            data_next = input[g+1][idx];
        }

        // STEP2. do the computation
        out_reg += psum_quantization<scalar_t>(data, scaled_step, half_num_levels, center, pbits, pzero);

        // STEP3. update for the next stage
        data = data_next;
    }

    // update the computation result on the global memory
    output[idx] += out_reg;
}

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
Tensor& pquant_merge_cuda(
    Tensor&                         output,
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

    // get GPU storage for input
    auto d_inputs_storage = empty({inputPointersSize}, output.options().dtype(/*'at::'*/kByte));
    auto d_inputs = static_cast<data_t**>(d_inputs_storage.data_ptr()); 

    // get current GPU stream
    //c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();

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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "pquant_merge_forward_cuda", ([&] {
        pquant_merge_cuda_forward_kernel<scalar_t><<<grid, blocks>>>(
        //pquant_merge_cuda_forward_kernel<scalar_t><<<grid, blocks, 0, stream.stream()>>>(
            output.data_ptr<scalar_t>(),
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
Tensor pquant_group_merge_cuda_forward(
    Tensor&                     output,
    const std::vector<Tensor>   input,
    const float                 pbits,
    const float                 step,
    const float                 half_num_levels,
    const float                 weight,
    const float                 center,
    const int                   groups,
    const bool                  pzero) {

    // get tensor information of input group
    const auto input_group = input[0];
    const auto batch_size = input_group.size(0);
    const auto num_features_per_batch = input_group.numel() / batch_size;


    if ( pbits == 32 ) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "merge_cuda", ([&] {
            merge_cuda<scalar_t>(
                output, input, groups, batch_size, num_features_per_batch);
        }));
    } else {
        float scaled_step = step * weight; // scale step to bitplane
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "pquant_merge_cuda", ([&] {
            pquant_merge_cuda<scalar_t>(output, input, scaled_step,
                                        half_num_levels, center,
                                        groups, pbits, pzero, batch_size,
                                        num_features_per_batch);
        }));
    }

    return output;
}

#undef CEIL
#undef THREADS
