//#include <torch/extension.h>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CEIL(x,y)   (((x) + (y) -1) / (y))

#define THREADS 128

using namespace at;

template <typename scalar_t>
__global__ void conv_sweight_cuda_forward_kernel(
    scalar_t*__restrict__           sweight,
    scalar_t* __restrict__          weight,
    int*     __restrict__           group_in_offset,
    size_t                          sweight_group_fanin,
    size_t                          sweight_group,
    size_t                          weight_group_fanin,
    size_t                          weight_fanin,
    int                             groups) {

    // get index for element
    uint sweight_in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // break threads not in the range
    if (sweight_in_idx >= sweight_group_fanin) return;

    // get index & data
    uint oc_idx = blockIdx.y;
    int weight_in_idx;
    uint sweight_idx, weight_idx;
    int in_offset;
    
    for (int g = 0; g < groups; g++) {
        // get sweight_idx
        sweight_idx = g * sweight_group + oc_idx * sweight_group_fanin + sweight_in_idx;
        // update weight_in_idx with offset
        in_offset = group_in_offset[g];
        weight_in_idx = sweight_in_idx - in_offset;
        if ( (weight_in_idx < 0) | (weight_in_idx >= weight_group_fanin)) {
            sweight[sweight_idx] = 0;
        } else {
            weight_idx = oc_idx * weight_fanin + g * weight_group_fanin  + (sweight_in_idx - in_offset);
            sweight[sweight_idx] = weight[weight_idx];
        }
    }

}


Tensor conv_sweight_cuda_forward(
    Tensor&                 sweight,
    const Tensor            weight,
    const Tensor            group_in_offset,
    const int               groups) {

    //// get tensor information of input group
    //const auto batch_size = input.size(0);
    //const auto num_features_per_batch = input.numel() / batch_size;
    
    const auto kSpatial = sweight.size(2) * sweight.size(3);
    const auto group_out_channels = weight.size(0);

    const auto sweight_group_fanin = sweight.size(1) * kSpatial;
    const auto sweight_group = sweight_group_fanin * group_out_channels;

    const auto weight_fanin = weight.size(1) * kSpatial;
    const auto weight_group_fanin = weight_fanin / groups;


    // launch kernel
    const dim3 grid( CEIL(sweight_group_fanin, THREADS), group_out_channels);
    const dim3 blocks( THREADS, 1, 1 );
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sweight.scalar_type(), "conv_sweight_forward_cuda", ([&] {
        conv_sweight_cuda_forward_kernel<scalar_t><<<grid, blocks>>>(
            sweight.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            group_in_offset.data_ptr<int>(),
            (size_t)sweight_group_fanin,
            (size_t)sweight_group,
            (size_t)weight_group_fanin,
            (size_t)weight_fanin,
            groups);
    }));

    return sweight;
}

#undef CEIL
#undef THREADS
