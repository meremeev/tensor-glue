#ifndef TGL_TENSOR_BINARY_OPS_H
#define TGL_TENSOR_BINARY_OPS_H

#include <cstdio>
#include <cassert>

#include "common.h"

namespace tgl {

template<typename T, typename U>
using binary_op = void (T* a, U* b);

template<typename T, typename U, binary_op<T, U> op>
__global__ void binary_op_kernel(T *data, U *other_data, int64_t nelem) {
    assert(gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    int64_t stride = gridDim.x * blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < nelem) {
        op(data + idx, other_data + idx);
        idx += stride;
    }
}

template<typename T, typename U>
__device__ void add_op(T *a, U *b) {
    *a += *b;
}

template<typename T, typename U>
void launch_add_kernel(T *data, U *other_data, int64_t data_size, dim3 &grid_dims, dim3 &block_dims,
        cudaStream_t &stream) {
    binary_op_kernel<T, U, add_op<T, U>> <<<grid_dims, block_dims, 0, stream>>>(data, other_data, data_size);
    check_cuda_error();
}

template<typename T, typename U>
__device__ void mult_op(T *a, U *b) {
    *a *= *b;
}

template<typename T, typename U>
void launch_mult_kernel(T *data, U *other_data, int64_t data_size, dim3 &grid_dims, dim3 &block_dims,
        cudaStream_t &stream) {
    binary_op_kernel<T, U, mult_op<T, U>> <<<grid_dims, block_dims, 0, stream>>>(data, other_data, data_size);
    check_cuda_error();
}

}
#endif /* TGL_TENSOR_BINARY_OPS_H */
