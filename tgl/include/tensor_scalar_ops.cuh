#ifndef TGL_TENSOR_SCALAR_OPS_H
#define TGL_TENSOR_SCALAR_OPS_H

#include <cstdio>
#include <cassert>

#include "common.h"

namespace tgl {

template<typename T>
using scalar_op = void (T* a, T b);

template<typename T, scalar_op<T> op>
__global__ void scalar_op_kernel(T *data, T literal, int64_t nelem) {
    assert(gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    //ToDo: fetch to shared memory
    int64_t N = ceilf(nelem / static_cast<float>(gridDim.x * blockDim.x));
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    int64_t stop_idx = min(idx + N, nelem);
    data += idx;
    while (idx < stop_idx) {
        assert(idx < stop_idx);
        op(data++, literal);
        ++idx;
    }
}

template<typename T>
__device__ void set_val_op(T *a, T b) {
    *a = b;
}

template<typename T>
void launch_fill_kernel(T *data, T literal, int64_t data_size, dim3 &grid_dims, dim3 &block_dims,
        cudaStream_t &stream) {
    scalar_op_kernel<T, set_val_op<T>> <<<grid_dims, block_dims, 0, stream>>>(data, literal,
            data_size);
    check_cuda_error();
}

template<typename T>
__device__ void add_scalar_op(T *a, T b) {
    *a += b;
}

template<typename T>
void launch_add_scalar_kernel(T *data, T literal, int64_t data_size, dim3 &grid_dims, dim3 &block_dims,
        cudaStream_t &stream) {
    scalar_op_kernel<T, add_scalar_op<T>> <<<grid_dims, block_dims, 0, stream>>>(data, literal,
            data_size);
    check_cuda_error();
}

template<typename T>
__device__ void mult_scalar_op(T *a, T b) {
    *a *= b;
}

template<typename T>
void launch_mult_scalar_kernel(T *data, T literal, int64_t data_size, dim3 &grid_dims, dim3 &block_dims,
        cudaStream_t &stream) {
    scalar_op_kernel<T, mult_scalar_op<T>> <<<grid_dims, block_dims, 0, stream>>>(data, literal,
            data_size);
    check_cuda_error();
}

}

#endif /* TGL_TENSOR_SCALAR_OPS_H */
