#ifndef TGL_CUDA_KERNELS_H
#define TGL_CUDA_KERNELS_H

#include <cstdio>

#include "common.h"

namespace tgl {

template<typename T>
__global__ void fill_kernel(T *data, int64_t nelem, T val) {
    assert(gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    //ToDo: fetch to shared memory
    int64_t N = ceilf(nelem / static_cast<float>(gridDim.x * blockDim.x));
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    int64_t stop_idx = min(idx + N, nelem);
    while (idx < stop_idx) {
        data[idx] = val;
        ++idx;
    }
}

template<typename T, typename U>
using binary_op = void (T* a, U* b);

template<typename T, typename U, binary_op<T, U> op>
__global__ void binary_op_kernel(T *data, U *other_data, int64_t nelem) {
    assert(gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    //ToDo: fetch to shared memory
    int64_t N = ceilf(nelem / static_cast<float>(gridDim.x * blockDim.x));
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    int64_t stop_idx = min(idx + N, nelem);
    data += idx;
    other_data += idx;
    while (idx < stop_idx) {
        assert(idx < stop_idx);
        op(data++, other_data++);
        ++idx;
    }
}

template<typename T, typename U>
__device__ void add_op(T *a, U *b) {
    *a += *b;
}

template<typename T, typename U>
void launch_add_kernel(T *data, U *other_data, int64_t data_size, dim3 &grid_dims, dim3 &block_dims,
        cudaStream_t &stream) {
    binary_op_kernel<T, U, add_op<T, U>> <<<grid_dims, block_dims, 0, stream>>>(data, other_data,
            data_size);
    check_cuda_error();
}

template<typename T, typename U>
__device__ void mult_op(T *a, U *b) {
    *a *= *b;
}

template<typename T, typename U>
void launch_mult_kernel(T *data, U *other_data, int64_t data_size, dim3 &grid_dims, dim3 &block_dims,
        cudaStream_t &stream) {
    binary_op_kernel<T, U, mult_op<T, U>> <<<grid_dims, block_dims, 0, stream>>>(data, other_data,
            data_size);
    check_cuda_error();
}

}
#endif /* TGL_CUDA_KERNELS_H */
