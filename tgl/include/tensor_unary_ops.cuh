#ifndef TGL_TENSOR_UNARY_OPS_H
#define TGL_TENSOR_UNARY_OPS_H

#include <cstdio>
#include <cassert>

#include "common.h"

namespace tgl {

template<typename T>
using unary_op = void (T* a);

template<typename T, typename U, unary_op<T> op>
__global__ void binary_op_kernel(T *data, int64_t nelem) {
    assert(gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.y == 1 && blockDim.z == 1);

    int64_t stride = gridDim.x * blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < nelem) {
        op(data + idx);
        idx += stride;
    }
}

}
#endif /* TGL_TENSOR_UNARY_OPS_H */
