#ifndef TGL_ELEMENTWISE_OPS_H
#define TGL_ELEMENTWISE_OPS_H

#include <cassert>
#include <cstdio>

#include "common.h"

namespace tgl
{
// Scalar operations

template <typename T> using scalar_op = void( T* a, T b );

template <typename T, scalar_op<T> op> __global__ void scalar_op_kernel( T* data, T value, int64_t nelem )
{
    assert( gridDim.y == 1 && gridDim.z == 1 );
    assert( blockDim.y == 1 && blockDim.z == 1 );

    int64_t stride = gridDim.x * blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while( idx < nelem ) {
        op( data + idx, value );
        idx += stride;
    }
}

template <typename T> __device__ void set_val_op( T* a, T b )
{
    *a = b;
}

template <typename T> __device__ void set_if_zero_op( T* a, T b )
{
    if( *a == 0 )
        *a = b;
}

template <typename T> __device__ void add_scalar_op( T* a, T b )
{
    *a += b;
}

template <typename T> __device__ void sub_scalar_op( T* a, T b )
{
    *a -= b;
}

template <typename T> __device__ void mult_scalar_op( T* a, T b )
{
    *a *= b;
}

template <typename T> __device__ void div_scalar_op( T* a, T b )
{
    *a /= b;
}

// Binary operations

template <typename T, typename U> using binary_op = void( T* a, U* b );

template <typename T, typename U, binary_op<T, U> op>
__global__ void binary_op_kernel( T* data, U* other_data, int64_t nelem )
{
    assert( gridDim.y == 1 && gridDim.z == 1 );
    assert( blockDim.y == 1 && blockDim.z == 1 );

    int64_t stride = gridDim.x * blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while( idx < nelem ) {
        op( data + idx, other_data + idx );
        idx += stride;
    }
}

template <typename T, typename U> __device__ void add_op( T* a, U* b )
{
    *a += *b;
}

template <typename T, typename U> __device__ void sub_op( T* a, U* b )
{
    *a -= *b;
}

template <typename T, typename U> __device__ void mult_op( T* a, U* b )
{
    *a *= *b;
}

template <typename T, typename U> __device__ void div_op( T* a, U* b )
{
    *a /= *b;
}

// Unary operations

template <typename T> using unary_op = void( T* a );

template <typename T, unary_op<T> op> __global__ void unary_op_kernel( T* data, int64_t nelem )
{
    assert( gridDim.y == 1 && gridDim.z == 1 );
    assert( blockDim.y == 1 && blockDim.z == 1 );

    int64_t stride = gridDim.x * blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    while( idx < nelem ) {
        op( data + idx );
        idx += stride;
    }
}

template <typename T> __device__ void neg_op( T* a )
{
    *a = -( *a );
}

template <typename T> __device__ void recip_op( T* a )
{
    *a = 1 / ( *a );
}
} // namespace tgl
#endif /* TGL_ELEMENTWISE_OPS_H */
