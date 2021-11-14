#ifndef TGL_ELEMENTWISE_KERNELS_H
#define TGL_ELEMENTWISE_KERNELS_H

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

namespace tgl {
// Binary scalar operations

template <typename T>
using scalar_op = void( T *a, T b );

template <typename T, scalar_op<T> op>
__global__ void scalar_op_kernel( T *data, T value, int64_t nelem ) {
  assert( gridDim.y == 1 && gridDim.z == 1 );
  assert( blockDim.y == 1 && blockDim.z == 1 );

  int64_t stride = gridDim.x * blockDim.x;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  while( idx < nelem ) {
    op( data + idx, value );
    idx += stride;
  }
}

template <typename T>
__device__ void set_val_op( T *a, T b ) {
  *a = b;
}

template <typename T>
__device__ void set_if_zero_op( T *a, T b ) {
  if( *a == 0 )
    *a = b;
}

template <typename T>
__device__ void clip_above_op( T *a, T b ) {
  if( *a > b )
    *a = b;
}

template <typename T>
__device__ void clip_below_op( T *a, T b ) {
  if( *a < b )
    *a = b;
}

template <typename T>
__device__ void add_scalar_op( T *a, T b ) {
  *a += b;
}

template <typename T>
__device__ void sub_scalar_op( T *a, T b ) {
  *a -= b;
}

template <typename T>
__device__ void mult_scalar_op( T *a, T b ) {
  *a *= b;
}

template <typename T>
__device__ void div_scalar_op( T *a, T b ) {
  *a /= b;
}

template <typename T>
__device__ void fmod_scalar_op( T *a, T b ) {
  *a = fmod( *a, b );
}

template <typename T>
__device__ void pow_scalar_op( T *a, T b ) {
  *a = pow( *a, b );
}

// Binary tensor operations

template <typename T, typename U>
using binary_op = void( T *a, const U *b );

template <typename T, typename U, binary_op<T, U> op>
__global__ void binary_op_kernel( T *data, const U *other_data, int64_t nelem ) {
  assert( gridDim.y == 1 && gridDim.z == 1 );
  assert( blockDim.y == 1 && blockDim.z == 1 );

  int64_t stride = gridDim.x * blockDim.x;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  while( idx < nelem ) {
    op( data + idx, other_data + idx );
    idx += stride;
  }
}

template <typename T, typename U>
__device__ void add_op( T *a, const U *b ) {
  *a += *b;
}

template <typename T, typename U>
__device__ void sub_op( T *a, const U *b ) {
  *a -= *b;
}

template <typename T, typename U>
__device__ void mult_op( T *a, const U *b ) {
  *a *= *b;
}

template <typename T, typename U>
__device__ void div_op( T *a, const U *b ) {
  *a /= *b;
}

template <typename T, typename U>
__device__ void fmod_op( T *a, const U *b ) {
  *a = fmod( *a, *b );
}

template <typename T, typename U>
__device__ void pow_op( T *a, const U *b ) {
  *a = pow( *a, *b );
}

// Unary operations

template <typename T>
using unary_op = void( T *a );

template <typename T, unary_op<T> op>
__global__ void unary_op_kernel( T *data, int64_t nelem ) {
  assert( gridDim.y == 1 && gridDim.z == 1 );
  assert( blockDim.y == 1 && blockDim.z == 1 );

  int64_t stride = gridDim.x * blockDim.x;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  while( idx < nelem ) {
    op( data + idx );
    idx += stride;
  }
}

template <typename T>
__device__ void neg_op( T *a ) {
  *a = -( *a );
}

template <typename T>
__device__ void recip_op( T *a ) {
  *a = 1 / ( *a );
}

template <typename T>
__device__ void exp_op( T *a ) {
  *a = expf( *a );
}

template <typename T>
__device__ void fabs_op( T *a ) {
  *a = fabsf( *a );
}

template <typename T>
__device__ void log_op( T *a ) {
  *a = logf( *a );
}

template <typename T>
__device__ void log10_op( T *a ) {
  *a = log10f( *a );
}

template <typename T>
__device__ void sqrt_op( T *a ) {
  *a = sqrtf( *a );
}

} // namespace tgl
#endif /* TGL_ELEMENTWISE_KERNELS_H */
