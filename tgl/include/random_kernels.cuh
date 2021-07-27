#ifndef TGL_RANDOM_INIT_KERNELS_H
#define TGL_RANDOM_INIT_KERNELS_H

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

template <typename T>
using random_generator = T( curandState_t * );

template <typename T>
__device__ T uniform_generator( curandState_t *state )
{
    assert(0);
}

template <>
__device__ inline int8_t uniform_generator<int8_t>( curandState_t *state )
{
    return static_cast<int8_t>( curand( state ) );
}

template <>
__device__ inline int64_t uniform_generator<int64_t>( curandState_t *state )
{
    int64_t temp = static_cast<int64_t>( curand( state ) );
    return ( temp << 32 ) | static_cast<int32_t>( curand( state ) );
}

template <>
__device__ inline float uniform_generator<float>( curandState_t *state )
{
    return curand_uniform( state );
}

template <>
__device__ inline double uniform_generator<double>( curandState_t *state )
{
    return curand_uniform_double( state );
}

template <typename T>
__device__ T normal_generator( curandState_t *state )
{  
    assert(0);
}

template <>
__device__ inline int8_t normal_generator<int8_t>( curandState_t *state )
{
    printf("TGL Warning: Initialization based on normal distribution is not available for int8_t\n");
    return -1;
}

template <>
__device__ inline int64_t normal_generator<int64_t>( curandState_t *state )
{
    printf("TGL Warning: Initialization based on normal distribution is not available for int64_t\n");
    return -1;
}

template <>
__device__ inline float normal_generator<float>( curandState_t *state )
{
    return curand_normal( state );
}

template <>
__device__ inline double normal_generator<double>( curandState_t *state )
{
    return curand_normal_double( state );
}

template <typename T, random_generator<T> gen>
__global__ void random_init_kernel( T *data, int64_t nelem, int64_t seed )
{
    assert( gridDim.y == 1 && gridDim.z == 1 );
    assert( blockDim.y == 1 && blockDim.z == 1 );

    int64_t stride = gridDim.x * blockDim.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init( seed, idx, 0, &state );

    while ( idx < nelem ) {
        data[idx] = gen( &state );
        idx += stride;
    }
}

#endif /* TGL_RANDOM_INIT_KERNELS_H */
