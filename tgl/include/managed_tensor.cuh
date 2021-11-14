#ifndef TGL_MANAGED_TENSOR_H
#define TGL_MANAGED_TENSOR_H

#include <cstdint>
#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"
#include "elementwise_kernels.cuh"
#include "random_kernels.cuh"

namespace tgl {

constexpr int64_t THREADS_PER_BLOCK = 256;

using TensorDims = std::vector<int64_t>;
using TensorIndex = std::vector<int64_t>;
using IndexRange = std::vector<std::array<int64_t, 2>>;

template <typename T,
          typename = typename std::enable_if<( !std::is_pointer<T>::value && !std::is_const<T>::value ), T>::type>
class ManagedTensor {
public:
  // ToDo: provide ctor with device selection and kernel launch specification
  // ToDo: provide automatic device selection for multi-GPU platform
  ManagedTensor( const TensorDims &dims, bool set_to_zero = false ) noexcept( false )
      : dims_( dims ), lines_( dims_.size(), 1 ), nelem_( 1 ) {
    init_layout();
    cuda_check( cudaMallocManaged( &data_, data_size_ ) );
    if( set_to_zero ) {
      cuda_check( cudaMemset( data_, 0, data_size_ ) );
    }
  }

  ManagedTensor( const TensorDims &dims, T *data, bool take_ownership = false ) noexcept( false )
      : dims_( dims ), lines_( dims_.size(), 1 ), nelem_( 1 ) {
    init_layout();
    try {
      std::ignore = *data;
    } catch( ... ) {
      throw std::runtime_error( "Attempt to create tgl::tensor from invalid data pointer" );
    }

    if( !take_ownership ) {
      cuda_check( cudaMallocManaged( &data_, data_size_ ) );
      cuda_check( cudaMemcpy( data_, data, data_size_, cudaMemcpyHostToDevice ) );
    } else {
      cudaPointerAttributes mem_attr;
      cudaPointerGetAttributes( &mem_attr, data );
      if( mem_attr.type == cudaMemoryType::cudaMemoryTypeManaged ) {
        data_ = data;
      } else {
        throw std::runtime_error( "Attempt to take ownershop over not CUDA managed memory" );
      }
    }
  }

  explicit ManagedTensor( const ManagedTensor<T> &other ) noexcept( false ) : ManagedTensor( other.dims_ ) {
    set_stream( other.get_stream() );
    other.sync();
    cuda_check( cudaMemcpy( data_, other.data_, data_size_, cudaMemcpyDefault ) );
  }

  ManagedTensor( ManagedTensor<T> &&other )
      : dims_( other.dims_ ), lines_( other.lines_ ), nelem_( other.nelem_ ), data_size_( other.data_size_ ),
        data_( other.data_ ), grid_dims_( other.grid_dims_ ), block_dims_( other.block_dims_ ),
        stream_( other.stream_ ), device_( other.device_ ) {

    other.dims_.clear();
    other.lines_.clear();
    other.nelem_ = 0;
    other.data_size_ = 0;
    other.data_ = nullptr;
  }

  virtual ~ManagedTensor() noexcept( false ) {
    sync();
    if( data_ ) {
      cuda_check( cudaFree( data_ ) );
    }
  }

  ManagedTensor &operator=( const ManagedTensor &other ) = delete;
  ManagedTensor &operator=( ManagedTensor &&other ) = delete;

  int64_t ndims() const {
    return dims_.size();
  }

  int64_t size() const {
    return nelem_;
  }

  int64_t size( int64_t i ) const {
    return dims_[i];
  }

  const TensorDims &dims() const {
    return dims_;
  }

  const T *data() const {
    return data_;
  }

  T *data() {
    return data_;
  }

  void set_stream( cudaStream_t stream ) const {
    stream_ = stream;
  }

  cudaStream_t get_stream() const {
    return stream_;
  }

  int get_device() const {
    return device_;
  }

  void prefetch( int device ) const {
    cuda_check( cudaMemPrefetchAsync( data_, data_size_, device, stream_ ) );
    device_ = device;
  }

  void sync() const {
    cuda_check( cudaStreamSynchronize( stream_ ) );
  }

  const T &operator[]( TensorIndex index ) const {
    sync();
    return data_[get_flat_index( index )];
  }

  T &operator[]( TensorIndex index ) {
    sync();
    return data_[get_flat_index( index )];
  }

  const T &operator[]( int64_t index ) const {
    sync();
    return data_[index];
  }

  T &operator[]( int64_t index ) {
    sync();
    return data_[index];
  }

  // Scalar operations
  ManagedTensor<T> &set_val( T value ) {
    launch_kernel<set_val_op<T>>( value );
    return *this;
  }

  ManagedTensor<T> &set_if_zero( T value ) {
    launch_kernel<set_if_zero_op<T>>( value );
    return *this;
  }

  ManagedTensor<T> &clip_above( T value ) {
    launch_kernel<clip_above_op<T>>( value );
    return *this;
  }

  ManagedTensor<T> &clip_below( T value ) {
    launch_kernel<clip_below_op<T>>( value );
    return *this;
  }

  ManagedTensor<T> &fill_random_uniform( int64_t seed ) {
    launch_kernel<uniform_generator<T>>();
    return *this;
  }

  ManagedTensor<T> &fill_random_normal( int64_t seed ) {
    launch_kernel<normal_generator<T>>();
    return *this;
  }

  ManagedTensor<T> &operator+=( T value ) {
    launch_kernel<add_scalar_op<T>>( value );
    return *this;
  };

  ManagedTensor<T> &operator-=( T value ) {
    launch_kernel<sub_scalar_op<T>>( value );
    return *this;
  };

  ManagedTensor<T> &operator*=( T value ) {
    launch_kernel<mult_scalar_op<T>>( value );
    return *this;
  }

  ManagedTensor<T> &operator/=( T value ) {
    launch_kernel<div_scalar_op<T>>( value );
    return *this;
  }

  ManagedTensor<T> &operator%=( T value ) {
    launch_kernel<fmod_scalar_op<T>>( value );
    return *this;
  }

  ManagedTensor<T> &pow( T value ) {
    launch_kernel<pow_scalar_op<T>>( value );
    return *this;
  }

  // Binary operations
  template <typename U>
  ManagedTensor<T> &operator+=( const ManagedTensor<U> &other ) {
    launch_kernel<U, add_op<T, U>>( other );
    return *this;
  }

  template <typename U>
  ManagedTensor<T> &operator-=( const ManagedTensor<U> &other ) {
    launch_kernel<U, sub_op<T, U>>( other );
    return *this;
  }

  template <typename U>
  ManagedTensor<T> &operator*=( const ManagedTensor<U> &other ) {
    launch_kernel<U, mult_op<T, U>>( other );
    return *this;
  }

  template <typename U>
  ManagedTensor<T> &operator/=( const ManagedTensor<U> &other ) {
    launch_kernel<U, div_op<T, U>>( other );
    return *this;
  }

  template <typename U>
  ManagedTensor<T> &operator%=( const ManagedTensor<U> &other ) {
    launch_kernel<U, fmod_op<T, U>>( other );
    return *this;
  }

  template <typename U>
  ManagedTensor<T> &pow( const ManagedTensor<U> &other ) {
    launch_kernel<U, pow_op<T, U>>( other );
    return *this;
  }

  // Unary operations
  ManagedTensor<T> &neg() {
    launch_kernel<neg_op<T>>();
    return *this;
  }
  ManagedTensor<T> &recip() {
    launch_kernel<recip_op<T>>();
    return *this;
  }
  ManagedTensor<T> &exp() {
    launch_kernel<exp_op<T>>();
    return *this;
  }
  ManagedTensor<T> &fabs() {
    launch_kernel<fabs_op<T>>();
    return *this;
  }
  ManagedTensor<T> &log() {
    launch_kernel<log_op<T>>();
    return *this;
  }
  ManagedTensor<T> &log10() {
    launch_kernel<log10_op<T>>();
    return *this;
  }
  ManagedTensor<T> &sqrt() {
    launch_kernel<sqrt_op<T>>();
    return *this;
  }

private:
  void init_layout() {
    assert( dims_.size() > 0 );
    assert( lines_.size() > 0 );
    assert( lines_[dims_.size() - 1] == 1 );
    assert( nelem_ == 1 );
    if( dims_.size() > 1 ) {
      for( int i = dims_.size() - 2; i >= 0; --i ) {
        lines_[i] = dims_[i + 1] * lines_[i + 1];
      }
    }
    for( uint64_t dim_size : dims_ ) {
      nelem_ *= dim_size;
    }
    data_size_ = nelem_ * sizeof( T );
    cuda_check( cudaGetDevice( &device_ ) );
    block_dims_ = THREADS_PER_BLOCK;
    grid_dims_ = ( nelem_ + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
  }

  std::int64_t get_flat_index( TensorIndex index ) const {
    if( index.size() != dims_.size() ) {
      std::ostringstream ss;
      ss << "Index has wrond number of dimentions: " << index.size() << " != " << dims_.size();
      throw std::runtime_error( ss.str() );
    }

    std::int64_t flat_idx = 0;
    for( int i = 0; i < index.size(); ++i ) {
      if( index[i] < dims_[i] ) {
        flat_idx += index[i] * lines_[i];
      } else {
        std::ostringstream ss;
        ss << "Index for dimention " << i << " bigger than dimention size: " << dims_[i];
        throw std::runtime_error( ss.str() );
      }
    }
    return flat_idx;
  }

  template <random_generator<T> gen>
  void launch_kernel() {
    random_fill_kernel<T, gen><<<grid_dims_, block_dims_, 0, stream_>>>( data_, nelem_, 123 );
    cuda_check();
    sync();
  }

  template <scalar_op<T> op>
  void launch_kernel( T value ) {
    scalar_op_kernel<T, op><<<grid_dims_, block_dims_, 0, stream_>>>( data_, value, nelem_ );
    cuda_check();
  }

  template <typename U, binary_op<T, U> op>
  void launch_kernel( const ManagedTensor<U> &other ) {
    if( nelem_ != other.size() ) {
      std::ostringstream ss;
      ss << "Tensors have different number of elements: " << nelem_ << " != " << other.size();
      throw std::runtime_error( ss.str() );
    }
    if( other.get_stream() != stream_ ) {
      other.sync();
    }
    binary_op_kernel<T, U, op><<<grid_dims_, block_dims_, 0, stream_>>>( data_, other.data(), nelem_ );
    cuda_check();
  }

  template <unary_op<T> op>
  void launch_kernel() {
    unary_op_kernel<T, op><<<grid_dims_, block_dims_, 0, stream_>>>( data_, nelem_ );
    cuda_check();
  }

  TensorDims dims_{};
  TensorDims lines_{};
  int64_t nelem_{0};
  int64_t data_size_{0};
  T *data_{nullptr};
  dim3 grid_dims_;
  dim3 block_dims_;
  mutable cudaStream_t stream_{0};
  mutable int device_{0};
};
} // namespace tgl
#endif /* TGL_MANAGED_TENSOR_H */
