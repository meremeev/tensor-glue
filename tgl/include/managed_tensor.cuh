#ifndef TGL_MANAGED_TENSOR_H
#define TGL_MANAGED_TENSOR_H

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <tuple>

#include "common.h"
#include "elementwise_kernels.cuh"
#include "random_kernels.cuh"
#include "tensor.h"

namespace tgl {

template <typename T>
class ManagedTensor : public Tensor<T> {
public:
  // ToDo: provide ctor with device selection and kernel launch specification
  // ToDo: provide automatic device selection for multi-GPU platform
  ManagedTensor( const TensorDims &dims, bool set_to_zero = false ) noexcept( false )
      : dims_( dims ), lines_( dims_.size(), 1 ), nelem_( 1 ) {
    init_layout();
    init_cuda_params();
    check_cuda_error( cudaMallocManaged( &data_, data_size_ ) );
    if( set_to_zero ) {
      check_cuda_error( cudaMemset( data_, 0, data_size_ ) );
    }
  }

  ManagedTensor( const TensorDims &dims, T *data, bool take_ownership = false ) noexcept( false )
      : dims_( dims ), lines_( dims_.size(), 1 ), nelem_( 1 ) {
    init_layout();
    init_cuda_params();
    try {
      std::ignore = *data;
    } catch( ... ) {
      throw std::runtime_error( "Attempt to create tgl::tensor from invalid data pointer" );
    }

    if( !take_ownership ) {
      check_cuda_error( cudaMallocManaged( &data_, data_size_ ) );
      check_cuda_error( cudaMemcpy( data_, data, data_size_, cudaMemcpyHostToDevice ) );
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
    other.sync();
    check_cuda_error( cudaMemcpy( data_, other.data_, data_size_, cudaMemcpyDefault ) );
  }

  ManagedTensor( ManagedTensor<T> &&other )
      : dims_( other.dims_ ), lines_( dims_.size(), 1 ), nelem_( 1 ), data_( other.data_ ) {
    init_layout();
    init_cuda_params();
    other.sync();
    other.dims_.clear();
    other.lines_.clear();
    other.nelem_ = 0;
    other.data_size_ = 0;
    other.data_ = nullptr;
    prefetch( device_ );
  }

  virtual ~ManagedTensor() noexcept( false ) {
    sync();
    if( data_ ) {
      check_cuda_error( cudaFree( data_ ) );
    }
    if( stream_ ) {
      check_cuda_error( cudaStreamDestroy( stream_ ) );
    }
  }

  ManagedTensor &operator=( const ManagedTensor &other ) = delete;
  ManagedTensor &operator=( ManagedTensor &&other ) = delete;

  int64_t ndims() const override {
    return dims_.size();
  }
  int64_t size() const override {
    return nelem_;
  }
  int64_t size( int64_t i ) const override {
    return dims_[i];
  }
  const TensorDims &dims() const override {
    return dims_;
  }
  const T *data() const override {
    return data_;
  }
  T *data() override {
    return data_;
  }
  void prefetch( int device ) const override {
    check_cuda_error( cudaMemPrefetchAsync( data_, data_size_, device, stream_ ) );
  }
  void sync() const override {
    check_cuda_error( cudaStreamSynchronize( stream_ ) );
  }
  const T &operator[]( TensorIndex index ) const override {
    sync();
    return data_[get_flat_index( index )];
  }
  T &operator[]( TensorIndex index ) override {
    sync();
    return data_[get_flat_index( index )];
  }
  const T &operator[]( int64_t index ) const override {
    sync();
    return data_[index];
  }
  T &operator[]( int64_t index ) override {
    sync();
    return data_[index];
  }
  // Scalar operations
  Tensor<T> &fill( T value ) override {
    launch_kernel<set_val_op<T>>( value );
    return *this;
  }
  Tensor<T> &fill_if_zero( T value ) override {
    launch_kernel<set_if_zero_op<T>>( value );
    return *this;
  }
  Tensor<T> &fill_random_uniform( int64_t seed ) override {
    launch_kernel<uniform_generator<T>>();
    return *this;
  }
  Tensor<T> &fill_random_normal( int64_t seed ) override {
    launch_kernel<normal_generator<T>>();
    return *this;
  }
  Tensor<T> &add( T value ) override {
    launch_kernel<add_scalar_op<T>>( value );
    return *this;
  }
  Tensor<T> &sub( T value ) override {
    launch_kernel<sub_scalar_op<T>>( value );
    return *this;
  }
  Tensor<T> &mult( T value ) override {
    launch_kernel<mult_scalar_op<T>>( value );
    return *this;
  }
  Tensor<T> &div( T value ) override {
    launch_kernel<div_scalar_op<T>>( value );
    return *this;
  }
  Tensor<T> &fmod( T value ) override {
    launch_kernel<fmod_scalar_op<T>>( value );
    return *this;
  }
  // Binary operations
  Tensor<T> &add( Tensor<std::int8_t> &other ) override {
    launch_kernel<std::int8_t, add_op<T, std::int8_t>>( other );
    return *this;
  }
  Tensor<T> &add( Tensor<std::int64_t> &other ) override {
    launch_kernel<std::int64_t, add_op<T, std::int64_t>>( other );
    return *this;
  }
  Tensor<T> &add( Tensor<float> &other ) override {
    launch_kernel<float, add_op<T, float>>( other );
    return *this;
  }
  Tensor<T> &add( Tensor<double> &other ) override {
    launch_kernel<double, add_op<T, double>>( other );
    return *this;
  }
  Tensor<T> &mult( Tensor<std::int8_t> &other ) override {
    launch_kernel<std::int8_t, mult_op<T, std::int8_t>>( other );
    return *this;
  }
  Tensor<T> &mult( Tensor<std::int64_t> &other ) override {
    launch_kernel<std::int64_t, mult_op<T, std::int64_t>>( other );
    return *this;
  }
  Tensor<T> &mult( Tensor<float> &other ) override {
    launch_kernel<float, mult_op<T, float>>( other );
    return *this;
  }
  Tensor<T> &mult( Tensor<double> &other ) override {
    launch_kernel<double, mult_op<T, double>>( other );
    return *this;
  }
  // Unary operations
  Tensor<T> &neg() override {
    launch_kernel<neg_op<T>>();
    return *this;
  }
  Tensor<T> &recip() override {
    launch_kernel<recip_op<T>>();
    return *this;
  }
  Tensor<T> &exp() override {
    launch_kernel<exp_op<T>>();
    return *this;
  }
  Tensor<T> &fabs() override {
    launch_kernel<fabs_op<T>>();
    return *this;
  }
  Tensor<T> &log() override {
    launch_kernel<log_op<T>>();
    return *this;
  }
  Tensor<T> &log10() override {
    launch_kernel<log10_op<T>>();
    return *this;
  }
  Tensor<T> &sqrt() override {
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
  }

  void init_cuda_params() noexcept( false ) {
    // ToDo: define based on device properties and data size
    grid_dims_ = 12;
    block_dims_ = 64;
    check_cuda_error( cudaStreamCreate( &stream_ ) );
    check_cuda_error( cudaGetDevice( &device_ ) );
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
    random_init_kernel<T, gen><<<grid_dims_, block_dims_, 0, stream_>>>( data_, nelem_, 123 );
    check_cuda_error();
    sync();
  }

  template <scalar_op<T> op>
  void launch_kernel( T value ) {
    scalar_op_kernel<T, op><<<grid_dims_, block_dims_, 0, stream_>>>( data_, value, nelem_ );
    check_cuda_error();
  }

  template <typename U, binary_op<T, U> op>
  void launch_kernel( Tensor<U> &other ) {
    if( nelem_ != other.size() ) {
      std::ostringstream ss;
      ss << "Tensors have different number of elements: " << nelem_ << " != " << other.size();
      throw std::runtime_error( ss.str() );
    }
    other.sync();
    binary_op_kernel<T, U, op><<<grid_dims_, block_dims_, 0, stream_>>>( data_, other.data(), nelem_ );
    check_cuda_error();
  }

  template <unary_op<T> op>
  void launch_kernel() {
    unary_op_kernel<T, op><<<grid_dims_, block_dims_, 0, stream_>>>( data_, nelem_ );
    check_cuda_error();
  }

  TensorDims dims_{};
  TensorDims lines_{};
  int64_t nelem_{0};
  int64_t data_size_{0};
  T *data_{nullptr};
  int device_{0};
  dim3 grid_dims_;
  dim3 block_dims_;
  cudaStream_t stream_;
};
} // namespace tgl
#endif /* TGL_MANAGED_TENSOR_H */
