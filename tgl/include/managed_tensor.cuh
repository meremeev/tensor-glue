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

constexpr int64_t THREADS_PER_BLOCK = 256;

template <typename T, typename U>
T &operator+=( T &lhs, const U &rhs ){

};

template <typename T>
class ManagedTensor : public Tensor<T> {
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
  void set_stream( cudaStream_t stream ) override {
    stream_ = stream;
  }
  cudaStream_t get_stream() const override {
    return stream_;
  }
  int get_device() const override {
    return device_;
  }
  void prefetch( int device ) const override {
    cuda_check( cudaMemPrefetchAsync( data_, data_size_, device, stream_ ) );
    device_ = device;
  }
  void sync() const override {
    cuda_check( cudaStreamSynchronize( stream_ ) );
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
    random_init_kernel<T, gen><<<grid_dims_, block_dims_, 0, stream_>>>( data_, nelem_, 123 );
    cuda_check();
    sync();
  }

  template <scalar_op<T> op>
  void launch_kernel( T value ) {
    scalar_op_kernel<T, op><<<grid_dims_, block_dims_, 0, stream_>>>( data_, value, nelem_ );
    cuda_check();
  }

  template <typename U, binary_op<T, U> op>
  void launch_kernel( Tensor<U> &other ) {
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
