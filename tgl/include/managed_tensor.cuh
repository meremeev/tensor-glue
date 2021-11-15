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

/// Type to hold tensor dimensions
using TensorDims = std::vector<int64_t>;
/// Type to hold index of tensor element
using TensorIndex = std::vector<int64_t>;
/// Type to hold range of tensor indexes
using IndexRange = std::vector<std::array<int64_t, 2>>;

/**
 * ManagedTensor class template. Class assumes full ownership over tensor memory.
 * It will be unconditionally released in destructor.
 */
template <typename T,
          typename = typename std::enable_if<( !std::is_pointer<T>::value && !std::is_const<T>::value ), T>::type>
class ManagedTensor {
public:
  // ToDo: provide ctor with device selection and kernel launch specification
  // ToDo: provide automatic device selection for multi-GPU platformEXTRACT_ALL

  /**
    ManagedTensor class constructor.
    Create a new tensor by allocating GPU managed memory. Memory optionally could be reset to zero.

    @param dims: tensor dimensions
    @param set_to_zero: if True tensor memory will be reser to zero. Default is False.
  */
  ManagedTensor( const TensorDims &dims, bool set_to_zero = false ) noexcept( false )
      : dims_( dims ), lines_( dims_.size(), 1 ), nelem_( 1 ) {
    init_layout();
    cuda_check( cudaMallocManaged( &data_, data_size_ ) );
    if( set_to_zero ) {
      cuda_check( cudaMemset( data_, 0, data_size_ ) );
    }
  }

  /**
    ManagedTensor class constructor.
    Create a new tensor with initialization from external data.
    New GPU memory will be allocated unless taking ownership over initialized memory is permitted and possible.

    @param dims: tensor dimensions
    @param take_ownership: if True constructor will attempt to take ownership over initialization memory.
    If it is not possible tgl::cuda_error will be thrown. Default is False.
  */
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

  /**
    ManagedTensor class explicit copy constructor.
    Create a new tensor by allocating a new GPU memory and initializing it from other tensor.

    Note: both tensors will share the same execution stream.

    @param other: reference on existing tensor
  */
  explicit ManagedTensor( const ManagedTensor<T> &other ) noexcept( false ) : ManagedTensor( other.dims_ ) {
    set_stream( other.get_stream() );
    other.sync();
    cuda_check( cudaMemcpy( data_, other.data_, data_size_, cudaMemcpyDefault ) );
  }

  /**
    ManagedTensor class move constructor.
    Create a new tensor taking ownership over GPU memory  of other tensor.

    @param other: pr-reference on existing tensor
  */
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

  /**
    ManagedTensor class destructor.
    Tensor memory will be released.
  */
  virtual ~ManagedTensor() noexcept( false ) {
    sync();
    if( data_ ) {
      cuda_check( cudaFree( data_ ) );
    }
  }

  ManagedTensor &operator=( const ManagedTensor &other ) = delete;
  ManagedTensor &operator=( ManagedTensor &&other ) = delete;

  /**
    Return number of tensor dimensions.
  */
  int64_t ndims() const {
    return dims_.size();
  }

  /**
    Return number of tensor elements.
  */
  int64_t size() const {
    return nelem_;
  }

  /**
    Return size of tensor dimension.
  */
  int64_t size( int64_t i ) const {
    return dims_[i];
  }

  /**
    Return tensor dimensions.
  */
  const TensorDims &dims() const {
    return dims_;
  }

  /**
    Return pointer on const tensor data in GPU managed memory.
  */
  const T *data() const {
    return data_;
  }

  /**
    Return pointer on tensor data in GPU managed memory.
  */
  T *data() {
    return data_;
  }

  /**
    Set CUDA stream to be used for all operations with this tensor.
    @param stream: created CUDA stream. Note: stream will not be destroyed in tensor destructor.
  */
  void set_stream( cudaStream_t stream ) const {
    stream_ = stream;
  }

  /**
    Return current CUDA stream of the tensor.
  */
  cudaStream_t get_stream() const {
    return stream_;
  }

  /**
    Return GPU device where tensor memory was allocated or prefetched.
  */
  int get_device() const {
    return device_;
  }

  /**
    Move tensor memory on different device. If device is invalid tgl::cuda_error will be thrown.
  */
  void prefetch( int device ) const {
    cuda_check( cudaMemPrefetchAsync( data_, data_size_, device, stream_ ) );
    device_ = device;
  }

  /**
    Block execution until all tensor operations are completed.
  */
  void sync() const {
    cuda_check( cudaStreamSynchronize( stream_ ) );
  }

  /**
    Return const tensor element corresponding to n-dimensional index.
  */
  const T &operator[]( TensorIndex index ) const {
    sync();
    return data_[get_flat_index( index )];
  }

  /**
    Return tensor element corresponding to n-dimensional index.
  */
  T &operator[]( TensorIndex index ) {
    sync();
    return data_[get_flat_index( index )];
  }

  /**
    Return const tensor element corresponding to flat offset in tensor memory.
  */
  const T &operator[]( int64_t offset ) const {
    sync();
    return data_[offset];
  }

  /**
    Return tensor element corresponding to flat offset in tensor memory.
  */
  T &operator[]( int64_t offset ) {
    sync();
    return data_[offset];
  }

  // Scalar operations

  /**
    Set all elements of tensor to given value
    @param value: elements value to set
  */
  ManagedTensor<T> &set_val( T value ) {
    launch_kernel<set_val_op<T>>( value );
    return *this;
  }

  /**
    Set all zero elements of tensor to given value
    @param value: elements value to set
  */
  ManagedTensor<T> &set_if_zero( T value ) {
    launch_kernel<set_if_zero_op<T>>( value );
    return *this;
  }

  /**
    Replace all elements of tensor with value above given value with value
    @param value: elements value threshold
  */
  ManagedTensor<T> &clip_above( T value ) {
    launch_kernel<clip_above_op<T>>( value );
    return *this;
  }

  /**
    Replace all elements of tensor with value below given value with value
    @param value: elements value threshold
  */
  ManagedTensor<T> &clip_below( T value ) {
    launch_kernel<clip_below_op<T>>( value );
    return *this;
  }

  /**
    Fill tensor with uniformly distributed values in range between 0.0 (excludes) and 1.0 (includes).
    @param seed: distribution seed
  */
  ManagedTensor<T> &fill_random_uniform( int64_t seed ) {
    launch_kernel<uniform_generator<T>>();
    return *this;
  }

  /**
    Fill tensor with normally distributed values with mean 0.0f and standard deviation 1.0f.
    @param seed: distribution seed
  */
  ManagedTensor<T> &fill_random_normal( int64_t seed ) {
    launch_kernel<normal_generator<T>>();
    return *this;
  }

  /**
    Elementwise scalar addition operation.
    @param value: value to add
  */
  ManagedTensor<T> &operator+=( T value ) {
    launch_kernel<add_scalar_op<T>>( value );
    return *this;
  };

  /**
    Elementwise scalar subtraction operation.
    @param value: value to subtract
  */
  ManagedTensor<T> &operator-=( T value ) {
    launch_kernel<sub_scalar_op<T>>( value );
    return *this;
  };

  /**
    Elementwise multiplication by scalar operation.
    @param value: value to multiply
  */
  ManagedTensor<T> &operator*=( T value ) {
    launch_kernel<mult_scalar_op<T>>( value );
    return *this;
  }

  /**
    Elementwise division by scalar operation.
    @param value: divisor value
  */
  ManagedTensor<T> &operator/=( T value ) {
    launch_kernel<div_scalar_op<T>>( value );
    return *this;
  }

  /**
    Elementwise scalar modulo operation.
    @param value: value for operation
  */
  ManagedTensor<T> &operator%=( T value ) {
    launch_kernel<fmod_scalar_op<T>>( value );
    return *this;
  }

  /**
    Elementwise scalar exponentiation operation.
    @param value: power value
  */
  ManagedTensor<T> &pow( T value ) {
    launch_kernel<pow_scalar_op<T>>( value );
    return *this;
  }

  // Binary operations

  /**
    Elementwise tensor addition operation.
    @param value: tensor to add
  */
  template <typename U>
  ManagedTensor<T> &operator+=( const ManagedTensor<U> &other ) {
    launch_kernel<U, add_op<T, U>>( other );
    return *this;
  }

  /**
    Elementwise tensor subtraction operation.
    @param value: tensor to subtract
  */
  template <typename U>
  ManagedTensor<T> &operator-=( const ManagedTensor<U> &other ) {
    launch_kernel<U, sub_op<T, U>>( other );
    return *this;
  }

  /**
    Elementwise tensor multiplication operation.
    @param value: tensor to multiply
  */
  template <typename U>
  ManagedTensor<T> &operator*=( const ManagedTensor<U> &other ) {
    launch_kernel<U, mult_op<T, U>>( other );
    return *this;
  }

  /**
    Elementwise tensor division operation.
    @param value: divisor tensor
  */
  template <typename U>
  ManagedTensor<T> &operator/=( const ManagedTensor<U> &other ) {
    launch_kernel<U, div_op<T, U>>( other );
    return *this;
  }

  /**
    Elementwise tensor modulo operation.
    @param value: modulo tensor
  */
  template <typename U>
  ManagedTensor<T> &operator%=( const ManagedTensor<U> &other ) {
    launch_kernel<U, fmod_op<T, U>>( other );
    return *this;
  }

  /**
    Elementwise tensor exponentiation operation.
    @param value: power tensor
  */
  template <typename U>
  ManagedTensor<T> &pow( const ManagedTensor<U> &other ) {
    launch_kernel<U, pow_op<T, U>>( other );
    return *this;
  }

  // Unary operations

  /**
    Replace tensor elements with it's negation (x = -x).
  */
  ManagedTensor<T> &neg() {
    launch_kernel<neg_op<T>>();
    return *this;
  }

  /**
    Replace tensor elements with it's reciprocal (x = /x).
  */
  ManagedTensor<T> &recip() {
    launch_kernel<recip_op<T>>();
    return *this;
  }

  /**
    Replace tensor elements with it's base-e exponential value.
  */
  ManagedTensor<T> &exp() {
    launch_kernel<exp_op<T>>();
    return *this;
  }

  /**
    Replace tensor elements with it's absolute value.
  */
  ManagedTensor<T> &fabs() {
    launch_kernel<fabs_op<T>>();
    return *this;
  }

  /**
    Replace tensor elements with it's base-e logarithmic value.
  */
  ManagedTensor<T> &log() {
    launch_kernel<log_op<T>>();
    return *this;
  }

  /**
    Replace tensor elements with it's base-10 logarithmic value.
  */
  ManagedTensor<T> &log10() {
    launch_kernel<log10_op<T>>();
    return *this;
  }

  /**
    Replace tensor elements with it's square root value.
  */
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
