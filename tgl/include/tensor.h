#ifndef TGL_TENSOR_H
#define TGL_TENSOR_H

#include <cstdint>
#include <type_traits>
#include <vector>

namespace tgl {
using TensorDims = std::vector<int64_t>;

template <typename T>
class Tensor {
  static_assert( std::is_same<T, std::int8_t>::value || std::is_same<T, std::int64_t>::value ||
                 std::is_same<T, float>::value || std::is_same<T, double>::value );

public:
  virtual ~Tensor() noexcept( false ) {
  }
  virtual int64_t ndims() const = 0;
  virtual int64_t size() const = 0;
  virtual int64_t size( int64_t i ) const = 0;
  virtual const TensorDims &dims() const = 0;
  virtual const T *data() const = 0;
  virtual T *data() = 0;
  virtual void prefetch( int device ) const = 0;
  virtual void sync() const = 0;
  virtual Tensor<T> &fill( T value ) = 0;
  virtual Tensor<T> &fill_if_zero( T value ) = 0;
  virtual Tensor<T> &fill_random_uniform( int64_t seed ) = 0;
  virtual Tensor<T> &fill_random_normal( int64_t seed ) = 0;
  virtual Tensor<T> &add( T value ) = 0;
  virtual Tensor<T> &sub( T value ) = 0;
  virtual Tensor<T> &mult( T value ) = 0;
  virtual Tensor<T> &div( T value ) = 0;
  virtual Tensor<T> &fmod( T value ) = 0;
  virtual Tensor<T> &add( Tensor<std::int8_t> &other ) = 0;
  virtual Tensor<T> &add( Tensor<std::int64_t> &other ) = 0;
  virtual Tensor<T> &add( Tensor<float> &other ) = 0;
  virtual Tensor<T> &add( Tensor<double> &other ) = 0;
  virtual Tensor<T> &mult( Tensor<std::int8_t> &other ) = 0;
  virtual Tensor<T> &mult( Tensor<std::int64_t> &other ) = 0;
  virtual Tensor<T> &mult( Tensor<float> &other ) = 0;
  virtual Tensor<T> &mult( Tensor<double> &other ) = 0;
  virtual Tensor<T> &neg() = 0;
  virtual Tensor<T> &recip() = 0;
  virtual Tensor<T> &exp() = 0;
  virtual Tensor<T> &fabs() = 0;
  virtual Tensor<T> &log() = 0;
  virtual Tensor<T> &log10() = 0;
  virtual Tensor<T> &sqrt() = 0;

protected:
  Tensor &operator=( Tensor &&other ) = delete;
  Tensor &operator=( const Tensor &other ) = delete;
};
} // namespace tgl
#endif /* TGL_TENSOR_H */
