#ifndef TGL_TENSOR_FACTORY_H
#define TGL_TENSOR_FACTORY_H

#include <tensor.h>
#include <cstdint>

namespace tgl {

Tensor<std::int8_t>* new_int8_tensor(const TensorDims &dims, bool set_to_zero = false);

Tensor<std::int32_t>* new_int32_tensor(const TensorDims &dims, bool set_to_zero = false);

Tensor<std::int64_t>* new_int64_tensor(const TensorDims &dims, bool set_to_zero = false);

Tensor<float>* new_float_tensor(const TensorDims &dims, bool set_to_zero = false);

Tensor<double>* new_double_tensor(const TensorDims &dims, bool set_to_zero = false);

}

#endif /* TGL_TENSOR_FACTORY_H */
