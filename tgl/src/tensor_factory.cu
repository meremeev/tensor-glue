#include <tensor_factory.h>
#include <managed_tensor.cuh>

namespace tgl {

Tensor<std::int8_t>* new_int8_tensor(const TensorDims &dims, bool set_to_zero) {
    return new ManagedTensor<std::int8_t>(dims, set_to_zero);
}

Tensor<std::int64_t>* new_int64_tensor(const TensorDims &dims, bool set_to_zero) {
    return new ManagedTensor<std::int64_t>(dims, set_to_zero);
}

Tensor<float>* new_float_tensor(const TensorDims &dims, bool set_to_zero) {
    return new ManagedTensor<float>(dims, set_to_zero);
}

Tensor<double>* new_double_tensor(const TensorDims &dims, bool set_to_zero) {
    return new ManagedTensor<double>(dims, set_to_zero);
}

}

