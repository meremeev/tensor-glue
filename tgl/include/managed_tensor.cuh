#ifndef TGL_MANAGED_TENSOR_H
#define TGL_MANAGED_TENSOR_H

#include <cstdint>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

#include "common.h"
#include "tensor.h"
#include "elementwise_kernels.cuh"

namespace tgl {

template<typename T>
class ManagedTensor: public Tensor<T> {
public:
    // ToDo: provide ctor with device selection and kernel launch specification
    // ToDo: provide automatic device selection for multi-GPU platform
    ManagedTensor(const TensorDims &dims, bool set_to_zero = false) noexcept(false) :
            dims_(dims), nelem_(1), data_size_(0), grid_dims_(0), block_dims_(0) {

        for (uint64_t dim_size : dims_) {
            nelem_ *= dim_size;
        }
        data_size_ = nelem_ * sizeof(T);

        check_cuda_error(cudaMallocManaged(&data_, data_size_));
        if (set_to_zero) {
            check_cuda_error(cudaMemset(data_, 0, data_size_));
        }

        // ToDo: define based on device properties and data size
        grid_dims_ = 12;
        block_dims_ = 64;
        check_cuda_error(cudaStreamCreate(&stream_));
    }
    virtual ~ManagedTensor() noexcept(false) {
        check_cuda_error(cudaFree(data_));
        check_cuda_error(cudaStreamDestroy(stream_));
    }
    ManagedTensor(const ManagedTensor &other) = delete;
    ManagedTensor(ManagedTensor &&other) = delete;
    ManagedTensor& operator=(const ManagedTensor &other) = delete;
    ManagedTensor& operator=(ManagedTensor &&other) = delete;

    int64_t ndims() const override {
        return dims_.size();
    }
    int64_t size() const override {
        return nelem_;
    }
    int64_t size(int64_t i) const override {
        return dims_[i];
    }
    const TensorDims& dims() const override {
        return dims_;
    }
    const T* data() const override {
        return data_;
    }
    T* data() override {
        return data_;
    }
    // Scalar operations
    void fill(T value) override {
        launch_kernel<set_val_op<T>>(value);
    }
    void add(T value) override {
        launch_kernel<add_scalar_op<T>>(value);
    }
    void mult(T value) override {
        launch_kernel<mult_scalar_op<T>>(value);
    }
    // Binary operations
    void add(Tensor<std::int8_t> &other) override {
        launch_kernel<std::int8_t, add_op<T, std::int8_t>>(other);
    }
    void add(Tensor<std::int64_t> &other) override {
        launch_kernel<std::int64_t, add_op<T, std::int64_t>>(other);
    }
    void add(Tensor<float> &other) override {
        launch_kernel<float, add_op<T, float>>(other);
    }
    void add(Tensor<double> &other) override {
        launch_kernel<double, add_op<T, double>>(other);
    }
    void mult(Tensor<std::int8_t> &other) override {
        launch_kernel<std::int8_t, mult_op<T, std::int8_t>>(other);
    }
    void mult(Tensor<std::int64_t> &other) override {
        launch_kernel<std::int64_t, mult_op<T, std::int64_t>>(other);
    }
    void mult(Tensor<float> &other) override {
        launch_kernel<float, mult_op<T, float>>(other);
    }
    void mult(Tensor<double> &other) override {
        launch_kernel<double, mult_op<T, double>>(other);
    }
    // Unary operations


    void sync() override {
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

private:

    template<scalar_op<T> op>
    void launch_kernel(T value) {
        scalar_op_kernel<T, op> <<<grid_dims_, block_dims_, 0, stream_>>>(data_, value, nelem_);
        check_cuda_error();
    }

    template<typename U, binary_op<T, U> op>
    void launch_kernel(Tensor<U> &other) {
        if (nelem_ != other.size()) {
            std::ostringstream ss;
            ss << "Tensors have different number of elements: " << nelem_ << " != " << other.size();
            throw  std::runtime_error(ss.str());
        }
        other.sync();
        binary_op_kernel<T, U, op> <<<grid_dims_, block_dims_, 0, stream_>>>(data_, other.data(), nelem_);
        check_cuda_error();
    }

    TensorDims dims_;
    int64_t nelem_;
    int64_t data_size_;
    T *data_;
    dim3 grid_dims_;
    dim3 block_dims_;
    cudaStream_t stream_;
};

}
#endif /* TGL_MANAGED_TENSOR_H */
