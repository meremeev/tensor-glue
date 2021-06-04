#ifndef TGL_MANAGED_TENSOR_H
#define TGL_MANAGED_TENSOR_H

#include <cstdint>
#include <cuda_runtime.h>

#include "common.h"
#include "tensor.h"
#include "cuda_kernels.cuh"

namespace tgl {

template<typename T>
class ManagedTensor: public Tensor<T> {
public:
    // ToDo: provide ctor with device selection and kernel launch specification
    // ToDo: provide automatic device selection for multi-GPU platform
    ManagedTensor(const TensorDims &dims, bool set_to_zero = false)
    noexcept(false) :
            dims_(dims), nelem_(1), data_size_(0), grid_dims_(0), block_dims_(0) {

        for (uint64_t dim_size : dims_) {
            nelem_ *= dim_size;
        }
        data_size_ = nelem_ * sizeof(T);
        // ToDo: consider asynchronous allocation
        check_cuda_error(cudaMallocManaged(&data_, data_size_));
        if (set_to_zero) {
            check_cuda_error(cudaMemset(data_, 0, data_size_));
        }
        // ToDo: define based on device properties and data size
        grid_dims_ = 128;
        block_dims_ = 256;
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
    void fill(T val) override {
        fill_kernel<T> <<<grid_dims_, block_dims_, 0, stream_>>>(data_, nelem_, val);
        check_cuda_error();
    }
    void add(Tensor<float> &other) override {
        check_dims(other.dims());
        other.sync();
        launch_add_kernel<T, float>(data_, other.data(), nelem_, grid_dims_, block_dims_, stream_);
    }
    void add(Tensor<double> &other) override {
        check_dims(other.dims());
        other.sync();
        launch_add_kernel<T, double>(data_, other.data(), nelem_, grid_dims_, block_dims_, stream_);
    }

    void sync() override {
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

private:
    inline void check_dims(const TensorDims &other_dims) {
        if (dims_.size() != other_dims.size()) {
            throw tensor_size_mismatch("Tensors have different number of dimensions");
        }
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (dims_[i] != other_dims[i]) {
                throw tensor_size_mismatch("Tensors have different dimensions");
            }
        }

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