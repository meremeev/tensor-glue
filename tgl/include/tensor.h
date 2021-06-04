#ifndef TGL_TENSOR_H
#define TGL_TENSOR_H

#include <cstdint>
#include <type_traits>
#include <vector>

namespace tgl {

using TensorDims = std::vector<int64_t>;

template<typename T>
class Tensor {
    static_assert(
            std::is_same<T,std::int8_t>::value ||
            std::is_same<T,std::int32_t>::value ||
            std::is_same<T,std::int64_t>::value ||
            std::is_same<T,float>::value ||
            std::is_same<T,double>::value);
public:
    virtual ~Tensor() noexcept(false) {
    }
    virtual int64_t ndims() const = 0;
    virtual int64_t size() const = 0;
    virtual int64_t size(int64_t i) const = 0;
    virtual const TensorDims& dims() const = 0;
    virtual const T* data() const = 0;
    virtual T* data() = 0;
    virtual void fill(T val) = 0;
    virtual void add(Tensor<float> &other) = 0;
    virtual void add(Tensor<double> &other) = 0;
    //virtual void add(Tensor<double>& other) = 0;
    virtual void sync() = 0;

protected:
    Tensor() = default;
    Tensor(const Tensor &other) = delete;
    Tensor(Tensor &&other) = delete;
    Tensor& operator=(Tensor &&other) = delete;
    Tensor& operator=(const Tensor &other) = delete;

};

}
#endif /* TGL_TENSOR_H */
