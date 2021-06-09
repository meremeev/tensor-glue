#ifndef TGL_COMMON_H
#define TGL_COMMON_H

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace tgl {

class cuda_error: public std::runtime_error {
public:
    explicit cuda_error(const char *msg) :
            std::runtime_error(msg) {
    }
};

class tensor_size_mismatch: public std::runtime_error {
public:
    explicit tensor_size_mismatch(const std::string& msg) :
            std::runtime_error(msg) {
    }
    explicit tensor_size_mismatch(const char* msg) :
            std::runtime_error(msg) {
    }
};

inline void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        throw tgl::cuda_error(cudaGetErrorString(cudaGetLastError()));
    }
}

inline void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw tgl::cuda_error(cudaGetErrorString(err));
    }
}

}
#endif /* TGL_COMMON_H */
