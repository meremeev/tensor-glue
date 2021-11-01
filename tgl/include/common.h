#ifndef TGL_COMMON_H
#define TGL_COMMON_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tgl {
class cuda_error : public std::runtime_error {
public:
  explicit cuda_error( const char *msg ) : std::runtime_error( msg ) {
  }
};

inline void cuda_check( cudaError_t err ) {
  if( err != cudaSuccess ) {
    throw tgl::cuda_error( cudaGetErrorString( cudaGetLastError() ) );
  }
}

inline void cuda_check() {
  cudaError_t err = cudaGetLastError();
  if( err != cudaSuccess ) {
    throw tgl::cuda_error( cudaGetErrorString( err ) );
  }
}
} // namespace tgl
#endif /* TGL_COMMON_H */
