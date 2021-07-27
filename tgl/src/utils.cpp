#include "utils.h"
#include "common.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace tgl {

int select_device( int device ) {
  int device_count;
  check_cuda_error( cudaGetDeviceCount( &device_count ) );
  if( device < 0 ) {
    int attr_val;
    for( device = 0; device < device_count; ++device ) {
      check_cuda_error( cudaDeviceGetAttribute( &attr_val, cudaDevAttrManagedMemory, device ) );
      if( attr_val == 1 ) {
        break;
      }
    }
    if( device == device_count ) {
      device = 0;
    }
  } else {
    if( device >= device_count ) {
      throw std::range_error( "Requested device is out of range" );
    }
  }
  cudaSetDevice( device );
  return device;
}

bool is_managed_memory( int device ) {
  int attr_val;
  check_cuda_error( cudaDeviceGetAttribute( &attr_val, cudaDevAttrManagedMemory, device ) );
  return attr_val == 1;
}
} // namespace tgl
