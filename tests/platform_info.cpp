#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

int main() {
  int device_count;
  assert( cudaGetDeviceCount( &device_count ) == cudaSuccess );

  struct cudaDeviceProp device_prop;
  for( int i = 0; i < device_count; ++i ) {
    assert( cudaGetDeviceProperties( &device_prop, i ) == cudaSuccess );
    std::cout << "Device: " << i << std::endl;
    std::cout << "    managedMemory: " << device_prop.managedMemory << std::endl;
    std::cout << "    pageableMemoryAccess: " << device_prop.pageableMemoryAccess << std::endl;
    std::cout << "    unifiedAddressing: " << device_prop.unifiedAddressing << std::endl;
  }
}
