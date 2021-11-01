#ifndef TGL_UTILS_H_
#define TGL_UTILS_H_

namespace tgl {

inline bool is_managed_memory( int device ) {
  int attr_val;
  cuda_check( cudaDeviceGetAttribute( &attr_val, cudaDevAttrManagedMemory, device ) );
  return attr_val == 1;
}

} // namespace tgl

#endif /* TGL_UTILS_H_ */
