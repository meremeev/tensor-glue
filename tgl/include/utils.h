#ifndef TGL_UTILS_H_
#define TGL_UTILS_H_

namespace tgl
{
int select_device( int device = -1 );
bool is_managed_memory( int device );
} // namespace tgl

#endif /* TGL_UTILS_H_ */
