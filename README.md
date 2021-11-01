# Tensor-glue library

Generic tensor library is intended to bridge a gap between CPU and GPU computational devices in heterogeneous applications. 

Currently the library provides only one implementation based on CUDA Unified Memory - *ManagedTensor<T>* which provide smooth synchronization across GPU and CPU domains and across multiple GPU devices. This implementation is still WIP.

Supported data types: *int8*, *int64*, *float*, *double*.

ManageTensor implementation requires:
- 64-bit host application.
- GPU with Kepler architecture (compute capability 3.0) or higher. But GPU with Pascal architecture (compute capability 6.0) or higher will deliver better performance because of hardware support for virtual memory page faulting and migration.

More about CUDA Unified Memory:

[https://developer.nvidia.com/blog/unified-memory-cuda-beginners/](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

[https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/)

### Usage

Library is a header-only template library. This option require compilation of CUDA code as part of your application, and this lets you compile with options (PTX, compute capability, etc.) specific to your application.

```
#include <managed_tensor.cuh>

ManagedTensor<float> tgl::tensor( { 2, 3, 3, 4 }, true);
```
