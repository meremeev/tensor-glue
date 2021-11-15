# Tensor-glue library

Generic header-only tensor library, intended be use as a middle layer between GPU and CPU devices in heterogeneous applications. 

Library provides template of tensor implementation based on CUDA Unified Memory - *ManagedTensor<T>* which provide smooth synchronization across GPU and CPU domains and across multiple GPU devices. Is supports all CUDA compatible types.

ManageTensor implementation requires:
- 64-bit host application.
- GPU with Kepler architecture (compute capability 3.0) or higher. But GPU with Pascal architecture (compute capability 6.0) or higher will deliver better performance because of hardware support for virtual memory page faulting and migration.

More about CUDA Unified Memory:

[https://developer.nvidia.com/blog/unified-memory-cuda-beginners/](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

[https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/)

### Usage

```
#include <managed_tensor.cuh>

tgl::ManagedTensor<float> tensor( { 2, 3, 3, 4 }, true);
```

Library requires compilation of CUDA code as part of your application.

### Documentation

```
$ cd <repo root>
$ doxygen
```
Compiled documentation will be available in `doc` folder. Open `doc/index.html` with your browser. 