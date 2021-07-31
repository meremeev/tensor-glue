# Tensor-glue library

Generic tensor library is intended to bridge a gap between CPU and GPU computational devices in heterogeneous applications. 

Currently the library provides only one implementation based on CUDA Unified Memory - *ManagedTensor<T>* which provide smooth synchronization across GPU and CPU domains and across multiple GPU devices.

Supported data types: *int8*, *int64*, *float*, *double*.

ManageTensor implementation requires:
- 64-bit host application.
- GPU with Kepler architecture (compute capability 3.0) or higher. But GPU with Pascal architecture (compute capability 6.0) or higher will deliver better performance because of hardware support for virtual memory page faulting and migration.

More about CUDA Unified Memory:

[https://developer.nvidia.com/blog/unified-memory-cuda-beginners/](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

[https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/)

### Usage

Library can be used as header-only template library. This option require compilation of CUDA code as part of your application, and this lets you compile with options (PTX, compute capability, etc.) specific to your application.

```
#include <managed_tensor.cuh>

ManagedTensor<float> tgl::tensor( { 2, 3, 3, 4 }, true);
```

Also library can be used as a binary library. In this case you do not need to compile a CUDA code. Just link your application with `libtensor-glue.so` and use factory functions to create a tensor. Note, in this case library should be compiled for your device architecture.

```
#include <tensor_factory.h>

Tensor<float>* tensor = tgl::new_float_tensor( { 2, 3, 3, 4 }, true);
```
