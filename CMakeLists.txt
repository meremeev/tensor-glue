cmake_minimum_required(VERSION 3.10)

project(TensorGlue LANGUAGES CUDA CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(LIB_NAME tgl)

add_subdirectory(tgl)
add_subdirectory(tests)
