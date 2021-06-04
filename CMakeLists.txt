cmake_minimum_required(VERSION 3.10)

project(TensorGlue LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(tgl)
add_subdirectory(tests)
