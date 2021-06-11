set(TGL_LIB "tensor-glue")

add_library(${TGL_LIB} SHARED
	src/tensor_factory.cu
	src/utils.cpp
)

target_include_directories(${TGL_LIB} PUBLIC include)

set_target_properties(${TGL_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${TGL_LIB} PUBLIC cudart.so )

#target_compile_options(${TGL_LIB} PUBLIC -v)

