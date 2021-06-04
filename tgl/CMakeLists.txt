set(_lib_name "tensor-glue")

add_library(${_lib_name} SHARED 
	src/utils.cpp
)

target_include_directories(${_lib_name} PUBLIC include)

set_target_properties(${_lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${_lib_name} PUBLIC cudart.so )

#target_compile_options(${_lib_name} PUBLIC -v)

