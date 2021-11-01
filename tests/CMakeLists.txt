find_package(Catch2 REQUIRED)

add_executable(platform_info platform_info.cpp )

target_link_libraries(platform_info PRIVATE cudart.so)


set(TARGET1 "header_only_lib_test")

add_executable(${TARGET1} managed_tensor_creation.cu
                          managed_tensor_ops.cu
)

set_target_properties(${TARGET1} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(${TARGET1} PRIVATE Catch2::Catch2WithMain ${LIB_NAME} cudart.so)

