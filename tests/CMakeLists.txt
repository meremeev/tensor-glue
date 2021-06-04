find_package(Catch2 REQUIRED)

add_executable(test_runner platform_info.cpp managed_tensor.cu)

target_link_libraries(test_runner PRIVATE cudart.so tensor-glue Catch2::Catch2WithMain)

#target_compile_options(test_runner PUBLIC -v)