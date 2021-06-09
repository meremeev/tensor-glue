find_package(Catch2 REQUIRED)

add_executable(platform_info platform_info.cpp )

target_link_libraries(platform_info PRIVATE cudart.so)


set(TARGET1 "header_only_lib_test")

add_executable(${TARGET1} managed_tensor_test.cu)

target_include_directories(${TARGET1} PUBLIC ../tgl/include)

target_link_libraries(${TARGET1} PRIVATE Catch2::Catch2WithMain)


set(TARGET2 "binary_lib_test")

add_executable(${TARGET2} tensor_factory_test.cpp)

target_link_libraries(${TARGET2} PRIVATE tensor-glue Catch2::Catch2WithMain)
