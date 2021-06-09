#include <cstdint>
#include <iostream>

#include <catch2/catch_all.hpp>
#include <tensor_factory.h>

using namespace tgl;

TEST_CASE( "Create int8 tensor" ) {
    Tensor<std::int8_t> &tensor = *new_int8_tensor( { 2, 3, 3, 4 });
    REQUIRE(tensor.ndims() == 4);
    REQUIRE(tensor.size(1) == 3);
    REQUIRE(tensor.size(3) == 4);
    REQUIRE(tensor.size() == 72);
}

TEST_CASE( "Create int32 tensor" ) {
    Tensor<std::int32_t> &tensor = *new_int32_tensor( { 2, 3, 3, 4 });
    REQUIRE(tensor.ndims() == 4);
    REQUIRE(tensor.size(1) == 3);
    REQUIRE(tensor.size(3) == 4);
    REQUIRE(tensor.size() == 72);
}

TEST_CASE( "Create int64 tensor" ) {
    Tensor<std::int64_t> &tensor = *new_int64_tensor( { 2, 3, 3, 4 });
    REQUIRE(tensor.ndims() == 4);
    REQUIRE(tensor.size(1) == 3);
    REQUIRE(tensor.size(3) == 4);
    REQUIRE(tensor.size() == 72);
}

TEST_CASE( "Create float tensor" ) {
    Tensor<float> &tensor = *new_float_tensor( { 2, 3, 3, 4 }, true);
    REQUIRE(tensor.ndims() == 4);
    REQUIRE(tensor.size(1) == 3);
    REQUIRE(tensor.size(3) == 4);
    REQUIRE(tensor.size() == 72);

    float *p = tensor.data();
    for (uint64_t i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == 0.0);
    }
}

TEST_CASE( "Create double tensor" ) {
    Tensor<double> &tensor = *new_double_tensor( { 2, 3, 3, 4 });
    REQUIRE(tensor.ndims() == 4);
    REQUIRE(tensor.size(1) == 3);
    REQUIRE(tensor.size(3) == 4);
    REQUIRE(tensor.size() == 72);
}
