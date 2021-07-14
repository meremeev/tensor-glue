#include <catch2/catch_all.hpp>
#include <iostream>

#include <managed_tensor.cuh>

using namespace tgl;


TEST_CASE( "Create tensor" ) {
    ManagedTensor<float> tensor( { 2, 3, 3, 4 }, true);
    REQUIRE(tensor.ndims() == 4);
    REQUIRE(tensor.size(1) == 3);
    REQUIRE(tensor.size(3) == 4);
    REQUIRE(tensor.size() == 72);

    float *p = tensor.data();
    for (uint64_t i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == 0.0);
    }
}

TEST_CASE( "Create initialized tensor" ) {
    float data[] { 1, 2, 3, 4, 5, 6, 7, 8 };
    ManagedTensor<float> tensor( { 2, 4 }, data);

    float *p = tensor.data();
    for (uint64_t i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(i+1));
    }
}

TEST_CASE( "Fill tensor" ) {
    ManagedTensor<float> tensor( { 200, 300 });
    tensor.fill(3.14);
    tensor.sync();

    float *p = tensor.data();
    for (int i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(3.14));
    }
}

TEST_CASE( "Fill zeros in tensor" ) {
    float data[] {1.2, 0.0, 1.2, 0.0, 1.2, 1.2, 0.0, 1.2};
    ManagedTensor<float> tensor( { 2, 4 }, data);
    tensor.fill_if_zero(1.2);
    tensor.sync();

    float *p = tensor.data();
    for (uint64_t i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(1.2));
    }
}

TEST_CASE( "Copy tensor" ) {
    ManagedTensor<float> tensor( { 20, 30 });
    tensor.fill(3.14);
    ManagedTensor<float> tensor2(tensor);

    float *p = tensor.data();
    for (uint64_t i = 0; i < tensor2.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(3.14));
    }
}

TEST_CASE( "Add value to tensor" ) {
    ManagedTensor<float> tensor( { 200, 300 });
    tensor.fill(2.0);
    tensor.add(3L);
    tensor.add(1.0F);
    tensor.add(4.0L);
    tensor.sync();

    float *p = tensor.data();
    for (int i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(10.0));
    }
}

TEST_CASE( "Add tensors (same type)" ) {
    ManagedTensor<float> tensor1( { 200, 300 });
    tensor1.fill(2.0);
    ManagedTensor<float> tensor2( { 200, 300 });
    tensor2.fill(8.0);
    tensor1.add(tensor2);
    tensor1.sync();

    float *p = tensor1.data();
    for (int i = 0; i < tensor1.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(10.0));
    }
}

TEST_CASE( "Add tensors (different types)" ) {
    ManagedTensor<float> tensor1( { 20, 30 });
    tensor1.fill(20.0);
    ManagedTensor<double> tensor2( { 20, 30 });
    tensor2.fill(-8.0);
    ManagedTensor<std::int64_t> tensor3( { 20, 30 });
    tensor3.fill(3);
    ManagedTensor<std::int8_t> tensor4( { 20, 30 });
    tensor4.fill(-5);
    tensor1.add(tensor2);
    tensor1.add(tensor3);
    tensor1.add(tensor4);
    tensor1.sync();

    float *p = tensor1.data();
    for (int i = 0; i < tensor1.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(10.0));
    }
}

TEST_CASE( "Multiply tensor by value" ) {
    ManagedTensor<float> tensor( { 200, 300 });
    tensor.fill(2.0);
    tensor.mult(5L);
    tensor.mult(0.5F);
    tensor.mult(2.0L);
    tensor.sync();

    float *p = tensor.data();
    for (int i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(10.0));
    }
}

TEST_CASE( "Multiply tensors (same type)" ) {
    ManagedTensor<float> tensor1( { 20, 30 });
    tensor1.fill(2.0);
    ManagedTensor<float> tensor2( { 20, 30 });
    tensor2.fill(5.0);
    tensor1.mult(tensor2);
    tensor1.sync();

    float *p = tensor1.data();
    for (int i = 0; i < tensor1.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(10.0));
    }
}

TEST_CASE( "Negating tensor" ) {
    ManagedTensor<float> tensor( { 2, 3 });
    tensor.fill(2.0);
    tensor.neg();
    tensor.sync();

    float *p = tensor.data();
    for (int i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(-2.0));
    }
}

TEST_CASE( "Reciprocate tensor" ) {
    ManagedTensor<float> tensor( { 2, 3 });
    tensor.fill(2.0);
    tensor.recip();
    tensor.sync();

    float *p = tensor.data();
    for (int i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(0.5));
    }
}

