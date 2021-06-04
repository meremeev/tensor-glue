#include <catch2/catch_all.hpp>
#include <iostream>

#include <cuda_runtime.h>

#include <managed_tensor.cuh>

using namespace tgl;

TEST_CASE( "Tensor creation" ) {
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

TEST_CASE( "Fill tensor" ) {
    ManagedTensor<float> tensor( { 200, 300 });
    tensor.fill(3.14);
    tensor.sync();
    float *p = tensor.data();
    for (int i = 0; i < tensor.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(3.14));
    }
}

TEST_CASE( "Add tensors (same type)" ) {
    ManagedTensor<float> tensor1( { 20, 30 });
    tensor1.fill(2.0);
    ManagedTensor<float> tensor2( { 20, 30 });
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
    tensor1.fill(2.0);
    ManagedTensor<double> tensor2( { 20, 30 });
    tensor2.fill(8.0);
    tensor1.add(tensor2);
    tensor1.sync();

    float *p = tensor1.data();
    for (int i = 0; i < tensor1.size(); ++i, ++p) {
        REQUIRE(*p == Catch::Approx(10.0));
    }
}

