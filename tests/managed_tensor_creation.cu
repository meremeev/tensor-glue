#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>

#include <managed_tensor.cuh>

using namespace tgl;

TEST_CASE( "Create tensor with zero initialization" ) {
  ManagedTensor<float> tensor( {2, 3, 3, 4}, true );
  REQUIRE( tensor.ndims() == 4 );
  REQUIRE( tensor.size( 1 ) == 3 );
  REQUIRE( tensor.size( 3 ) == 4 );
  REQUIRE( tensor.size() == 72 );

  float *p = tensor.data();
  for( uint64_t i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == 0.0 );
  }
}

TEST_CASE( "Create int8_t tensor from raw data" ) {
  int8_t data[]{1, 2, 3, 4, 5, 6, 7, 8};
  ManagedTensor<int8_t> tensor( {2, 4}, data );

  auto *p = tensor.data();
  for( uint64_t i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == ( i + 1 ) );
  }
}

TEST_CASE( "Create float tensor from raw data" ) {
  float data[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  ManagedTensor<float> tensor( {2, 4}, data );

  float *p = tensor.data();
  for( uint64_t i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( i + 1 ) );
  }
}

TEST_CASE( "Copy tensor" ) {
  ManagedTensor<float> tensor1( {20, 30} );
  tensor1.fill( 3.14 );
  ManagedTensor<float> tensor2( tensor1 );

  float *p = tensor2.data();
  for( uint64_t i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 3.14 ) );
  }
}

TEST_CASE( "Move tensor" ) {
  ManagedTensor<float> tensor1( {20, 30} );
  tensor1.fill( 3.14 );

  ManagedTensor<float> tensor2( std::move( tensor1 ) );

  float *p = tensor2.data();
  for( uint64_t i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 3.14 ) );
  }

  REQUIRE( tensor1.ndims() == 0 );
  REQUIRE( tensor1.size() == 0 );
  REQUIRE( tensor1.data() == nullptr );
}

TEST_CASE( "Fill tensor with value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 3.14 );
  tensor.sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 3.14 ) );
  }
}

TEST_CASE( "Fill zeros in tensor" ) {
  float data[]{1.2, 0.0, 1.2, 0.0, 1.2, 1.2, 0.0, 1.2};
  ManagedTensor<float> tensor( {2, 4}, data );
  tensor.fill_if_zero( 1.2 );
  tensor.sync();

  float *p = tensor.data();
  for( uint64_t i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 1.2 ) );
  }
}

TEST_CASE( "Fill int8_t tensor with random values from uniform distribution" ) {
  ManagedTensor<int8_t> tensor( {20, 40} );
  REQUIRE_NOTHROW( tensor.fill_random_uniform( 123 ) );
}

TEST_CASE( "Fill int64_t tensor with random values from uniform distribution" ) {
  ManagedTensor<int64_t> tensor( {20, 40} );
  REQUIRE_NOTHROW( tensor.fill_random_uniform( 123 ) );
}

TEST_CASE( "Fill float tensor with random values from uniform distribution" ) {
  ManagedTensor<float> tensor( {20, 40} );
  tensor.fill_random_uniform( 123 );
  tensor.sync();

  auto *p = tensor.data();
  for( uint64_t i = 0; i < tensor.size(); ++i, ++p ) {
    auto val = ( 0.0 < *p ) && ( *p <= 1.0 );
    REQUIRE( val );
  }
}

TEST_CASE( "Fill double tensor with random values from uniform distribution" ) {
  ManagedTensor<double> tensor( {20, 40} );
  tensor.fill_random_uniform( 123 );
  tensor.sync();

  auto *p = tensor.data();
  for( uint64_t i = 0; i < tensor.size(); ++i, ++p ) {
    auto val = ( 0.0 < *p ) && ( *p <= 1.0 );
    REQUIRE( val );
  }
}

TEST_CASE( "Fill int8_t tensor with random values from normal distribution" ) {
  ManagedTensor<int8_t> tensor( {1, 1} );
  REQUIRE_NOTHROW( tensor.fill_random_normal( 123 ) );
}

TEST_CASE( "Fill int64_t tensor with random values from normal distribution" ) {
  ManagedTensor<int64_t> tensor( {1, 1} );
  REQUIRE_NOTHROW( tensor.fill_random_normal( 123 ) );
}

TEST_CASE( "Fill float tensor with random values from normal distribution" ) {
  ManagedTensor<float> tensor( {2, 4} );
  REQUIRE_NOTHROW( tensor.fill_random_normal( 123 ) );
}

TEST_CASE( "Fill double tensor with random values from normal distribution" ) {
  ManagedTensor<double> tensor( {2, 4} );
  REQUIRE_NOTHROW( tensor.fill_random_normal( 123 ) );
}
