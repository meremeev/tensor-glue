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

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    REQUIRE( tensor[i] == ( i + 1 ) );
  }
}

TEST_CASE( "Create float tensor from raw data" ) {
  float data[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  ManagedTensor<float> tensor( {2, 4}, data );

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    REQUIRE( tensor[i] == Catch::Approx( i + 1 ) );
  }
}

TEST_CASE( "Create float tensor with ownership of raw data" ) {
  float *data;
  cudaMallocManaged( &data, 2 * 4 * sizeof( float ) );
  ManagedTensor<float> tensor( {2, 4}, data, true );
  tensor.fill_random_normal( 123 );
  tensor.sync();

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    REQUIRE( tensor[i] == Catch::Approx( data[i] ) );
  }
}

TEST_CASE( "Attempt to create float tensor from invalid data" ) {
  int exception_count = 0;
  try {
    ManagedTensor<float> tensor( {2, 4}, nullptr );
  } catch( ... ) {
    exception_count += 1;
  }
  try {
    ManagedTensor<float> tensor( {2, 4}, new float[8], true );
  } catch( ... ) {
    exception_count += 1;
  }
  REQUIRE( exception_count == 2 );
}

TEST_CASE( "Copy tensor" ) {
  ManagedTensor<float> tensor1( {20, 30} );
  tensor1.set_val( 3.14 );
  ManagedTensor<float> tensor2( tensor1 );

  for( uint64_t i = 0; i < tensor2.size(); ++i ) {
    REQUIRE( tensor2[i] == Catch::Approx( 3.14 ) );
  }
}

TEST_CASE( "Move tensor" ) {
  ManagedTensor<float> tensor1( {20, 30} );
  tensor1.set_val( 3.14 );

  ManagedTensor<float> tensor2( std::move( tensor1 ) );

  float *p = tensor2.data();
  for( uint64_t i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 3.14 ) );
  }

  REQUIRE( tensor1.ndims() == 0 );
  REQUIRE( tensor1.size() == 0 );
  REQUIRE( tensor1.data() == nullptr );
}

TEST_CASE( "Access to tensor elements" ) {
  float data[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const ManagedTensor<float> tensor( {2, 4}, data );

  REQUIRE( tensor[{0, 0}] == Catch::Approx( 1.0 ) );
  REQUIRE( tensor[{0, 1}] == Catch::Approx( 2.0 ) );
  REQUIRE( tensor[{0, 3}] == Catch::Approx( 4.0 ) );
  REQUIRE( tensor[{1, 0}] == Catch::Approx( 5.0 ) );
  REQUIRE( tensor[{1, 3}] == Catch::Approx( 8.0 ) );
  float x = tensor[{1, 3}];
  REQUIRE( x == Catch::Approx( 8.0 ) );
}

TEST_CASE( "Modification of tensor elements" ) {
  float data[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  ManagedTensor<float> tensor( {2, 4}, data );

  REQUIRE( tensor[{0, 0}] == Catch::Approx( 1.0 ) );
  tensor[{0, 0}] = 2.5;
  REQUIRE( tensor[{0, 0}] == Catch::Approx( 2.5 ) );

  REQUIRE( tensor[{0, 1}] == Catch::Approx( 2.0 ) );
  tensor[{0, 1}] += 4.2;
  REQUIRE( tensor[{0, 1}] == Catch::Approx( 6.2 ) );

  REQUIRE( tensor[{0, 3}] == Catch::Approx( 4.0 ) );
  tensor[{0, 3}] /= 2;
  REQUIRE( tensor[{0, 3}] == Catch::Approx( 2.0 ) );

  REQUIRE( tensor[{1, 0}] == Catch::Approx( 5.0 ) );
  tensor[{1, 0}] *= 2.0;
  REQUIRE( tensor[{1, 0}] == Catch::Approx( 10.0 ) );

  REQUIRE( tensor[{1, 3}] == Catch::Approx( 8.0 ) );
  tensor[{1, 3}] -= 3;
  REQUIRE( tensor[{1, 3}] == Catch::Approx( 5.0 ) );
}

TEST_CASE( "Set tensor with value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 3.14 );

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    REQUIRE( tensor[i] == Catch::Approx( 3.14 ) );
  }
}

TEST_CASE( "Replace zeros in tensor with value" ) {
  float data[]{1.2, 0.0, 1.2, 0.0, 1.2, 1.2, 0.0, 1.2};
  ManagedTensor<float> tensor( {2, 4}, data );
  tensor.set_if_zero( 1.2 );

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    REQUIRE( tensor[i] == Catch::Approx( 1.2 ) );
  }
}

TEST_CASE( "Clip tensor above value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 3.14 );
  tensor.clip_above( 3.1 );

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    REQUIRE( tensor[i] == Catch::Approx( 3.1 ) );
  }
}

TEST_CASE( "Clip tensor below value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( -3.14 );
  tensor.clip_below( -3.1 );

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    REQUIRE( tensor[i] == Catch::Approx( -3.1 ) );
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

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    auto val = ( 0.0 < tensor[i] ) && ( tensor[i] <= 1.0 );
    REQUIRE( val );
  }
}

TEST_CASE( "Fill double tensor with random values from uniform distribution" ) {
  ManagedTensor<double> tensor( {20, 40} );
  tensor.fill_random_uniform( 123 );

  for( uint64_t i = 0; i < tensor.size(); ++i ) {
    auto val = ( 0.0 < tensor[i] ) && ( tensor[i] <= 1.0 );
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
