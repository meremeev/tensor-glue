#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>

#include <managed_tensor.cuh>

using namespace tgl;

TEST_CASE( "Add value to tensor" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 2.0 ).add( 3L ).add( 1.0F ).add( 4.0L ).sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}
TEST_CASE( "Subtract value from tensor" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 12.0 ).sub( 5L ).sub( 0.5F ).sub( 2.0L ).sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 4.5 ) );
  }
}

TEST_CASE( "Multiply tensor by value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 2.0 ).mult( 5L ).mult( 0.5F ).mult( 2.0L ).sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Divide tensor by value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 21.0 ).div( 3L ).div( 0.5F ).div( 2.0L ).sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 7.0 ) );
  }
}

TEST_CASE( "Tensor fmod()" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 21.7 ).fmod( 15L ).fmod( 4.5F ).fmod( 2L ).sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 0.2 ) );
  }
}

TEST_CASE( "Add tensors (same type)" ) {
  ManagedTensor<float> tensor1( {200, 300} );
  tensor1.fill( 2.0 );
  ManagedTensor<float> tensor2( {200, 300} );
  tensor2.fill( 8.0 ).add( tensor1 ).sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Add tensors (different types)" ) {
  ManagedTensor<float> tensor1( {20, 30} );
  tensor1.fill( 20.0 );
  ManagedTensor<double> tensor2( {20, 30} );
  tensor2.fill( -8.0 );
  ManagedTensor<std::int64_t> tensor3( {20, 30} );
  tensor3.fill( 3 );
  ManagedTensor<std::int8_t> tensor4( {20, 30} );
  tensor4.fill( -5 );
  tensor1.add( tensor2 ).add( tensor3 ).add( tensor4 ).sync();

  float *p = tensor1.data();
  for( int i = 0; i < tensor1.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Add tensors (different streams)" ) {
  ManagedTensor<float> tensor1( {200, 300} );
  cudaStream_t stream1;
  check_cuda_error( cudaStreamCreate( &stream1 ) );
  tensor1.set_stream( stream1 );
  tensor1.fill( 2.0 );

  ManagedTensor<float> tensor2( {200, 300} );
  tensor2.fill( 8.0 ).add( tensor1 ).sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Multiply tensors (same type)" ) {
  ManagedTensor<float> tensor1( {20, 30} );
  tensor1.fill( 2.0 );
  ManagedTensor<float> tensor2( {20, 30} );
  tensor2.fill( 5.0 ).mult( tensor1 ).sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Tensor neg()" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 2.0 ).neg().sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( -2.0 ) );
  }
}

TEST_CASE( "Tensor recip()" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.fill( 2.0 ).recip().add( 2 ).sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 2.5 ) );
  }
}

TEST_CASE( "Tensor  exp()" ) {
  ManagedTensor<double> tensor( {2, 3} );
  tensor.fill( 2.0 ).exp().sync();

  double *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( exp( 2.0 ) ) );
  }
}

TEST_CASE( "Tensor fabs()" ) {
  float data[]{1, -2, -3.4, -4.6, 5.8, -6., -7.2, -8};
  ManagedTensor<float> tensor1( {2, 4}, data );
  ManagedTensor<float> tensor2( tensor1 );
  tensor2.fabs().sync();

  float *p1 = tensor1.data();
  float *p2 = tensor2.data();
  for( uint64_t i = 0; i < tensor1.size(); ++i, ++p1, ++p2 ) {
    REQUIRE( *p2 == Catch::Approx( fabs( *p1 ) ) );
  }
}

TEST_CASE( "Tensor log()" ) {
  float data[]{0.1, 0.2, 3.4, 4.6, 5.8, 6., 7.2, 8};
  ManagedTensor<float> tensor1( {2, 4}, data );
  ManagedTensor<float> tensor2( tensor1 );
  tensor2.log().sync();

  float *p1 = tensor1.data();
  float *p2 = tensor2.data();
  for( uint64_t i = 0; i < tensor1.size(); ++i, ++p1, ++p2 ) {
    REQUIRE( *p2 == Catch::Approx( log( *p1 ) ) );
  }
}

TEST_CASE( "Tensor log10()" ) {
  float data[]{0.1, 0.2, 3.4, 4.6, 5.8, 6., 7.2, 8};
  ManagedTensor<float> tensor1( {2, 4}, data );
  ManagedTensor<float> tensor2( tensor1 );
  tensor2.log10().sync();

  float *p1 = tensor1.data();
  float *p2 = tensor2.data();
  for( uint64_t i = 0; i < tensor1.size(); ++i, ++p1, ++p2 ) {
    REQUIRE( *p2 == Catch::Approx( log10( *p1 ) ) );
  }
}

TEST_CASE( "Tensor sqrt()" ) {
  float data[]{0.1, 0.2, 3.4, 4.6, 5.8, 6., 7.2, 8};
  ManagedTensor<float> tensor1( {2, 4}, data );
  ManagedTensor<float> tensor2( tensor1 );
  tensor2.sqrt().sync();

  float *p1 = tensor1.data();
  float *p2 = tensor2.data();
  for( uint64_t i = 0; i < tensor1.size(); ++i, ++p1, ++p2 ) {
    REQUIRE( *p2 == Catch::Approx( sqrt( *p1 ) ) );
  }
}
