#include <catch2/catch_all.hpp>
#include <cmath>
#include <iostream>

#include <managed_tensor.cuh>

using namespace tgl;

TEST_CASE( "Add value to tensor" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 2.0 );
  tensor += 3L;
  tensor += 1.0F;
  tensor += 4L;
  tensor.sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Subtract value from tensor" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 12.0 );
  tensor -= 5L;
  tensor -= 0.5F;
  tensor -= 2.0L;
  tensor.sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 4.5 ) );
  }
}

TEST_CASE( "Multiply tensor by value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 2.0 );
  tensor *= 5L;
  tensor *= 0.5F;
  tensor *= 2.0L;
  tensor.sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Divide tensor by value" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 21.0 );
  tensor /= 3L;
  tensor /= 0.5F;
  tensor /= 2.0L;
  tensor.sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 7.0 ) );
  }
}

TEST_CASE( "Tensor fmod" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 21.7 );
  tensor %= 15L;
  tensor %= 4.5F;
  tensor %= 2L;
  tensor.sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 0.2 ) );
  }
}

TEST_CASE( "Tensor pow() op" ) {
  ManagedTensor<double> tensor( {2, 3} );
  tensor.set_val( 2.0 );
  tensor.pow( 2L );
  tensor.pow( 2.0F );
  tensor.sync();

  double *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 16.0 ) );
  }
}

TEST_CASE( "Add tensors (same type)" ) {
  ManagedTensor<float> tensor1( {2, 3} );
  tensor1.set_val( 2.0 );
  ManagedTensor<float> tensor2( {2, 3} );
  tensor2.set_val( 8.0 );
  tensor2 += tensor1;
  tensor2.sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Add tensors (different types)" ) {
  ManagedTensor<float> tensor1( {2, 3} );
  tensor1.set_val( 20.0 );
  ManagedTensor<double> tensor2( {2, 3} );
  tensor2.set_val( -8.0 );
  ManagedTensor<std::int64_t> tensor3( {2, 3} );
  tensor3.set_val( 3 );
  ManagedTensor<std::int8_t> tensor4( {2, 3} );
  tensor4.set_val( -5 );
  tensor1 += tensor2;
  tensor1 += tensor3;
  tensor1 += tensor4;
  tensor1.sync();

  float *p = tensor1.data();
  for( int i = 0; i < tensor1.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Add tensors (different streams)" ) {
  ManagedTensor<float> tensor1( {2, 3} );
  cudaStream_t stream1;
  cuda_check( cudaStreamCreate( &stream1 ) );
  tensor1.set_stream( stream1 );
  tensor1.set_val( 2.0 );

  ManagedTensor<float> tensor2( {2, 3} );
  tensor2.set_val( 8.0 ) += tensor1;
  tensor2.sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Subtract tensors" ) {
  ManagedTensor<float> tensor1( {2, 3} );
  tensor1.set_val( 2.0 );

  ManagedTensor<float> tensor2( {2, 3} );
  tensor2.set_val( 8.0 ) -= tensor1;
  tensor2.sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 6.0 ) );
  }
}

TEST_CASE( "Multiply tensors (same type)" ) {
  ManagedTensor<float> tensor1( {2, 3} );
  tensor1.set_val( 2.0 );
  ManagedTensor<float> tensor2( {2, 3} );
  tensor2.set_val( 5.0 ) *= tensor1;
  tensor2.sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 10.0 ) );
  }
}

TEST_CASE( "Divide tensors" ) {
  ManagedTensor<int> tensor1( {2, 3} );
  tensor1.set_val( 2 );
  ManagedTensor<float> tensor2( {2, 3} );
  tensor2.set_val( 15.0 ) /= tensor1;
  tensor2.sync();

  float *p = tensor2.data();
  for( int i = 0; i < tensor2.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 7.5 ) );
  }
}

TEST_CASE( "Tensors fmod" ) {
  ManagedTensor<float> tensor1( {2, 3} );
  tensor1.set_val( 15.0 );
  ManagedTensor<float> tensor2( {2, 3} );
  tensor2.set_val( 20.0 ) %= tensor1;
  tensor2.sync();

  for( int i = 0; i < tensor2.size(); ++i ) {
    REQUIRE( tensor2[i] == Catch::Approx( 5.0 ) );
  }
}

TEST_CASE( "Tensors pow" ) {
  ManagedTensor<int> tensor1( {2, 3} );
  tensor1.set_val( 2 );
  ManagedTensor<float> tensor2( {2, 3} );
  tensor2.set_val( 3.0 ).pow( tensor1 );
  tensor2.sync();

  for( int i = 0; i < tensor2.size(); ++i ) {
    REQUIRE( tensor2[i] == Catch::Approx( 9.0 ) );
  }
}

TEST_CASE( "Tensor neg()" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 2.0 ).neg().sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( -2.0 ) );
  }
}

TEST_CASE( "Tensor recip()" ) {
  ManagedTensor<float> tensor( {2, 3} );
  tensor.set_val( 2.0 ).recip();
  tensor += 2;
  tensor.sync();

  float *p = tensor.data();
  for( int i = 0; i < tensor.size(); ++i, ++p ) {
    REQUIRE( *p == Catch::Approx( 2.5 ) );
  }
}

TEST_CASE( "Tensor  exp()" ) {
  ManagedTensor<double> tensor( {2, 3} );
  tensor.set_val( 2.0 ).exp().sync();

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
