/**	\file 	test_main.cpp
*	\brief	Testing Kernels in MLearnKernels.h
*/
// Test framework
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <test_common.h>

// MLearn
#include <Core>

TEST_CASE(){	
	// setup stage
	typedef double FT;
	int dim = 2;
	using namespace MLearn;
	
	MLVector<FT> x(dim), y(dim);

	x[0] = 0.29;
	x[1] = -0.37;
	y[0] = 0.46;
	y[1] = 0.12;

	SECTION("Testing the linear kernel"){
		Kernel<KernelType::LINEAR> k_linear;
		FT actual_result = k_linear.compute(x,y);
		FT expected_result = x[0]*y[0] + x[1]*y[1];
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}
}