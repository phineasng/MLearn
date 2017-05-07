/**	\file 	test_main.cpp
*	\brief	Testing Kernels in MLearnKernels.h
*/
// Test framework
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <test_common.h>

// MLearn
#include <MLearn/Core>

TEST_CASE("Common functions (MLearnCommonFuncs.h)"){	
	typedef double FT;
	using namespace MLearn;
	
	uint dim = 10;

	SECTION("Test pseudoinverse"){
		MLMatrix<FT> A = MLMatrix<FT>::Random(dim, dim);
		MLMatrix<FT> pinv = pseudoinverse(A);

		// Test requirements of pseudoinverse as listed on wikipedia
		// (https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse)
		REQUIRE(TestUtils::diff_norm(A*pinv*A, A) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm(pinv*A*pinv, pinv) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm((A*pinv).transpose(), A*pinv) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm((pinv*A).transpose(), pinv*A) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
	}
}