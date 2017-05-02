/**	\file 	test_main.cpp
*	\brief	Testing Kernels in MLearnKernels.h
*/
// Test framework
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <test_common.h>

// MLearn
#include <MLearn/Core>
#include <MLearn/StochasticProcess/GaussianProcess/GP.h>
#include "test_gp_utils.h"

// Eigen 
#include <Eigen/Core>

TEST_CASE("Test computation of covariance matrix"){
	typedef double FT;
	using namespace MLearn;
	using namespace TestUtils;
	using namespace SP;
	using namespace GP;

	uint N = 5;
	uint dim = 10;

	MLMatrix<FT> pts = MLMatrix<FT>::Random(dim, N);
	MLMatrix<FT> covariance = MLMatrix<FT>::Zero(N, N);

	SECTION("Test expected result with mock kernel"){
		MockKernel K;

		compute_gp_covariance(pts, K, covariance);

		REQUIRE(diff_norm(pts.transpose()*pts, covariance) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

	}
	
	SECTION("Test compilation with a real kernel"){
		Kernel< KernelType::LINEAR > K;
		compute_gp_covariance(pts, K, covariance);
	}
}

TEST_CASE("Test gaussian process class"){
	typedef double FT;
	using namespace MLearn;
	using namespace TestUtils;
	using namespace SP;
	using namespace GP;

	uint N = 3;
	uint dim = 10;

	MLMatrix<FT> pts = MLMatrix<FT>::Random(dim, N);
	MLMatrix<FT> covariance(N, N);
	MLVector<FT> mean = MLVector<FT>::Random(N);

	SECTION("Test class constructors and assign"){
		covariance << 4, 1,-1,
					  1, 2, 1,
					 -1, 1, 2;  

		GaussianProcess<FT> ref_gp(mean, covariance);
		REQUIRE(diff_norm(mean, ref_gp.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(diff_norm(covariance, ref_gp.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		GaussianProcess<FT> copy_gp(ref_gp);
		REQUIRE(diff_norm(mean, copy_gp.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(diff_norm(covariance, copy_gp.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		GaussianProcess<FT> move_gp(std::move(copy_gp));
		REQUIRE(diff_norm(mean, move_gp.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(diff_norm(covariance, move_gp.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		GaussianProcess<FT> copy_assign_gp;
		copy_assign_gp = ref_gp;
		REQUIRE(diff_norm(mean, copy_assign_gp.mean()) == 
		 	Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(diff_norm(covariance, copy_assign_gp.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		GaussianProcess<FT> move_assign_gp;
		move_assign_gp = std::move(GaussianProcess<FT>(ref_gp));
		REQUIRE(diff_norm(mean, move_assign_gp.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(diff_norm(covariance, move_assign_gp.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

	}
	
	SECTION("Test class sampling"){
		uint N_samples = 7;

		Kernel< KernelType::LINEAR > K;
		compute_gp_covariance(pts, K, covariance);

		GaussianProcess<FT> gp(mean, covariance);

		MLMatrix<FT> samples = gp.sample(N_samples);
		REQUIRE(samples.rows() == N);
		REQUIRE(samples.cols() == N_samples);

		gp.set_covariance(MLMatrix<FT>::Zero(N, N));
		CHECK(diff_norm(gp.covariance(), covariance) > 0.0);

		gp.set_covariance(pts, K);
		REQUIRE(diff_norm(gp.covariance(), covariance) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		MLMatrix<FT> samples_static = gp.sample(mean, pts, K, N_samples);
		REQUIRE(samples_static.rows() == N);
		REQUIRE(samples_static.cols() == N_samples);

	}

	SECTION("Test confidence intervals"){
		GaussianProcess<FT> 
			gp(MLVector<FT>::Zero(N), MLMatrix<FT>::Identity(N, N));
		FT prob = 0.95;
		MLMatrix<FT> conf_interval = gp.confidence_interval(prob);

		conf_interval = conf_interval.cwiseAbs();
		conf_interval.array() -= 1.96;

		REQUIRE(diff_norm(MLMatrix<FT>::Zero(2, N), conf_interval) == 
			Approx(0).margin(1e-4));

	}
}