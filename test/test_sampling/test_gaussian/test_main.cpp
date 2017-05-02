/**	\file 	test_main.cpp
*	\brief	Testing Kernels in MLearnKernels.h
*/
// Test framework
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <test_common.h>

// MLearn
#include <MLearn/Core>
#include <MLearn/Sampling/GaussianSampling.h>

// Eigen 
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

// Boost includes
#include <boost/random/taus88.hpp>

TEST_CASE("Test utility functions for multivariate gaussian sampling."){	
	typedef double FT;
	using namespace MLearn;
	using namespace Eigen;
	using namespace Sampling::Gaussian;
	using namespace SamplingImpl; 
	uint dim = 3;
	uint N_samples = 10;
	MLMatrix<FT> A(dim, dim);	

	A << 4, 1 ,-1,
	     1, 2 , 1,
	    -1, 1 , 2;

	SECTION("Test transformation extraction"){
		// Preallocation
		MLMatrix<FT> transform(dim, dim);

		SECTION("Test using pure cholesky!"){
			LLT<MLMatrix<FT>> cholesky(A);
			transform = MLMatrix<FT>::Random(dim, dim);

			REQUIRE_FALSE( 
				TestUtils::diff_norm(transform*transform.transpose(), A) ==
				Approx(0).margin(TEST_FLOAT_TOLERANCE));

			transform_from_decomposition(transform, cholesky);
			
			REQUIRE( 
				TestUtils::diff_norm(transform*transform.transpose(), A) ==
				Approx(0).margin(TEST_FLOAT_TOLERANCE));
		}

		SECTION("Test using eigensolver!"){
			SelfAdjointEigenSolver<MLMatrix<FT>> eigensolver(A);
			transform = MLMatrix<FT>::Random(dim, dim);
			
			REQUIRE_FALSE( 
				TestUtils::diff_norm(transform*transform.transpose(), A) ==
				Approx(0).margin(TEST_FLOAT_TOLERANCE));

			transform_from_decomposition(transform, eigensolver);
			
			REQUIRE( 
				TestUtils::diff_norm(transform*transform.transpose(), A) ==
				Approx(0).margin(TEST_FLOAT_TOLERANCE));

		}

	}

	SECTION("Test samples transformation"){
		MLMatrix<FT> orig_samples = MLMatrix<FT>::Random(dim, N_samples);
		MLMatrix<FT> samples = orig_samples;
		MLVector<FT> mean = MLVector<FT>::Random(dim);

		MLMatrix<FT> transform(dim, dim);
		transform_from_covariance<TransformMethod::CHOL>(transform, A);

		SECTION("Test transformation using transform"){
			transform_gaussian_samples_with_transform(mean, transform, samples);

			MLMatrix<FT> reverted_samples = samples;
			reverted_samples.colwise() -= mean;
			reverted_samples = transform.inverse()*reverted_samples;

			REQUIRE(
				TestUtils::diff_norm(orig_samples, reverted_samples) ==
				Approx(0).margin(TEST_FLOAT_TOLERANCE));
		}

		SECTION("Test transformation using covariance"){
			transform_gaussian_samples_with_covariance<TransformMethod::CHOL>
				(mean, A, samples);

			MLMatrix<FT> reverted_samples = samples;
			reverted_samples.colwise() -= mean;
			reverted_samples = transform.inverse()*reverted_samples;

			REQUIRE(
				TestUtils::diff_norm(orig_samples, reverted_samples) ==
				Approx(0).margin(TEST_FLOAT_TOLERANCE));
		}
	}

	SECTION("Test samples generation ..."){
		uint N_samples = 34;
		uint dim = 5;
		SECTION("... with custom random number generator."){
			boost::random::taus88 rng(seed_from_time());
	
			MLMatrix<FT> samples = 
				sample_standard_gaussian<FT>(dim, N_samples, rng);
	
			REQUIRE(samples.rows() == dim);
			REQUIRE(samples.cols() == N_samples);
			REQUIRE(samples.norm() > 0.0);
		}
	
		SECTION("... without random number generator."){
			MLMatrix<FT> samples = 
				sample_standard_gaussian<FT>(dim, N_samples);
	
			REQUIRE(samples.rows() == dim);
			REQUIRE(samples.cols() == N_samples);
			REQUIRE(samples.norm() > 0.0);
		}
	}

}

TEST_CASE("Multivariate gaussian class test"){	
	typedef double FT;
	using namespace MLearn;
	using namespace Eigen;
	using namespace Sampling::Gaussian;

	uint N_samples = 17;
	uint dim = 3;

	MLVector<FT> mean = MLVector<FT>::Random(dim);
	MLMatrix<FT> covariance(dim, dim);

	covariance << 4, 1 ,-1,
	              1, 2 , 1,
	             -1, 1 , 2;

	SECTION("Test static sampling function"){
		MLMatrix<FT> samples = MultivariateGaussian<FT>::sample(
			mean, covariance, N_samples);

		REQUIRE(samples.cols() == N_samples);
		REQUIRE(samples.rows() == dim);
		REQUIRE(samples.norm() > 0.0);
	}

	SECTION("Test sampling using instantiated class"){
		MultivariateGaussian<FT> mg(mean, covariance);
		MLMatrix<FT> samples = mg.sample(N_samples);

		REQUIRE(samples.cols() == N_samples);
		REQUIRE(samples.rows() == dim);
		REQUIRE(samples.norm() > 0.0);
	}

	SECTION("Test class"){
		MultivariateGaussian<FT> ref_mg(mean, covariance);

		REQUIRE(TestUtils::diff_norm(mean, ref_mg.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm(covariance, ref_mg.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		MultivariateGaussian<FT> copy_mg(ref_mg);

		REQUIRE(TestUtils::diff_norm(mean, copy_mg.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm(covariance, copy_mg.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));


		MultivariateGaussian<FT> 
			move_mg(std::move(MultivariateGaussian<FT>(ref_mg)));

		REQUIRE(TestUtils::diff_norm(mean, move_mg.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm(covariance, move_mg.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		MultivariateGaussian<FT> copy_assign_mg;
		copy_assign_mg = ref_mg;
		REQUIRE(TestUtils::diff_norm(mean, copy_assign_mg.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm(covariance, copy_assign_mg.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));


		MultivariateGaussian<FT> move_assign_mg;
		move_assign_mg = std::move(MultivariateGaussian<FT>(ref_mg));
		REQUIRE(TestUtils::diff_norm(mean, move_assign_mg.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm(covariance, move_assign_mg.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));

		MLVector<FT> new_mean = MLVector<FT>::Random(dim);
		MLMatrix<FT> new_covariance = 5.0*covariance;

		ref_mg.set_mean(new_mean);
		REQUIRE(TestUtils::diff_norm(new_mean, ref_mg.mean()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		ref_mg.set_covariance(new_covariance);
		REQUIRE(TestUtils::diff_norm(new_covariance, ref_mg.covariance()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
		REQUIRE(TestUtils::diff_norm(new_covariance, 
						ref_mg.transform()*ref_mg.transform().transpose()) == 
			Approx(0).margin(TEST_FLOAT_TOLERANCE));
	}

}