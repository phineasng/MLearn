/**	\file 	test_main.cpp
*	\brief	Testing Kernels in MLearnKernels.h
*/
// Test framework
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <test_common.h>

// MLearn
#include <MLearn/Core>
#include <MLearn/Regression/GPRegressor/GPRegressor.h>

// STL includes
#include <cmath>

TEST_CASE("Gaussian Process Regressor"){	
	typedef double FT;
	using namespace MLearn;
	using namespace Regression;

	typedef Kernel<KernelType::LINEAR> K;

	GPRegressor<K, FT> regressor;

	// Data
	uint N_pts = 20;
	MLMatrix<FT> X_train = MLMatrix<FT>::Random(1, N_pts);
	MLVector<FT> y_train = X_train.transpose();
	MLMatrix<FT> X_pred = MLMatrix<FT>::Random(1, N_pts);

	// Fit
	SECTION("Test fitting ..."){
		REQUIRE_FALSE(regressor.fitted());
		regressor.fit(X_train, y_train);
		REQUIRE(regressor.fitted());
		REQUIRE_FALSE(regressor.has_posterior());

		SECTION("... and prediction ..."){
			MLVector<FT> y_pred = regressor.predict(X_pred);

			REQUIRE(regressor.has_posterior());
			REQUIRE(TestUtils::diff_norm(y_pred, X_pred.transpose())== 
				Approx(0).margin(TEST_FLOAT_TOLERANCE));

			SECTION("... sampling and confidence intervals."){
				uint N_samples = 5;
				FT prob = 0.5;
				MLMatrix<FT> samples = regressor.sample(N_samples);
				MLMatrix<FT> confidence_intervals = 
					regressor.confidence_interval(prob);

				REQUIRE(samples.rows() == N_pts);
				REQUIRE(samples.cols() == N_samples);
				REQUIRE(confidence_intervals.rows() == 2);
				REQUIRE(confidence_intervals.cols() == N_pts);
			}

			SECTION("... copy constructor."){
				GPRegressor<K, FT> copy_regressor(regressor);

				REQUIRE(regressor.fitted() == copy_regressor.fitted());
				REQUIRE(regressor.has_posterior() == 
					copy_regressor.has_posterior());
				REQUIRE(TestUtils::diff_norm(regressor.posterior_mean(), 
					copy_regressor.posterior_mean()) == 
					Approx(0).margin(TEST_FLOAT_TOLERANCE));
				REQUIRE(TestUtils::diff_norm(regressor.posterior_covariance(), 
					copy_regressor.posterior_covariance()) == 
					Approx(0).margin(TEST_FLOAT_TOLERANCE));
				REQUIRE(TestUtils::diff_norm(y_pred, 
					copy_regressor.posterior_mean()) == 
					Approx(0).margin(TEST_FLOAT_TOLERANCE));

				SECTION("... move constructor."){
					GPRegressor<K, FT> 
						move_regressor(std::move(copy_regressor));

					REQUIRE(regressor.fitted() == move_regressor.fitted());
					REQUIRE(regressor.has_posterior() == 
						move_regressor.has_posterior());
					REQUIRE(TestUtils::diff_norm(regressor.posterior_mean(), 
						move_regressor.posterior_mean()) == 
						Approx(0).margin(TEST_FLOAT_TOLERANCE));
					REQUIRE(
						TestUtils::diff_norm(regressor.posterior_covariance(), 
						move_regressor.posterior_covariance()) == 
						Approx(0).margin(TEST_FLOAT_TOLERANCE));
					REQUIRE(TestUtils::diff_norm(y_pred, 
						move_regressor.posterior_mean()) == 
						Approx(0).margin(TEST_FLOAT_TOLERANCE));
				}
			}

		}
	}

	SECTION("Test kernel-related member functions"){
		typedef Kernel<KernelType::POLYNOMIAL, FT> KPOL;
		KPOL kernel;

		uint N_kernel = 3;
		FT r_kernel = 0.1;

		kernel.get<KPOL::N_index>() = N_kernel;
		kernel.get<KPOL::r_index>() = r_kernel;

		GPRegressor<KPOL, FT> pol_regressor(kernel);

		REQUIRE(pol_regressor.kernel().get<KPOL::N_index>() == N_kernel);
		REQUIRE(pol_regressor.kernel().get<KPOL::r_index>() == 
			Approx(r_kernel).margin(TEST_FLOAT_TOLERANCE));

		pol_regressor.kernel().get<KPOL::N_index>() = 2*N_kernel;
		pol_regressor.kernel().get<KPOL::r_index>() = 0.5*r_kernel;

	}
	

	SECTION("Test fitting with noise (just correct running)"){
		REQUIRE_FALSE(regressor.fitted());
		regressor.fit(X_train, y_train, 1.0);
		REQUIRE(regressor.fitted());
	}
}