/**	\file 	test_main.cpp
*	\brief	Testing Kernels in MLearnKernels.h
*/
// Test framework
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <test_common.h>

// MLearn
#include <MLearn/Core>

// STL includes
#include <cmath>

TEST_CASE("Kernel testing (MLearnKernels.h)"){	
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

	SECTION("Testing the polynomial kernel"){
		typedef Kernel<KernelType::POLYNOMIAL, FT> KERNEL_TYPE;
		KERNEL_TYPE k_polynomial;
		uint N = 15;
		FT r = 0.3;
		k_polynomial.get<KERNEL_TYPE::N_index>() = N;
		k_polynomial.get<KERNEL_TYPE::r_index>() = r;
		FT actual_result = k_polynomial.compute(x,y);
		FT base = (x[0]*y[0] + x[1]*y[1] + r);
		FT expected_result = 1;
		for (uint i = 1; i <= N; ++i){
			expected_result *= base;
		}
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the RBF kernel"){
		typedef Kernel<KernelType::RBF, FT> KERNEL_TYPE;
		KERNEL_TYPE k_rbf;
		FT sig_sq = 0.7;
		k_rbf.get<KERNEL_TYPE::sigma_sq_index>() = sig_sq;
		FT actual_result = k_rbf.compute(x,y);
		FT l2_dist = (x[0] - y[0])*(x[0] - y[0]) + (x[1] - y[1])*(x[1] - y[1]);
		FT expected_result = std::exp(-l2_dist/(2*sig_sq));
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the laplacian kernel"){
		typedef Kernel<KernelType::LAPLACIAN, FT> KERNEL_TYPE;
		KERNEL_TYPE k_laplacian;
		FT alpha = 0.05;
		k_laplacian.get<KERNEL_TYPE::alpha_index>() = alpha;
		FT actual_result = k_laplacian.compute(x,y);
		FT l2_dist = std::sqrt((x[0] - y[0])*(x[0] - y[0]) + 
			(x[1] - y[1])*(x[1] - y[1]));
		FT expected_result = std::exp(-alpha*l2_dist);
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the Abel kernel"){
		typedef Kernel<KernelType::ABEL, FT> KERNEL_TYPE;
		KERNEL_TYPE k_abel;
		FT alpha = 0.07;
		k_abel.get<KERNEL_TYPE::alpha_index>() = alpha;
		FT actual_result = k_abel.compute(x,y);
		FT l1_dist = std::abs(x[0] - y[0]) + std::abs(x[1] - y[1]);
		FT expected_result = std::exp(-alpha*l1_dist);
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the constant kernel"){
		typedef Kernel<KernelType::CONSTANT, FT> KERNEL_TYPE;
		KERNEL_TYPE k_constant;
		FT c = 3.2153;
		k_constant.get<KERNEL_TYPE::C_index>() = c;
		FT actual_result = k_constant.compute(x,y);
		FT expected_result = c;
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the min kernel"){
		typedef Kernel<KernelType::MIN> KERNEL_TYPE;
		KERNEL_TYPE k_min;
		FT actual_result = k_min.compute(x,y);
		FT expected_result = 0.5*(x.template lpNorm<1>() - 
				(x-y).template lpNorm<1>() + y.template lpNorm<1>());
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the white noise kernel"){
		typedef Kernel<KernelType::WHITE_NOISE, FT> KERNEL_TYPE;
		KERNEL_TYPE k_noise;
		FT noise = 0.39;
		k_noise.get<KERNEL_TYPE::noise_level_index>() = noise;
		FT actual_result_1 = k_noise.compute(x,y);
		FT expected_result_1 = 0.0;
		FT actual_result_2x = k_noise.compute(x,x);
		FT actual_result_2y = k_noise.compute(y,y);
		FT expected_result_2 = noise;
		REQUIRE( actual_result_1 == 
			Approx(expected_result_1).margin(TEST_FLOAT_TOLERANCE) );
		REQUIRE( actual_result_2x == 
			Approx(expected_result_2).margin(TEST_FLOAT_TOLERANCE) );
		REQUIRE( actual_result_2y == 
			Approx(expected_result_2).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the Matern32 kernel"){
		typedef Kernel<KernelType::MATERN32, FT> KERNEL_TYPE;
		KERNEL_TYPE k_matern;
		FT length = 0.4;
		k_matern.get<KERNEL_TYPE::length_scale_index>() = length;
		FT actual_result = k_matern.compute(x,y);
		FT d = std::sqrt((x[0] - y[0])*(x[0] - y[0]) + 
			(x[1] - y[1])*(x[1] - y[1]))*SQRT_3/length;
		FT expected_result = (1.0 + d)*std::exp(-d);
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the Rational Quadratic kernel"){
		typedef Kernel<KernelType::RATIONAL_QUADRATIC, FT> KERNEL_TYPE;
		KERNEL_TYPE k_rq;
		FT alpha = 0.13;
		FT length = 0.4;
		k_rq.get<KERNEL_TYPE::alpha_index>() = alpha;
		k_rq.get<KERNEL_TYPE::length_squared_index>() = length*length;
		FT actual_result = k_rq.compute(x,y);
		FT d = ((x[0] - y[0])*(x[0] - y[0]) + 
			(x[1] - y[1])*(x[1] - y[1]))*0.5/(alpha*length*length);
		FT expected_result = (1.0 + d);
		expected_result = std::pow(expected_result, -alpha);
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the periodic kernel"){
		typedef Kernel<KernelType::PERIODIC, FT> KERNEL_TYPE;
		KERNEL_TYPE k_rq;
		FT period = 0.13;
		FT length = 0.4;
		k_rq.get<KERNEL_TYPE::period_index>() = period;
		k_rq.get<KERNEL_TYPE::length_squared_index>() = length*length;
		FT actual_result = k_rq.compute(x,y);
		FT d = std::sqrt((x[0] - y[0])*(x[0] - y[0]) + 
			(x[1] - y[1])*(x[1] - y[1]));
		d *= M_PI/period;
		d = std::sin(d);
		d *= d;
		d *= -2.0/(length*length);
		FT expected_result = std::exp(d);
		REQUIRE( actual_result == 
			Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );
	}

	SECTION("Testing the composite kernels"){
		typedef Kernel<KernelType::PERIODIC, FT> K1;
		typedef Kernel<KernelType::MATERN32, FT> K2;
		typedef Kernel<KernelType::LINEAR> K3;

		// setup periodic kernel
		K1 ref_per_kernel;
		typedef CompKerIndex<0,K1::period_index> _0PeriodIDX;
		typedef CompKerIndex<0,K1::length_squared_index> _0LengthIDX;
		FT _0_period = 0.5;
		FT _0_length = 1.5;
		ref_per_kernel.get<K1::period_index>() = _0_period;
		ref_per_kernel.get<K1::length_squared_index>() = _0_length;

		// setup matern kernel
		K2 ref_mat_kernel;
		typedef CompKerIndex<1,K2::length_scale_index> _1AlphaIDX;
		FT _1_alpha = 1.7;
		ref_mat_kernel.get<K2::length_scale_index>() = _1_alpha;

		// setup linear kernel
		K3 ref_lin_kernel;


		SECTION("Sum kernel"){
			typedef Kernel<KernelType::SUM, K1, K2, K3> KERNEL_TYPE;
			KERNEL_TYPE k_sum;
			k_sum.get<_0PeriodIDX>() = _0_period;
			k_sum.get<_0LengthIDX>() = _0_length;
			k_sum.get<_1AlphaIDX>() = _1_alpha;
			FT actual_result = k_sum.compute(x, y);
			FT expected_result = ref_per_kernel.compute(x, y) + 
								 ref_mat_kernel.compute(x, y) +
								 ref_lin_kernel.compute(x, y);
			REQUIRE( actual_result == 
				Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );

		}

		SECTION("Product kernel"){
			typedef Kernel<KernelType::PRODUCT, K1, K2, K3> KERNEL_TYPE;
			KERNEL_TYPE k_prod;
			k_prod.get<_0PeriodIDX>() = _0_period;
			k_prod.get<_0LengthIDX>() = _0_length;
			k_prod.get<_1AlphaIDX>() = _1_alpha;
			FT actual_result = k_prod.compute(x, y);
			FT expected_result = ref_per_kernel.compute(x, y)* 
								 ref_mat_kernel.compute(x, y)*
								 ref_lin_kernel.compute(x, y);
			REQUIRE( actual_result == 
				Approx(expected_result).margin(TEST_FLOAT_TOLERANCE) );

		}
	}
}