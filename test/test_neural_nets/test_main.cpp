/**	\file 	test_main.cpp
*	\brief	Testing Kernels in MLearnKernels.h
*/
// Test framework
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <test_common.h>

// MLearn
#include <MLearn/Core>
#include <MLearn/NeuralNets/layers/base/utils.h>

// STL includes 
#include <cmath>
#include <chrono>
#include <random>


TEST_CASE("Test Neural Nets basic utils"){	
	
	SECTION("Testing definition of layer name"){
		DEFINE_LAYER_NAME(DUMMY_LAYER);
		REQUIRE(DUMMY_LAYER::get_name() == "DUMMY_LAYER");
	}

	SECTION("Testing activation functions"){
		using namespace MLearn::nn;
		typedef double scalar_t;

		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937_64 generator(seed);
		std::uniform_real_distribution<scalar_t> sample(-5.0,5.0);

		SECTION("Test sigmoid function"){
			constexpr ActivationType type = ActivationType::SIGMOID;
			typedef Activation<type> sigmoid_t;
			
			SECTION("Test range and consistency among overloaded versions"){
				ActivationParams<scalar_t, type> dummy;
				for (uint i = 0; i < 50; ++i){
					scalar_t x = sample(generator);

					// test range
					scalar_t f_x = sigmoid_t::compute(x);
					REQUIRE(f_x <= 1.0);
					REQUIRE(f_x >= 0.0);

					// test consistency 
					// -- activation with and without parameters
					scalar_t f_x_params = sigmoid_t::compute(x, dummy);
					REQUIRE(f_x == Approx(f_x_params));
					// -- derivative with and without parameters
					scalar_t der_f = sigmoid_t::compute_derivative(x);
					scalar_t der_f_params = sigmoid_t::compute_derivative(x, dummy);
					REQUIRE(der_f == Approx(der_f_params));
					// -- derivative given activated value with and without parameters
					scalar_t der_act = sigmoid_t::compute_derivative_activated(f_x);
					scalar_t der_act_params = sigmoid_t::compute_derivative_activated(f_x, dummy);
					REQUIRE(der_act == Approx(der_act_params));
					// -- derivative and derivative given activated value
					REQUIRE(der_f == Approx(der_act));
				} 
			}

			SECTION("Test simmetry"){
				scalar_t x = sample(generator);

				scalar_t f_x = sigmoid_t::compute(x);
				scalar_t f_neg_x = sigmoid_t::compute(-x);

				scalar_t der_f_x = sigmoid_t::compute_derivative(x);
				scalar_t der_f_neg_x = sigmoid_t::compute_derivative(-x);

				REQUIRE(f_x == Approx(scalar_t(1) - f_neg_x));
				REQUIRE(der_f_x == Approx(der_f_neg_x));
			}

			SECTION("Test if offset and scaling of hyperbolic tangent"){
				scalar_t x = sample(generator);

				scalar_t f_x = sigmoid_t::compute(x);
				scalar_t os_tanh_x = scalar_t(0.5) + scalar_t(0.5)*std::tanh(scalar_t(0.5)*x);

				REQUIRE(f_x == Approx(os_tanh_x));
			}

			SECTION("Test type traits"){
				REQUIRE(ActivationTraits<type>::n_parameters == 0);
				REQUIRE(ActivationTraits<type>::efficient_derivative);
			}
		}

		SECTION("Test relu function"){
			constexpr ActivationType type = ActivationType::RELU;
			typedef Activation<type> relu_t;
			
			SECTION("Test range and consistency among overloaded versions"){
				ActivationParams<scalar_t, type> dummy;

				for (uint i = 0; i < 50; ++i){
					scalar_t x = sample(generator);

					// test range
					scalar_t f_x = relu_t::compute(x);
					REQUIRE( f_x >= 0. );
					scalar_t der_f = relu_t::compute_derivative(x);
					REQUIRE(((der_f == Approx(0.)) || (der_f == Approx(1.))));

					// test consistency 
					// -- activation with and without parameters
					scalar_t f_x_params = relu_t::compute(x, dummy);
					REQUIRE(f_x == Approx(f_x_params));
					// -- derivative with and without parameters
					scalar_t der_f_params = relu_t::compute_derivative(x, dummy);
					REQUIRE(der_f == Approx(der_f_params));
				} 
			}

			SECTION("Test scale invariance"){
				scalar_t x = sample(generator);
				scalar_t a = std::abs(sample(generator));

				scalar_t f_x = relu_t::compute(x);
				scalar_t f_ax = relu_t::compute(a*x);

				REQUIRE(f_ax == Approx(f_x*a));
			}

			SECTION("Test type traits"){
				REQUIRE(ActivationTraits<type>::n_parameters == 0);
				REQUIRE_FALSE(ActivationTraits<type>::efficient_derivative);
			}
		}

		SECTION("Test tanh function"){
			constexpr ActivationType type = ActivationType::TANH;
			typedef Activation<type> tanh_t;
			
			SECTION("Test range and consistency among overloaded versions"){
				ActivationParams<scalar_t, type> dummy;
				for (uint i = 0; i < 50; ++i){
					scalar_t x = sample(generator);

					// test range
					scalar_t f_x = tanh_t::compute(x);
					REQUIRE(f_x <= 1.0);
					REQUIRE(f_x >= -1.0);

					// test consistency 
					// -- activation with and without parameters
					scalar_t f_x_params = tanh_t::compute(x, dummy);
					REQUIRE(f_x == Approx(f_x_params));
					// -- derivative with and without parameters
					scalar_t der_f = tanh_t::compute_derivative(x);
					scalar_t der_f_params = tanh_t::compute_derivative(x, dummy);
					REQUIRE(der_f == Approx(der_f_params));
					// -- derivative given activated value with and without parameters
					scalar_t der_act = tanh_t::compute_derivative_activated(f_x);
					scalar_t der_act_params = tanh_t::compute_derivative_activated(f_x, dummy);
					REQUIRE(der_act == Approx(der_act_params));
					// -- derivative and derivative given activated value
					REQUIRE(der_f == Approx(der_act));
				} 
			}

			SECTION("Test simmetry"){
				scalar_t x = sample(generator);

				scalar_t f_x = tanh_t::compute(x);
				scalar_t f_neg_x = tanh_t::compute(-x);

				scalar_t der_f_x = tanh_t::compute_derivative(x);
				scalar_t der_f_neg_x = tanh_t::compute_derivative(-x);

				REQUIRE(f_x == Approx( - f_neg_x));
				REQUIRE(der_f_x == Approx(der_f_neg_x));
			}

			SECTION("Test half argument formula"){
				scalar_t x = sample(generator);

				scalar_t f_half_x = tanh_t::compute(0.5*x);
				scalar_t exp_x = std::exp(x);
				scalar_t expected_f_half_x = (exp_x - 1.)/(exp_x + 1.);

				REQUIRE(f_half_x == Approx(expected_f_half_x));
			}

			SECTION("Test sum formula"){
				scalar_t x = sample(generator);
				scalar_t y = sample(generator);

				scalar_t f_xpy = tanh_t::compute(x + y);
				scalar_t tanh_x = std::tanh(x);
				scalar_t tanh_y = std::tanh(y);
				scalar_t expected_f_xpy = (tanh_x + tanh_y)/(1. + tanh_x*tanh_y);

				REQUIRE(f_xpy == Approx(expected_f_xpy));
			}

			SECTION("Test type traits"){
				REQUIRE(ActivationTraits<type>::n_parameters == 0);
				REQUIRE(ActivationTraits<type>::efficient_derivative);
			}
		}

		SECTION("Test atan function"){
			constexpr ActivationType type = ActivationType::ATAN;
			typedef Activation<type> atan_t;
			
			SECTION("Test range and consistency among overloaded versions"){
				ActivationParams<scalar_t, type> dummy;

				for (uint i = 0; i < 50; ++i){
					scalar_t x = sample(generator);

					// test range
					scalar_t f_x = atan_t::compute(x);
					REQUIRE( f_x >= -M_PI_2 );
					REQUIRE( f_x <= M_PI_2 );
					scalar_t der_f = atan_t::compute_derivative(x);
					REQUIRE(der_f > 0.);

					// test consistency 
					// -- activation with and without parameters
					scalar_t f_x_params = atan_t::compute(x, dummy);
					REQUIRE(f_x == Approx(f_x_params));
					// -- derivative with and without parameters
					scalar_t der_f_params = atan_t::compute_derivative(x, dummy);
					REQUIRE(der_f == Approx(der_f_params));
				} 
			}

			SECTION("Test definition"){
				scalar_t x = sample(generator);

				scalar_t f_x = atan_t::compute(x);
				scalar_t expected_x = std::tan(f_x);

				REQUIRE(x == Approx(expected_x));
			}

			SECTION("Test simmetry"){
				scalar_t x = sample(generator);

				scalar_t f_x = atan_t::compute(x);
				scalar_t f_neg_x = atan_t::compute(-x);
				REQUIRE(f_x == Approx(-f_neg_x));

				scalar_t der_x = atan_t::compute_derivative(x);
				scalar_t der_neg_x = atan_t::compute_derivative(-x);
				REQUIRE(der_x == Approx(der_neg_x));
			}

			SECTION("Test type traits"){
				REQUIRE(ActivationTraits<type>::n_parameters == 0);
				REQUIRE_FALSE(ActivationTraits<type>::efficient_derivative);
			}
		}

		SECTION("Test leaky relu function"){
			constexpr ActivationType type = ActivationType::LEAKY_RELU;
			typedef Activation<type> leaky_relu_t;
			typedef Activation<ActivationType::RELU> relu_t;
			
			SECTION("Test consistency with classic relu and among overloaded versions and range"){
				ActivationParams<scalar_t, type> params;

				for (uint i = 0; i < 50; ++i){
					scalar_t x = sample(generator);


					// test consistency with relu
					params.a = 0;

					scalar_t f_x = leaky_relu_t::compute(x);
					scalar_t der_f = leaky_relu_t::compute_derivative(x);
					scalar_t f_x_params = leaky_relu_t::compute(x, params);
					scalar_t der_f_params = leaky_relu_t::compute_derivative(x, params);
					scalar_t relu_f_x = relu_t::compute(x);
					scalar_t relu_der_f = relu_t::compute_derivative(x);
					REQUIRE(f_x == Approx(relu_f_x));
					REQUIRE(der_f == Approx(relu_der_f));
					REQUIRE(f_x == Approx(f_x_params));
					REQUIRE(der_f == Approx(der_f_params));

					params.a = sample(generator);
					
					f_x_params = leaky_relu_t::compute(x, params);
					if (x < 0.){
						REQUIRE( f_x_params == Approx(params.a*x) );
					}else{
						REQUIRE( f_x_params == Approx(x) );
					}

					der_f_params = leaky_relu_t::compute_derivative(x, params);
					REQUIRE(((der_f_params == Approx(params.a)) || (der_f_params == Approx(1.))));
					
				} 
			}

			SECTION("Test scale invariance"){
				scalar_t x = sample(generator);
				scalar_t a = std::abs(sample(generator));

				scalar_t f_x = leaky_relu_t::compute(x);
				scalar_t f_ax = leaky_relu_t::compute(a*x);

				REQUIRE(f_ax == Approx(f_x*a));
			}

			SECTION("Test type traits"){
				REQUIRE(ActivationTraits<type>::n_parameters == 1u);
				REQUIRE_FALSE(ActivationTraits<type>::efficient_derivative);
			}
		}

		SECTION("Test derivative wrapper"){

			SECTION("Test using an activation without efficient derivative"){
				constexpr ActivationType type = ActivationType::ATAN;
				typedef Activation<type> atan_t;
				ActivationDerivativeWrapper<scalar_t, type> wrapper;

				scalar_t x = sample(generator);
				scalar_t f_x = atan_t::compute(x);

				scalar_t expected_derivative = atan_t::compute_derivative(x);
				
				// Dummy random value to be super-sure that the wrapper correctly chose the compute function which uses 
				// the argument x instead of the efficient version
				scalar_t f_x_dummy = f_x + sample(generator);

				scalar_t derivative = wrapper(x, f_x_dummy);
				REQUIRE(expected_derivative == Approx(derivative));
			}

			SECTION("Test using an activation without efficient derivative"){
				constexpr ActivationType type = ActivationType::SIGMOID;
				typedef Activation<type> sigmoid_t;
				ActivationDerivativeWrapper<scalar_t, type> wrapper;

				scalar_t x = sample(generator);
				scalar_t f_x = sigmoid_t::compute(x);

				scalar_t expected_derivative = sigmoid_t::compute_derivative(x);
				
				// Dummy random value to be super-sure that the wrapper correctly chose the compute function which uses 
				// the activate value f_x instead of the basic version 
				scalar_t x_dummy = x + sample(generator);

				scalar_t derivative = wrapper(x_dummy, f_x);
				REQUIRE(expected_derivative == Approx(derivative));
			}

			SECTION("Test wrapper with parameters"){
				constexpr ActivationType type = ActivationType::LEAKY_RELU;
				typedef Activation<type> leaky_relu_t;
				ActivationParams<scalar_t, type> params;
				ActivationDerivativeWrapper<scalar_t, type> wrapper;

				for (int i = 0; i < 100; ++i){
					scalar_t x = sample(generator);
					scalar_t f_x = leaky_relu_t::compute(x);
					params.a = sample(generator);
					wrapper.set_params(params);
					scalar_t expected_derivative = leaky_relu_t::compute_derivative(x, params);
					scalar_t derivative = wrapper(x, f_x);
					REQUIRE(expected_derivative == Approx(derivative));

					ActivationParams<scalar_t, type> other_params = wrapper.get_params();
					REQUIRE(params.a == Approx(other_params.a));
				}
				
			}

		}
	}
	
}