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
#include <MLearn/NeuralNets/layers/fc_layer.h>
#include <MLearn/NeuralNets/neural_nets.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>

// STL includes 
#include <cmath>
#include <chrono>
#include <random>

using namespace MLearn;
using namespace MLearn::nn;
typedef double scalar_t;

TEST_CASE("Test Neural Nets basic utils"){	
	
	SECTION("Testing definition of layer name"){
		DEFINE_LAYER_NAME(DUMMY_LAYER);
		REQUIRE(DUMMY_LAYER::get_name() == "DUMMY_LAYER");
	}

	SECTION("Testing activation functions"){

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

TEST_CASE("Fully connected layer"){
	constexpr ActivationType type = ActivationType::SIGMOID;
	typedef Activation<type> sigmoid_t;
	srand((unsigned int) time(0));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(seed);
	std::uniform_int_distribution<int> sample(2,6);

	struct TestFCLayer: public FCLayer<scalar_t, type>{
	public:
		TestFCLayer(int output_dim, bool has_bias = true): FCLayer(output_dim, has_bias) {
			REQUIRE(_output_dim == output_dim);
			REQUIRE(_has_bias == has_bias);
		}

		void run_tests(int input_dim, int n_examples){
			test_constructors();
			test_getters();
			test_layer_core_functionalities(input_dim, n_examples);
		}

	private:
		void test_constructors() const{
			TestFCLayer copy_constructed(*this);
			REQUIRE(copy_constructed._output_dim == _output_dim);
			REQUIRE(copy_constructed._has_bias == _has_bias);

			TestFCLayer move_constructed(std::move(copy_constructed));
			REQUIRE(move_constructed._output_dim == _output_dim);
			REQUIRE(move_constructed._has_bias == _has_bias);

			REQUIRE_THROWS(FCLayer<scalar_t, type>(-1));
		}

		void test_initial_status() const{
			REQUIRE(_input_dim == -1);
			REQUIRE(_n_parameters == -1);
		}

		void test_getters() const{
			REQUIRE(_has_bias == this->has_bias());
			REQUIRE(_n_parameters == this->get_n_parameters());
			REQUIRE(_output_dim == this->get_output_dim());
		}

		void test_set_input_dim(int input_dim){
			this->test_initial_status();
			REQUIRE_THROWS(this->set_input_dim(0));
			REQUIRE_THROWS(this->set_input_dim(-1));
			this->set_input_dim(input_dim);
			REQUIRE(_n_parameters == (_input_dim + int(_has_bias))*_output_dim);
			REQUIRE_THROWS(this->set_input_dim(input_dim));
		}

		void test_layer_core_functionalities(int input_dim, int n_examples){
			REQUIRE_THROWS(this->set_weights(NULL));
			REQUIRE_THROWS(this->set_grad_weights(NULL));

			MLMatrix<scalar_t> input;
			MLMatrix<scalar_t> output_gradient;

			test_set_input_dim(input_dim);

			REQUIRE_THROWS(this->forward_pass(input));
			input.resize(_input_dim, 0);
			REQUIRE_THROWS(this->forward_pass(input));
			input = MLMatrix<scalar_t>::Random(_input_dim, n_examples);
			REQUIRE_THROWS(this->forward_pass(input));
			REQUIRE_THROWS(this->set_weights(NULL));
			REQUIRE_THROWS(this->set_grad_weights(NULL));

			std::vector<scalar_t> weights(this->get_n_parameters());
			std::vector<scalar_t> grad_weights(this->get_n_parameters());

			this->set_weights(&weights[0]);
			this->set_grad_weights(&grad_weights[0]);
			REQUIRE((_W - this->get_W()).lpNorm<Eigen::Infinity>() == Approx(0));	

			REQUIRE_THROWS(this->backpropagate(output_gradient));
			output_gradient.resize(_output_dim, 0);
			REQUIRE_THROWS(this->backpropagate(output_gradient));

			// Test forward pass
			Eigen::Map<MLMatrix<scalar_t>> W(&weights[0], _output_dim, _input_dim);
			Eigen::Map<MLVector<scalar_t>> b(NULL, 0);

			REQUIRE((_W - W).lpNorm<Eigen::Infinity>() == Approx(0));
			if (_has_bias){
				new (&b) Eigen::Map<MLVector<scalar_t>>(&weights[_input_dim*_output_dim], _output_dim);
				REQUIRE((_b - b).lpNorm<Eigen::Infinity>() == Approx(0));
				REQUIRE((_b - this->get_b()).lpNorm<Eigen::Infinity>() == Approx(0));		
			}

			MLMatrix<scalar_t> output = this->forward_pass(input);
			REQUIRE((_output - output).lpNorm<Eigen::Infinity>() == Approx(0));
			REQUIRE((_output - this->get_output()).lpNorm<Eigen::Infinity>() == Approx(0));

			MLMatrix<scalar_t> preactivation = W*input;
			if (_has_bias){
				preactivation.colwise() += b;
			}
			REQUIRE((_preactivation - preactivation).lpNorm<Eigen::Infinity>() == Approx(0));
			MLMatrix<scalar_t> expected_output(preactivation.rows(), preactivation.cols());
			for (int i = 0; i < expected_output.rows(); ++i){
				for (int j = 0; j < expected_output.cols(); ++j){
					expected_output(i,j) = sigmoid_t::compute(preactivation(i,j));
				}
			}
			REQUIRE((_output - expected_output).lpNorm<Eigen::Infinity>() == Approx(0));

			// Test backpropagation
			scalar_t h = 1e-7;
			std::vector<scalar_t> ref_weights(weights);
			MLMatrix<scalar_t> out_gradient = MLMatrix<scalar_t>::Zero(_output_dim, n_examples);
			MLMatrix<scalar_t> input_h(input);
			for (int i = 0; i < output.rows(); ++i){
				for (int j = 0; j < output.cols(); ++j){
					out_gradient.setZero();
					out_gradient(i,j) = 1;
					this->backpropagate(out_gradient, true);
					REQUIRE((_grad_input - this->get_grad_input()).lpNorm<Eigen::Infinity>() == Approx(0));
					MLMatrix<scalar_t> input_gradient(this->_grad_input);

					for (int idx = 0; idx < weights.size(); ++idx){
						weights = ref_weights;
						weights[idx] += h;

						MLMatrix<scalar_t> diff_output = this->forward_pass(input) - output;
						scalar_t numeric_grad_weights = diff_output(i, j)/h;
						REQUIRE(Approx(std::abs(grad_weights[idx] - numeric_grad_weights)).margin(h) == 0);
					}	

					weights = ref_weights;
					for (int i_input = 0; i_input < input.rows(); ++i_input){
						for (int j_input = 0; j_input < input.cols(); ++j_input){
							input_h = input;
							input_h(i_input, j_input) += h;
							MLMatrix<scalar_t> diff_output = this->forward_pass(input_h) - output;
							scalar_t grad_input = diff_output(i,j)/h;
							REQUIRE(Approx(std::abs(input_gradient(i_input,j_input) - grad_input)).margin(h) == 0);
						}
					}

				}
			}

		}
	};

	TestFCLayer test_with_bias(sample(generator), true);
	test_with_bias.run_tests(sample(generator), sample(generator));

	TestFCLayer test_no_bias(sample(generator), false);
	test_no_bias.run_tests(sample(generator), sample(generator));

}

TEST_CASE("Neural Network"){
	constexpr ActivationType type = ActivationType::SIGMOID;
	typedef double scalar_t;
	typedef FCLayer<scalar_t, type> layer_t;
	constexpr LossType loss_t = LossType::CROSS_ENTROPY;
	typedef NeuralNetwork<scalar_t, loss_t, layer_t, layer_t, layer_t> neuralnet_t;
	srand((unsigned int) time(0));

	// Inheriting from FCLayer to access its fields
	struct TestNeuralNetwork{

		static void run_test_on_network(neuralnet_t& net){
			
			REQUIRE(net.get_n_parameters() == -1);
			net.set_input_dim(10);

			REQUIRE(net.get_layer<0>().layer_t::get_n_parameters() > 0);
			REQUIRE(net.get_layer<1>().layer_t::get_n_parameters() > 0);
			REQUIRE(net.get_layer<2>().layer_t::get_n_parameters() > 0);
			REQUIRE(net.get_n_parameters() == 
					net.get_layer<0>().layer_t::get_n_parameters() + 
					net.get_layer<1>().layer_t::get_n_parameters() +
					net.get_layer<2>().layer_t::get_n_parameters());

			MLVector<scalar_t> weights = MLVector<scalar_t>::Random(net.get_n_parameters());
			MLVector<scalar_t> grad_weights(net.get_n_parameters());

			REQUIRE_THROWS(net.set_weights(NULL));
			REQUIRE_THROWS(net.set_grad_weights(NULL));

			net.set_weights(weights.data());
			net.set_grad_weights(grad_weights.data());

			MLMatrix<scalar_t> X = MLMatrix<scalar_t>::Random(10, 30);
			MLMatrix<scalar_t> Y = MLMatrix<scalar_t>::Random(net.get_layer<2>().layer_t::get_output_dim(), 30);

			neuralnet_t::NeuralNetCost cost(net, X, Y);

			MLVector<scalar_t> gradient;
			MLVector<scalar_t> gradient_numerical;
			cost.compute_gradient<Optimization::DifferentiationMode::ANALYTICAL>(weights, gradient);
			cost.compute_gradient<Optimization::DifferentiationMode::NUMERICAL_CENTRAL>(weights, gradient_numerical);
			REQUIRE((gradient - gradient_numerical).lpNorm<Eigen::Infinity>() == Approx(0).margin(1e-7));
			
			using namespace Optimization;
			LineSearch< LineSearchStrategy::FIXED,scalar_t,uint > line_search(0.015);
			Optimization::StochasticGradientDescent<LineSearchStrategy::FIXED,scalar_t,uint,3> minimizer;
			minimizer.setMaxIter(10);
			minimizer.setMaxEpoch(1);
			minimizer.setSizeBatch(3);
			minimizer.setNSamples(30);
			minimizer.setLineSearchMethod(line_search);
			minimizer.setSeed(std::chrono::system_clock::now().time_since_epoch().count());
			scalar_t error_before = net.evaluate(X, Y);
			net.fit(X, Y, minimizer);
			scalar_t error_after = net.evaluate(X, Y);
			REQUIRE(error_before > error_after);
		}

		static void run_tests(){
			int seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::mt19937_64 generator(seed);
			std::uniform_int_distribution<int> sample(2,6);

			neuralnet_t net(layer_t(sample(generator)), layer_t(sample(generator)), layer_t(sample(generator)));
			neuralnet_t net2 = make_network<scalar_t, loss_t>(layer_t(sample(generator)), layer_t(sample(generator)), layer_t(sample(generator)));
			neuralnet_t net3 = make_network<scalar_t, loss_t>(net.get_layer<0>(), net.get_layer<1>(), net.get_layer<2>());

			run_test_on_network(net);
			run_test_on_network(net2);
			run_test_on_network(net3);
		}
	};

	TestNeuralNetwork::run_tests();
}