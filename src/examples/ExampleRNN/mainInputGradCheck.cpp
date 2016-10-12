#include <iostream>
#include <chrono>

#include <MLearn/Core>
#include <MLearn/NeuralNets/RecurrentNets/RecurrentLayer.h>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/Optimization/GradientDescent.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>
#include <MLearn/Optimization/Momentum.h>
#include <MLearn/Optimization/AdaGrad.h>
#include <MLearn/Optimization/AdaDelta.h>
#include <MLearn/Optimization/BFGS.h>
#include "RNNCostInput.h"

int main(int argc, char* argv[]){
	srand((unsigned int) time(0));
	//
	using namespace MLearn;
	using namespace NeuralNets;
	using namespace RecurrentNets;
	using namespace Optimization;
	using namespace Utility;

	// typedefs
	typedef double float_type;
	typedef uint index_type;


	// Activation types
	constexpr ActivationType hidden_act = ActivationType::HYPER_TAN;
	constexpr ActivationType output_act = ActivationType::LINEAR;
	// Cell type
	constexpr RNNType cell_type = RNNType::LSTM;

	// Loss and regularization type
	constexpr LossType loss = LossType::L2_SQUARED;

	// Setup RNN
	uint out_size = 100u;
	uint out_steps= 3u;
	uint hid_size = 8u;
	uint in_size = 10u;
	uint in_steps = 5u;

	typedef RecurrentLayer< float_type, index_type, cell_type, hidden_act, output_act > RNNLayer;
	RNNLayer cell(in_size,hid_size,out_size);
	// -- Set some random weights
	MLVector<float_type> weights = 2.5*MLVector<float_type>::Random( cell.getNWeights() );
	MLMatrix<float_type> target_out = 0.5*MLMatrix<float_type>::Random( out_size,out_steps );
	MLVector<float_type> x0 = 0.5*MLVector<float_type>::Random( in_size*in_steps );
	MLVector<float_type> gradient( x0.size() );
	MLVector<float_type> num_gradient( x0.size() );
	cell.attachWeightsToCell(weights);

	RNNInputGradCheck<loss,RNNLayer> cost(cell);
	cost.setOutput(target_out);
	cost.setDelay(10);
	cost.setInputSize(in_size);
	cost.setNInputSamples(in_steps);

	std::cout << cost.evaluate(x0) << std::endl;

	Optimization::GradientOption<Optimization::DifferentiationMode::NUMERICAL_CENTRAL,float_type,index_type> options_numeric(5e-7);
	cost.compute_gradient<DifferentiationMode::NUMERICAL_CENTRAL>(x0,num_gradient,options_numeric);
	cost.compute_analytical_gradient(x0,gradient);
	std::cout << "Analytical gradient " << gradient.squaredNorm() << std::endl;

	for ( int i = 0;i < gradient.size(); ++i ){
		float_type grad_n = std::abs(gradient[i]);
		if ( grad_n > 1e-10 )
			std::cout << (gradient[i] - num_gradient[i])/grad_n << " === ";
		std::cout << gradient[i] << " v. " << num_gradient[i] << std::endl;
	}

 	return 0;
}