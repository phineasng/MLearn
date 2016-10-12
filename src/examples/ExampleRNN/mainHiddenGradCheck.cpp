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
#include "RNNCostHidden.h"

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
	constexpr ActivationType hidden_act = ActivationType::LOGISTIC;
	constexpr ActivationType output_act = ActivationType::LINEAR;
	// Cell type
	constexpr RNNType cell_type = RNNType::GRU;

	// Loss and regularization type
	constexpr LossType loss = LossType::L2_SQUARED;

	// Setup RNN
	uint out_size = 2u;
	uint hid_size = 8u;

	typedef RecurrentLayer< float_type, index_type, cell_type, hidden_act, output_act > RNNLayer;
	RNNLayer cell(10u,hid_size,out_size);
	// -- Set some random weights
	MLVector<float_type> weights = 2.5*MLVector<float_type>::Random( cell.getNWeights() );
	MLVector<float_type> target_out = 0.5*MLVector<float_type>::Random( out_size );
	MLVector<float_type> h0 = 0.5*MLVector<float_type>::Random( hid_size );
	MLVector<float_type> gradient( hid_size );
	MLVector<float_type> num_gradient( hid_size );
	cell.attachWeightsToCell(weights);

	RNNGradCheck<loss,RNNLayer> cost(cell);
	cost.setOutput(target_out);
	cost.setDelay(10);

	std::cout << cost.evaluate(h0) << std::endl;

	Optimization::GradientOption<Optimization::DifferentiationMode::NUMERICAL_CENTRAL,float_type,index_type> options_numeric(5e-7);
	cost.compute_gradient<DifferentiationMode::NUMERICAL_CENTRAL>(h0,num_gradient,options_numeric);
	cost.compute_analytical_gradient(h0,gradient);
	std::cout << "Analytical gradient " << gradient.squaredNorm() << std::endl;

	for ( int i = 0;i < gradient.size(); ++i ){
		float_type grad_n = std::abs(gradient[i]);
		if ( grad_n > 1e-10 )
			std::cout << (gradient[i] - num_gradient[i])/grad_n << " === ";
		std::cout << gradient[i] << " v. " << num_gradient[i] << std::endl;
	}

 	return 0;
}