#include <iostream>

#include <MLearn/Core>
#include <MLearn/NeuralNets/FeedForwardNets/Common/FCNetsExplorer.h>
#include <MLearn/NeuralNets/FeedForwardNets/Common/FCCostFunction.h>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/Optimization/ConjugateGradientDescent.h>
#include <MLearn/Optimization/BFGS.h>
#include <MLearn/Optimization/GradientDescent.h>

int main(int argc, char* argv[]){

	//
	using namespace MLearn;
	using namespace NeuralNets;
	using namespace FeedForwardNets;
	using namespace Optimization;
	
	// Set up
	uint N_layers = 3;

	// Setting layers
	MLVector<uint> layers(N_layers);
	layers << 30,15,30;

	// Activation types
	constexpr ActivationType hidden_act = ActivationType::LOGISTIC;
	constexpr ActivationType output_act = ActivationType::LOGISTIC;

	// Loss and regularization type
	constexpr LossType loss = LossType::L2_SQUARED;
	constexpr Regularizer reg = Regularizer::L2;

	// Generate dataset 
	srand((unsigned int) time(0));
	MLMatrix<double> samples = 10*MLMatrix<double>::Random(30,20);
	srand((unsigned int) time(0));
	MLMatrix<double> outputs = MLMatrix<double>::Random(30,20);

	// Set some random weights
	srand((unsigned int) time(0));
	MLVector<double> weights = 5*MLVector<double>::Random( layers.head(N_layers-1).dot(layers.tail(N_layers-1)) + layers.tail(N_layers-1).array().sum() );
	MLVector<double> gradient_pre_allocation(weights.size());
	MLVector<double> gradient(weights.size());
	MLVector<double> gradient_numerical(weights.size());

	// Build the net explorer
	FCNetsExplorer<double,uint,hidden_act,output_act> explorer(layers);

	// Allocate some supporting memory
	MLVector<double> grad_output(30);

	// Set Regularization options
	RegularizerOptions<double> options;
	Optimization::GradientOption<Optimization::DifferentiationMode::NUMERICAL_CENTRAL,double,uint> options_numeric(1e-7);
	
	// Build cost function
	TEMPLATED_FC_NEURAL_NET_COST_CONSTRUCTION( loss,reg,layers,samples,outputs,explorer,options,grad_output,gradient_pre_allocation,cost);

	cost.compute_gradient<Optimization::DifferentiationMode::ANALYTICAL>(weights,gradient);

	// Gradient check
	cost.compute_gradient<Optimization::DifferentiationMode::NUMERICAL_CENTRAL>(weights,gradient_numerical,options_numeric);
	std::cout << "diff = " << std::endl << (gradient-gradient_numerical) << std::endl << std::endl;
	std::cout << "Norm diff = " << (gradient-gradient_numerical).norm()/gradient.norm() << ", Max diff = "<< (gradient-gradient_numerical).array().abs().maxCoeff() << std::endl;

	Optimization::GradientDescent<Optimization::DifferentiationMode::ANALYTICAL,LineSearchStrategy::BACKTRACKING,double,uint,2> minimizer;
	minimizer.minimize(cost,weights);

 	return 0;
}