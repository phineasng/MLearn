#include <iostream>
#include <chrono>

#include <MLearn/Core>
#include <MLearn/NeuralNets/RecurrentNets/RecurrentLayer.h>
#include <MLearn/NeuralNets/RecurrentNets/SimpleRNNCost.h>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/Optimization/GradientDescent.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>
#include <MLearn/Optimization/Momentum.h>
#include <MLearn/Optimization/AdaGrad.h>
#include <MLearn/Optimization/AdaDelta.h>
#include <MLearn/Optimization/BFGS.h>
#include <MLearn/Utility/DataInterface/SequentialData.h>

int main(int argc, char* argv[]){

	//
	using namespace MLearn;
	using namespace NeuralNets;
	using namespace RecurrentNets;
	using namespace Optimization;
	using namespace Utility;
	using namespace DataInterface;

	// typedefs
	typedef double float_type;
	typedef uint index_type;
	typedef SequentialData<float_type,index_type> SeqData;


	// Activation types
	constexpr ActivationType hidden_act = ActivationType::HYPER_TAN;
	constexpr ActivationType output_act = ActivationType::LINEAR;
	// Cell type
	constexpr RNNType cell_type = RNNType::LSTM;

	// Loss and regularization type
	constexpr LossType loss = LossType::SOFTMAX_CROSS_ENTROPY;

	// Setup RNN
	typedef RecurrentLayer< float_type, index_type, cell_type, hidden_act, output_act > RNNLayer;
	RNNLayer cell(10u,64u,10u);
	// -- Set some random weights
	MLVector<float_type> weights = 0.5*MLVector<float_type>::Random( cell.getNWeights() );
	cell.attachWeightsToCell(weights);

	// Generate dataset 
	int N = 1000;
	MLMatrix<float_type> inputs = MLMatrix<float_type>::Zero(10,N);
	MLMatrix<float_type> outputs = MLMatrix<float_type>::Zero(10,N);
	for ( int i = 0; i < N; ++i ){
		int o = i + 1;
		inputs( i % 10, i ) = 1;
		outputs( o % 10, i ) = 1;
	}
	MLMatrix<index_type> info(SeqData::n_info_fields,900);
	for (int i = 0; i < 900; ++i){
		info(0,i) = i;
		info(1,i) = 1;
		info(2,i) = i;
		info(3,i) = 1;
		info(4,i) = 50;
		info(5,i) = 1;
	}
	SeqData data(inputs,outputs,info);
	MLVector<float_type> analytical_grad( cell.getNWeights() );
	MLVector<float_type> numerical_grad( cell.getNWeights() );
	MLVector<float_type> grad_tmp( cell.getNWeights() );
	MLMatrix<float_type> grad_out( 10,1 );

	std::cout << data.getInput(0u) << std::endl << std::endl;
	std::cout << data.getOutput(0u) << std::endl << std::endl;

	//

	SimpleRNNCost<loss,RNNLayer,SeqData> cost(data,cell,grad_tmp,grad_out);
	/*std::cout << cost.evaluate(weights) << std::endl;
	Optimization::GradientOption<Optimization::DifferentiationMode::NUMERICAL_FORWARD,float_type,index_type> options_numeric(1e-7);
	cost.compute_analytical_gradient(weights,analytical_grad);
	std::cout << analytical_grad.norm() << std::endl;
	cost.compute_gradient<DifferentiationMode::NUMERICAL_FORWARD>(weights,numerical_grad,options_numeric);
	std::cout << numerical_grad.norm() << std::endl;

	for ( int i = 0;i < analytical_grad.size(); ++i ){
		float_type grad_n = std::abs(analytical_grad[i] + numerical_grad[i]);
		if ( grad_n > 1e-10 )
			std::cout << (analytical_grad[i] - numerical_grad[i])/grad_n << " <- ";
		std::cout << analytical_grad[i] << " v. " << numerical_grad[i] << std::endl;
	}*/

	/*cell.forwardpass_unroll(data.getInput(0),data.getDelay(0),data.getNOutputSteps(0));
	MLMatrix<float_type> hidden = cell.getAllHiddenStates();
	MLMatrix<float_type> out = cell.getAllOutputs(); 

	std::cout << hidden << std::endl << std::endl;
	std::cout << out << std::endl << std::endl;

	cell.resetHiddenState();
	MLMatrix<float_type> hidden_bis = hidden;
	MLMatrix<float_type> out_bis = out;
	hidden_bis.setZero();
	out_bis.setZero();
	//std::cout << out_bis << std::endl << std::endl;
	int out_idx = 0;
	int in_idx = 0;
	for (int i = 0; i < data.getDelay(0)+data.getNOutputSteps(0);++i){
		if ( i < data.getDelay(0) ){
			if (in_idx < data.getInput(0).cols()){
				cell.forwardpass_step(data.getInput(0).col(in_idx));
				++in_idx;
			}else{
				cell.forwardpass_step();
			}
		}
		else{
			if (in_idx < data.getInput(0).cols()){
				out_bis.col(out_idx) = cell.forwardpass_step_output(data.getInput(0).col(in_idx));
				++in_idx;
			}else{
				out_bis.col(out_idx) = cell.forwardpass_step_output();
			}
			++out_idx;
		}
		hidden_bis.block(0,(i+1)*RNNCellTraits<cell_type>::N_internal_states,hidden_bis.rows(),RNNCellTraits<cell_type>::N_internal_states) = cell.getHiddenState(); 
	}

	std::cout << hidden_bis << std::endl << std::endl;
	std::cout << out_bis << std::endl;*/
	//MLearn::Optimization::GradientDescent< MLearn::Optimization::DifferentiationMode::ANALYTICAL,MLearn::Optimization::LineSearchStrategy::FIXED,float_type,index_type,2 > minimizer;
	//MLearn::Optimization::StochasticGradientDescent< MLearn::Optimization::LineSearchStrategy::FIXED,float_type,index_type,2 > minimizer;
	//MLearn::Optimization::GradientOption< MLearn::Optimization::DifferentiationMode::ANALYTICAL, float_type > opt_2;
	//MLearn::Optimization::AdaGrad< MLearn::Optimization::LineSearchStrategy::FIXED,float_type,index_type,2 > minimizer;
	MLearn::Optimization::AdaDelta< float_type,index_type,0 > minimizer;
	minimizer.setNSamples(900);
	minimizer.setSizeBatch(20);
	MLearn::Optimization::LineSearch<MLearn::Optimization::LineSearchStrategy::FIXED,float_type,index_type> strategy(0.05);
	minimizer.setTolerance(1e-25);
	minimizer.setMaxIter(10);
	minimizer.setLineSearchMethod(strategy);
	//minimizer.setGradientOptions(opt_2);
	MLVector<float_type> out(10);
	for (int i = 0; i < 100000; ++i){
		minimizer.minimize(cost,weights);
		cell.attachWeightsToCell(weights);
		cell.resetHiddenState();
		std::cout << "Sequence ( cost = " << cost.evaluate(weights) << ") : ";
		index_type start = i % 10;
		for (index_type j = start; j < 50+start; ++j){
			//out = cell.forwardpass_step_output(inputs.col(j));
			cell.resetHiddenState();
			for ( int k = 0; k < 1; ++k )
				out = cell.forwardpass_step_output(inputs.col(j+k));
			for ( int k = 0; k < 49; ++k )
				out = cell.forwardpass_step_output();
			out = cell.forwardpass_step_output();
			index_type idx = 100;
			out.maxCoeff(&idx);
			std::cout << idx+1 << " ";
		}
		std::cout << std::endl;
	}
	
 	return 0;
}