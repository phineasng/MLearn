#include <MLearn/NeuralNets/EBM/RBM/RBMSampler.h>
#include <MLearn/NeuralNets/EBM/RBM/RBMCost.h>
int main(){

	using namespace MLearn;
	using namespace NeuralNets;
	using namespace RBMSupport;

	typedef double FLOAT_TYPE;
	constexpr RBMUnitType visible = RBMUnitType::BERNOULLI;
	constexpr RBMUnitType hidden = RBMUnitType::BERNOULLI;
	constexpr Regularizer reg = Regularizer::L2;
	constexpr RBMTrainingMode mode = RBMTrainingMode::CONTRASTIVE_DIVERGENCE;

	uint N_vis = 900;
	uint N_hid = 100;
	uint N_params = N_vis*N_hid + N_vis + N_hid;

	MLVector<FLOAT_TYPE> grad_tmp(N_params);
	MLVector<FLOAT_TYPE> grad(N_params);
	RegularizerOptions<FLOAT_TYPE> opt;
	MLMatrix<FLOAT_TYPE> inputs = MLMatrix<FLOAT_TYPE>::Random(N_vis,1);
	auto un = [](const FLOAT_TYPE& s){
		return FLOAT_TYPE(s>0.8);
	};
	inputs = inputs.unaryExpr(un);

	RBMSampler<FLOAT_TYPE,visible,hidden> sampler(N_vis,N_hid);
	MLVector<FLOAT_TYPE> params = 0.05*MLVector<FLOAT_TYPE>::Random(N_params);
	sampler.attachParameters(params);
	/*MLVector<FLOAT_TYPE> starting_v =MLVector<FLOAT_TYPE>::Random(100);
	std::cout << std::endl << sampler.getHiddenUnits() << std::endl;
	sampler.sampleVFromH();
	std::cout << std::endl << sampler.getVisibleUnits() << std::endl;
	sampler.sampleHFromV(starting_v);
	std::cout << std::endl << sampler.getHiddenUnits() << std::endl;
	sampler.sampleVFromH();
	std::cout << std::endl << sampler.getVisibleUnits() << std::endl;
	sampler.sampleHFromV();
	std::cout << std::endl << sampler.getHiddenUnits() << std::endl;
	sampler.sampleVFromH();
	std::cout << std::endl << sampler.getVisibleUnits() << std::endl;*/
	RBMCost< reg, RBMSampler<FLOAT_TYPE,visible,hidden>, mode > cost(sampler,inputs,opt,grad_tmp);
	std::cout << std::endl << cost.evaluate(params) << std::endl; 
	cost.compute_analytical_gradient(params,grad);
	std::cout << std::endl << grad.squaredNorm() << std::endl;

}