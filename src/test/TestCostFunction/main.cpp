#include <iostream>

#include <MLearn/Core>
#include <MLearn/Optimization/CostFunction.h>

class SquareLoss: public MLearn::Optimization::CostFunction<SquareLoss>{
public:
	template < typename ScalarType >
	ScalarType eval(const MLearn::MLVector<ScalarType>& x) const{
		return x.squaredNorm();
	}
	template < typename ScalarType >
	void compute_analytical_gradient(const MLearn::MLVector<ScalarType>& x,MLearn::MLVector<ScalarType>& gradient) const{
		gradient = 2*x;
	}
};

int main(int argc, char* argv[]){

	SquareLoss cost;
	int dimension = 10;
	int num_tries = 8;
	MLearn::MLVector<double> x = MLearn::MLVector<double>::Random(dimension);
	MLearn::MLVector<double> gradient = MLearn::MLVector<double>::Random(dimension);
	MLearn::MLMatrix<double> numerical_gradients(dimension,num_tries);
	MLearn::Optimization::NumericalGradientOption< MLearn::Optimization::DifferentiationMode::NUMERICAL_BACKWARD, double > opt;
	opt.step_size = 1e-1;
	std::cout.precision(10);
	std::cout << "=================================" << std::endl;
	std::cout << x << std::endl;
	std::cout << "=================================" << std::endl;
	std::cout << cost.evaluate(x) << std::endl;
	std::cout << "=================================" << std::endl;
	cost.compute_gradient< MLearn::Optimization::DifferentiationMode::ANALYTICAL >(x,gradient);
	std::cout << gradient << std::endl;
	for (int i = 0; i < num_tries; ++i){
		cost.compute_gradient< MLearn::Optimization::DifferentiationMode::NUMERICAL_BACKWARD >(x,gradient,opt);
		numerical_gradients.col(i) = gradient;
		opt.step_size *= 1e-1;
	}
	std::cout << "=================================" << std::endl;
	std::cout << numerical_gradients << std::endl;
	return 0;
}