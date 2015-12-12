#include <iostream>

#include <MLearn/Core>
#include <MLearn/Optimization/CostFunction.h>
#include <MLearn/Optimization/GradientDescent.h>
#include <MLearn/Optimization/ConjugateGradientDescent.h>
#include <MLearn/Optimization/BFGS.h>
#include <math.h>


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
	MLearn::Optimization::GradientOption< MLearn::Optimization::DifferentiationMode::NUMERICAL_BACKWARD, double > opt;
	MLearn::Optimization::GradientOption< MLearn::Optimization::DifferentiationMode::ANALYTICAL, double > opt_2;
	MLearn::Optimization::BFGS< MLearn::Optimization::DifferentiationMode::ANALYTICAL,MLearn::Optimization::LineSearchStrategy::BACKTRACKING,double,uint,2 > minimizer;
	MLearn::Optimization::ConjugateGradientDescent< MLearn::Optimization::DifferentiationMode::ANALYTICAL,MLearn::Optimization::LineSearchStrategy::BACKTRACKING,MLearn::Optimization::ConjugateFormula::PR_WITH_RESET,double,uint,2 > minimizer2;
	MLearn::Optimization::LineSearch<MLearn::Optimization::LineSearchStrategy::BACKTRACKING,double,uint> strategy(0.7,0.5,10000);
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
	std::cout << "============MINIMIZE=============" << std::endl;
	minimizer.setTolerance(1e-25);
	minimizer.setMaxIter(100000);
	minimizer.setLineSearchMethod(strategy);
	minimizer.setGradientOptions(opt_2);
	minimizer.minimize(cost,x);
	std::cout << "Minimum in:\n" << x << std::endl;
	std::cout << "Cost = " << cost.evaluate(x) << std::endl;
	x = MLearn::MLVector<double>::Random(dimension);
	minimizer2.setTolerance(1e-25);
	minimizer2.setLineSearchMethod(strategy);
	minimizer2.minimize(cost,x);
	std::cout << "Minimum in:\n" << x << std::endl;
	std::cout << "Cost = " << cost.evaluate(x) << std::endl;
	return 0;
}