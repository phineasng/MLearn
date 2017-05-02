// gnuplot-c++ includes
#include <gnuplot-iostream.h>

// MLearn includes
#include <MLearn/Core>
#include <MLearn/StochasticProcess/GaussianProcess/GP.h>

// STL includes
#include <vector>
#include <array>
#include <string>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SVD>

// Boost includes
#include <boost/program_options.hpp>

#define N_STEPS 1000 // n_points along the time
#define N_SAMPLES 1 // n of curves to sample

int main(int argc, char** argv){
	std::srand((unsigned int) time(0));

	Gnuplot gp;

	using namespace MLearn;
	using namespace SP::GP;
	namespace po = boost::program_options;

	// Create command line options
	po::options_description 
	desc("This is a demo showing different samples from a gaussian process.");

	desc.add_options()
		("help", "Show the help")
		("kernel", po::value<int>(), "Value in [0, 10)." 
		"This value indicates which kernel will be used. Available kernels:\n"
		"LINEAR (0)\n"
		"POLYNOMIAL (1)\n"
		"RBF (2)\n"
		"LAPLACIAN (3)\n"
		"ABEL (4)\n"
		"CONSTANT (5)\n"
		"MIN (6)\n"
		"MATERN_32 (7)\n"
		"MATERN_52 (8)\n"
		"RATIONAL_QUADRATIC (9)\n"
		"PERIODIC (10)");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm); 

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}

	int K_id = 0;
	if (vm.count("kernel")){
		K_id = vm["kernel"].as<int>();
	}

	typedef double float_type;
	typedef MLMatrix<float_type> Matrix;
	typedef MLVector<float_type> Vector;


	// mean and query points
	Vector mean = Vector::Zero(N_STEPS);
	Matrix pts = Matrix::Zero(1, N_STEPS);
	pts.row(0) = Vector::LinSpaced(N_STEPS, -10, 10);

	Matrix samples(N_STEPS, N_SAMPLES);

	// Quite ugly, but classes were not designed for 
	// dynamic polymorphism
	switch(K_id){
		case 0: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::LINEAR>());
			break;
		case 1: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::POLYNOMIAL, float_type>());
			break;
		case 2: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::RBF, float_type>());
			break;
		case 3: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::LAPLACIAN, float_type>());
			break;
		case 4: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::ABEL, float_type>());
			break;
		case 5: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::CONSTANT, float_type>());
			break;
		case 6: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::MIN>());
			break;
		case 7: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, 
					Kernel<KernelType::MATERN32, float_type>());
			break;
		case 8: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, 
					Kernel<KernelType::MATERN52, float_type>());
			break;
		case 9: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, 
					Kernel<KernelType::RATIONAL_QUADRATIC, float_type>());
			break;
		case 10: 
			samples = 
				GaussianProcess<float_type>::sample(
					mean, pts, Kernel<KernelType::PERIODIC, float_type>());
			break;
	}

	gp << "set xrange [-10:10]\nset yrange [-3:3]\n";
	std::vector<std::pair<float_type, float_type>> gnu_pts;
	for(uint i = 0; i < N_STEPS; ++i) {
		gnu_pts.push_back(std::make_pair(pts(0, i), samples(i, 0)));
	}
	gp << "plot '-' with lines\n";
	gp.send1d(gnu_pts);
	return 0;
}