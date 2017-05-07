// gnuplot-c++ includes
#include <gnuplot-iostream.h>

// MLearn includes
#include <MLearn/Core>
#include <MLearn/Regression/GPRegressor/GPRegressor.h>

// STL includes
#include <vector>
#include <array>
#include <chrono>
#include <string>
#include <cmath>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SVD>

// Boost includes
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#define N_TRAIN_POINTS 50 // n_points along the time
#define N_TEST_POINTS 500 // n_points along the time
#define N_SAMPLES 1 // n of curves to sample

typedef double float_type;

float_type test_function(const float_type& value){
	return std::sin(value)*value + std::exp(-value*value);
}


int main(int argc, char** argv){
	std::srand((unsigned int) time(0));

	Gnuplot gp;

	using namespace MLearn;
	using namespace Regression;
	namespace po = boost::program_options;

	typedef MLMatrix<float_type> Matrix;
	typedef MLVector<float_type> Vector;

	// Create command line options
	po::options_description 
	desc("This is a demo showing gaussian process regression.");

	desc.add_options()
		("help", "Show the help")
		("noise", po::value<float_type>(), "Noise variance (default: 0.0).");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm); 

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}

	float_type noise = 0;
	if (vm.count("noise")){
		noise = vm["noise"].as<float_type>();
	}

	// noise
	boost::normal_distribution<float_type> dist;
	boost::random::mt19937 
		generator(std::chrono::system_clock::now().time_since_epoch().count());

	// train points
	Matrix X_train = Matrix::Random(1, N_TRAIN_POINTS);
	Matrix X_test(1, N_TEST_POINTS);
	X_test.row(0) = Vector::LinSpaced(N_TEST_POINTS, -6.0, 6.0);
	X_train.array() *= 12.0;
	X_train.array() -= 6.0;
	Vector y_train(N_TRAIN_POINTS);
	Vector y_real(N_TEST_POINTS);
	for (int i = 0; i < N_TRAIN_POINTS; ++i){
		y_train[i] = test_function(X_train(0, i)) + noise*dist(generator); 
	}
	for (int i = 0; i < N_TEST_POINTS; ++i){
		y_real[i] = test_function(X_test(0, i)); 
	}

	// Define GPRegressor
	typedef Kernel<KernelType::RBF, float_type> KT;
	Regression::GPRegressor<KT, float_type> regressor;
	regressor.fit(X_train, y_train, noise);
	Vector y_pred = regressor.predict(X_test);
	Matrix samples_pred = regressor.sample(5);
	Matrix confidence_intervals = regressor.confidence_interval(0.95);

	gp << "set multiplot\n";
	gp << "set xrange [-6:6]\nset yrange [-6:6]\n";
	std::vector<std::pair<float_type, float_type>> train_pts;
	std::vector<std::pair<float_type, float_type>> real_pts;
	std::vector<std::pair<float_type, float_type>> test_pts;
	std::vector<std::pair<float_type, float_type>> sample_pts_1;
	std::vector<std::pair<float_type, float_type>> sample_pts_2;
	std::vector<std::pair<float_type, float_type>> sample_pts_3;
	std::vector<std::tuple<float_type, float_type, float_type>> ci;
	for(uint i = 0; i < N_TRAIN_POINTS; ++i) {
		train_pts.push_back(std::make_pair(X_train(0, i), y_train[i]));
	}
	for(uint i = 0; i < N_TEST_POINTS; ++i) {
		test_pts.push_back(std::make_pair(X_test(0, i), y_pred[i]));
		real_pts.push_back(std::make_pair(X_test(0, i), y_real[i]));

		sample_pts_1.push_back(std::make_pair(X_test(0, i), samples_pred(i,0)));
		sample_pts_2.push_back(std::make_pair(X_test(0, i), samples_pred(i,1)));
		sample_pts_3.push_back(std::make_pair(X_test(0, i), samples_pred(i,2)));

		ci.push_back(std::make_tuple(
			X_test(0, i), confidence_intervals(0,i), confidence_intervals(1,i)
		));
	}
	gp << "plot '-' u 1:2:3 with filledcurves lc rgb \"grey\" notitle,"
		  "'-' with points lc rgb \"black\" notitle," 
		  "'-' with lines lc rgb \"black\" title \"Mean fn\", "
		  "'-' with lines lc rgb \"blue\" title \"Real fn\", "
		  "'-' with lines lc rgb \"red\" notitle, "
		  "'-' with lines lc rgb \"cyan\" notitle, "
		  "'-' with lines lc rgb \"orange\" notitle\n";
	gp.send1d(ci);
	gp.send1d(train_pts);
	gp.send1d(test_pts);
	gp.send1d(real_pts);
	gp.send1d(sample_pts_1);
	gp.send1d(sample_pts_2);
	gp.send1d(sample_pts_3);
	return 0;
}