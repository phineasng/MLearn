// gnuplot-c++ includes
#include <gnuplot-iostream.h>

// MLearn includes
#include <MLearn/Core>
#include <MLearn/Sampling/GaussianSampling.h>

// STL includes
#include <vector>
#include <tuple>
#include <string>
#include <cmath>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SVD>

int main(int argc, char** argv){
	std::srand((unsigned int) time(0));

	Gnuplot gp;

	using namespace MLearn;
	using namespace Sampling::Gaussian;

	typedef double float_type;
	typedef MLMatrix<float_type> Matrix;
	typedef MLVector<float_type> Vector;

	uint DIM = 2;
	uint N = 5000;

	// compute a random covariance matrix
	Matrix random = Matrix::Random(DIM, DIM);
	Eigen::JacobiSVD<Matrix> svd(random, 
		Eigen::ComputeFullU | Eigen::ComputeFullV );
	Matrix covariance = svd.matrixU();
	Matrix eig_val = (svd.singularValues().cwiseAbs().cwiseSqrt()).asDiagonal();
	covariance = covariance*eig_val*eig_val;
	covariance = covariance*(svd.matrixU().transpose());
	// compute mean
	Vector mean = Vector::Random(DIM);
	mean.array() -= float_type(0.5);
	
	// sample
	Matrix samples = MultivariateGaussian<float_type>::sample(
		mean, covariance, N);

	// draw the samples
	std::vector<std::pair<double, double> > xy_pts;
	for(uint i = 0; i < N; ++i) {
		xy_pts.push_back(std::make_pair(samples(0, i), samples(1, i)));
	}
	gp << "set multiplot\n";
	gp << "set xrange [-5:5]\nset yrange [-5:5]\n";
	gp << "plot '-' with points notitle\n";
	gp.send1d(xy_pts);

	gp << "set parametric\n";
	gp << "set trange [0:2*pi]\n";
	gp << "fx(t)=" + 
	      std::to_string(svd.matrixU()(0,0)*eig_val(0,0)*1.96) +
	      "*cos(t)+" + 
	      std::to_string(svd.matrixU()(0,1)*eig_val(1,1)*1.96) +
	      "*sin(t)+" + std::to_string(mean(0)) + "\n";
	gp << "fy(t) = " + 
	      std::to_string(svd.matrixU()(1,0)*eig_val(0,0)*1.96) +
	      "*cos(t) + " + 
	      std::to_string(svd.matrixU()(1,1)*eig_val(1,1)*1.96) +
	      "*sin(t) + " + std::to_string(mean(1)) + "\n"; 
	gp << "plot fx(t), fy(t) linecolor rgb \"blue\" notitle\n";

	


	return 0;
}