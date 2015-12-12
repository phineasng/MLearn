#include <iostream>
#include <iomanip>

#include <MLearn/Core>
#include <MLearn/Classification/Perceptron/Perceptron.h>

#define N_RANDOM_POINTS 5000

#include <chrono>
#include <random>
#include <cmath>
#include <fstream>

#define SCALAR_TYPE double
#define CLASS_TYPE ushort

#define CENTER_0_X 	0.0
#define CENTER_0_Y 	0.0
#define CENTER_0_Z 	0.0
#define RADIUS_0 	0.42

#define CENTER_1_X 	0.68
#define CENTER_1_Y 	0.64
#define CENTER_1_Z 	1.72
#define RADIUS_1 	0.3

#define FILE_NAME "output.txt"

void generateSamplesFrom2Spheres( SCALAR_TYPE cx0, SCALAR_TYPE cy0, SCALAR_TYPE cz0, SCALAR_TYPE r0,  SCALAR_TYPE cx1, SCALAR_TYPE cy1, SCALAR_TYPE cz1, SCALAR_TYPE r1, MLearn::MLMatrix<SCALAR_TYPE>& dataMatrix, MLearn::MLVector<CLASS_TYPE>& classes){
	// resize data 
	dataMatrix.resize(N_RANDOM_POINTS,3);
	classes.resize(N_RANDOM_POINTS);

	// get seed for random number generator
  	unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  	// initialize random generator
  	std::mt19937_64 generator(seed);

  	// uniform distribution for radius
  	std::uniform_real_distribution<SCALAR_TYPE> distance(0.0,1.0);

  	// normal distribution for the 3 coordinates
  	std::normal_distribution<SCALAR_TYPE> X(0.0,1.0);
  	std::normal_distribution<SCALAR_TYPE> Y(0.0,1.0);
  	std::normal_distribution<SCALAR_TYPE> Z(0.0,1.0);

  	SCALAR_TYPE x,y,z,d,norm;

  	// bernoulli distribution for picking the sphere
  	std::bernoulli_distribution pick_sphere(0.5);

  	for (int i = 0; i < N_RANDOM_POINTS; ++i){
  		d = distance(generator);
  		x = X(generator);
  		y = Y(generator);
  		z = Z(generator);

  		d = std::cbrt(d);

  		norm = std::sqrt(x*x + y*y + z*z);
  		if ( norm > 1e-30 ){
  			x /= norm;
  			y /= norm;
  			z /= norm;
  		}

  		if ( pick_sphere(generator) ){

  			dataMatrix(i,0) = x*d*r1 + cx1;
  			dataMatrix(i,1) = y*d*r1 + cy1;
  			dataMatrix(i,2) = z*d*r1 + cz1;

  			classes[i] = 1;

  		}else{

  			dataMatrix(i,0) = x*d*r0 + cx0;
  			dataMatrix(i,1) = y*d*r0 + cy0;
  			dataMatrix(i,2) = z*d*r0 + cz0;

  			classes[i] = 0;
  			
  		}
  	}

}

void outputFile(const MLearn::MLMatrix<SCALAR_TYPE>& data, const MLearn::MLVector<CLASS_TYPE>& classes){
	// create output matrix
	MLearn::MLMatrix<SCALAR_TYPE> toPlot(data.rows()+1,data.cols());
	toPlot.block(0,0,data.rows(),data.cols()) = data;
	for (int i = 0; i < data.cols(); ++i){
		toPlot(data.rows(),i) = (SCALAR_TYPE)classes[i];
	}	
	// initialize file
	std::ofstream file;
	file.open(FILE_NAME, std::ios::out | std::ios::trunc );
	file << toPlot.transpose();
	file.close();

}

int main(int argc, char* argv[]){

	uint n = 3;
	MLearn::Classification::Perceptron<double,ushort> perceptron(n);
	perceptron.initializeRandom(0.05f);
	perceptron.setMaxIter(1000u);
	
	perceptron.setTolerance( 1e-2 );
	perceptron.setLearningRate( 1 );
	MLearn::MLVector< CLASS_TYPE > labels(N_RANDOM_POINTS);
	MLearn::MLMatrix< SCALAR_TYPE > features(N_RANDOM_POINTS,3);

	std::cout<<std::setprecision(20);
	
	generateSamplesFrom2Spheres(CENTER_0_X,
								CENTER_0_Y,
								CENTER_0_Z,
								RADIUS_0,
								CENTER_1_X,
								CENTER_1_Y,
								CENTER_1_Z,
								RADIUS_1,
								features,
								labels);

	//std::cout << perceptron.getWeights() << std::endl;
	perceptron.train< MLearn::Classification::PerceptronTraining::BATCH_AVERAGE >( features, labels );
	//std::cout << perceptron.getWeights() << std::endl;
	if (std::fabs(perceptron.getWeights()[3]) > 1e-30){
		std::cout 	<< - perceptron.getWeights()[0]/perceptron.getWeights()[3] << " " 
					<< - perceptron.getWeights()[1]/perceptron.getWeights()[3] << " "
					<< - perceptron.getWeights()[2]/perceptron.getWeights()[3] << std::endl;
	}

	// output file to plot
	//outputFile(features,labels);

	generateSamplesFrom2Spheres(CENTER_0_X,
								CENTER_0_Y,
								CENTER_0_Z,
								RADIUS_0,
								CENTER_1_X,
								CENTER_1_Y,
								CENTER_1_Z,
								RADIUS_1,
								features,
								labels);
	MLearn::MLVector<ushort> predicted = perceptron.classify(features);
	std::cout << "Error 0-1 loss on new data: " << (predicted-labels).array().abs().mean() << std::endl;
	return 0;
}