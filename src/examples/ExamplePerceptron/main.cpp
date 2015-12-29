#include <iostream>

#include <MLearn/Core>
#include <MLearn/Classification/Perceptron/Perceptron.h>

#define N_RANDOM_POINTS 5000
#define N_RANDOM_POINTS_2 50000
#define EXAMPLE_TRAINING_MODE MLearn::Classification::PerceptronTraining::ONLINE
#define EXAMPLE_TRAINING_MODE_2 MLearn::Classification::PerceptronTraining::BATCH

#include <chrono>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>

#define SCALAR_TYPE double
#define CLASS_TYPE ushort

#define CENTER_0_X 	0.0
#define CENTER_0_Y 	0.0
#define CENTER_0_Z 	0.0
#define RADIUS_0 	0.1

#define CENTER_1_X 	0.5
#define CENTER_1_Y 	0.5
#define CENTER_1_Z 	0.1
#define RADIUS_1 	0.3

#define FILE_NAME "output.txt"

void generateSamplesFrom2Spheres( SCALAR_TYPE cx0, SCALAR_TYPE cy0, SCALAR_TYPE cz0, SCALAR_TYPE r0,  SCALAR_TYPE cx1, SCALAR_TYPE cy1, SCALAR_TYPE cz1, SCALAR_TYPE r1, MLearn::MLMatrix<SCALAR_TYPE>& dataMatrix, MLearn::MLVector<CLASS_TYPE>& classes){
	auto n_sample = classes.size(); 

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

  	for (int i = 0; i < n_sample; ++i){
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

  			dataMatrix(0,i) = x*d*r1 + cx1;
  			dataMatrix(1,i) = y*d*r1 + cy1;
  			dataMatrix(2,i) = z*d*r1 + cz1;

  			classes[i] = 1;

  		}else{

  			dataMatrix(0,i) = x*d*r0 + cx0;
  			dataMatrix(1,i) = y*d*r0 + cy0;
  			dataMatrix(2,i) = z*d*r0 + cz0;

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
	MLearn::MLMatrix< SCALAR_TYPE > features(3,N_RANDOM_POINTS);
	std::cout << std::setprecision(20);
	
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

	perceptron.train< EXAMPLE_TRAINING_MODE >( features, labels );

	// output file to plot
	//outputFile(features,labels);
	labels.resize(N_RANDOM_POINTS_2);
	features.resize(3,N_RANDOM_POINTS_2);

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
	std::cout << "Error 0-1 loss on new data: " << ((predicted-labels).array()>0).count()/(double)labels.size() << std::endl;

	perceptron.initializeRandom(0.05f);	
	labels.resize(N_RANDOM_POINTS);
	features.resize(3,N_RANDOM_POINTS);

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
	perceptron.train< EXAMPLE_TRAINING_MODE_2 >( features, labels );
	// output file to plot
	//outputFile(features,labels);
	labels.resize(N_RANDOM_POINTS_2);
	features.resize(3,N_RANDOM_POINTS_2);

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
	predicted = perceptron.classify(features);
	std::cout << "Error 0-1 loss on new data: " << ((predicted-labels).array()>0).count()/(double)labels.size() << std::endl;

	return 0;
}