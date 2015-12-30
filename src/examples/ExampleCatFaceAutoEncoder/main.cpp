#include <iostream>
#include <string>
#include <fstream>

#include <MLearn/Core>
#include <MLearn/NeuralNets/FeedForwardNets/Common/FCNetsExplorer.h>
#include <MLearn/NeuralNets/FeedForwardNets/Common/FCCostFunction.h>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#define PATH_TO_DATA "../../../../../Downloads/cat_face_gray/%05d.jpg"
#define FILENAME "trained_weights.txt"

int main(int argc, char* argv[]){

	int N_images =11465;

	//
	using namespace MLearn;
	using namespace NeuralNets;
	using namespace FeedForwardNets;
	using namespace Optimization;
	using namespace cv;

	// open windows

	// Read images 
	std::string pathToData(PATH_TO_DATA);
	cv::VideoCapture sequence(pathToData);
	Mat image;
	Mat image_gray;
	Mat_<double> image_to_eigen_gray;
	MLMatrix<double> eigen_image(32,32);
	Eigen::Map< MLVector<double> > view(eigen_image.data(),1024);
	MLMatrix<double> samples( 1024, N_images );
	MLVector<double> temp(1024);
	for (int i = 0;i < N_images; ++i){
		sequence >> image;
		cvtColor(image,image_gray,CV_BGR2GRAY);
		image_gray.convertTo(image_to_eigen_gray,CV_64FC1,1.0/255.0);
		cv2eigen(image_to_eigen_gray,eigen_image);
		samples.col(i) = view;
	}

	// Set up
	uint N_layers = 3;
	uint N_hidden = 600;

	// Setting layers
	MLVector<uint> layers(N_layers);
	layers << 1024,N_hidden,1024;

	// Activation types
	constexpr ActivationType hidden_act = ActivationType::LOGISTIC;
	constexpr ActivationType output_act = ActivationType::LINEAR;

	// Loss and regularization type
	constexpr LossType loss = LossType::L2_SQUARED;
	constexpr Regularizer reg = Regularizer::L2;

	// Generate dataset 
	//srand((unsigned int) time(0));
	MLMatrix<double>& outputs = samples;

	// Set some random weights
	srand((unsigned int) time(0));
	MLVector<double> weights = 0.1*MLVector<double>::Random( layers.head(N_layers-1).dot(layers.tail(N_layers-1)) + layers.tail(N_layers-1).array().sum() );
	weights.array() -= weights.array().mean();
	MLVector<double> gradient_pre_allocation(weights.size());
	MLVector<double> gradient(weights.size());
	MLVector<double> gradient_numerical(weights.size());

	// Build the net explorer
	FCNetsExplorer<double,uint,hidden_act,output_act> explorer(layers);

	// Allocate some supporting memory
	MLVector<double> grad_output(1024);

	// Set Regularization options
	RegularizerOptions<double> options;
	options._l2_param = 0.01;
	options._l1_param = 0.01;

	namedWindow( "First Activation - visualize", WINDOW_OPENGL );

	LineSearch< LineSearchStrategy::FIXED,double,uint > line_search(0.05);
	TEMPLATED_FC_NEURAL_NET_COST_CONSTRUCTION( loss,reg,layers,samples,samples,explorer,options,grad_output,gradient_pre_allocation,cost);	
	Optimization::StochasticGradientDescent<LineSearchStrategy::FIXED,double,uint,0> minimizer;
	minimizer.setMaxIter(50000);
	minimizer.setDistributionParameters(0,N_images-1);
	minimizer.setSizeBatch(20);
	minimizer.setLineSearchMethod(line_search);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	minimizer.setSeed(seed);
	minimizer.minimize(cost,weights);	
	
	MLMatrix< double > weight_matrix = Eigen::Map< MLMatrix< double > >( weights.data(),N_hidden,1024 );
	weight_matrix.transposeInPlace();

	for (uint i = 0; i < N_hidden; ++i){

		eigen_image = Eigen::Map<MLMatrix<double>>(weight_matrix.data()+i*1024,32,32);
		eigen_image /= eigen_image.array().abs2().sum();
		eigen2cv(eigen_image,image_to_eigen_gray);
		normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
		imshow( "First Activation - visualize", image_gray );
		cv::waitKey();
	
	}
	
	destroyWindow("First Activation - visualize");

  	std::ofstream myfile;
  	myfile.open(FILENAME, std::ofstream::out | std::ofstream::trunc);
  	myfile << weights;
 	myfile.close();


	

 	return 0;
}