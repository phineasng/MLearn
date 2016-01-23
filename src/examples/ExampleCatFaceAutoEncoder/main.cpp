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
#define FILENAME "trained_weights_hidden400.txt"

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
		normalize(image_gray,image_to_eigen_gray,0,1,NORM_MINMAX,CV_64FC1);
		cv2eigen(image_to_eigen_gray,eigen_image);
		samples.col(i) = view;
	}

	// Set up
	uint N_layers = 3;
	uint N_hidden = 100;

	// Setting layers
	MLVector<uint> layers(N_layers);
	layers << 1024,N_hidden,1024;

	// Activation types
	constexpr ActivationType hidden_act = ActivationType::LOGISTIC;
	constexpr ActivationType output_act = ActivationType::LINEAR;
	// Loss and regularization type
	constexpr LossType loss = LossType::L2_SQUARED;
	constexpr Regularizer reg = Regularizer::L2 | Regularizer::SHARED_WEIGHTS;

	// Generate dataset 
	//srand((unsigned int) time(0));
	MLMatrix<double>& outputs = samples;

	// Set some random weights
	srand((unsigned int) time(0));
	MLVector<double> weights = 0.005*MLVector<double>::Random( layers.head(N_layers-1).dot(layers.tail(N_layers-1)) + layers.tail(N_layers-1).array().sum() );
	weights.array() -= weights.array().mean();
	MLVector<double> gradient_pre_allocation(weights.size());
	MLVector<double> gradient(weights.size());
	MLVector<double> gradient_numerical(weights.size());

	Eigen::Map< MLMatrix<double> > v1(weights.data(),N_hidden,1024);
	Eigen::Map< MLMatrix<double> > v2(weights.data()+1025*N_hidden,1024,N_hidden);

	v2 = v1.transpose();

	// Build the net explorer
	FCNetsExplorer<double,uint,hidden_act,output_act> explorer(layers);

	// Allocate some supporting memory
	MLVector<double> grad_output(1024);

	// Set Regularization options
	RegularizerOptions<double> options;
	options._l2_param = 0.005;
	options._l1_param = 0.005;
	
	namedWindow( "First Layer Activation - visualize", WINDOW_OPENGL );

	Eigen::Matrix< uint, 2, -1, Eigen::ColMajor | Eigen::AutoAlign > shared(2,0);
	Eigen::Matrix< uint, 2, -1, Eigen::ColMajor | Eigen::AutoAlign > tr_shared(2,1);
	tr_shared(0,0) = 0;
	tr_shared(1,0) = 1;

	LineSearch< LineSearchStrategy::FIXED,double,uint > line_search(0.005);
	TEMPLATED_FC_NEURAL_NET_COST_CONSTRUCTION_WITH_SHARED_WEIGHTS( loss,reg,layers,samples,samples,explorer,options,grad_output,gradient_pre_allocation,shared,tr_shared,cost);	
	Optimization::StochasticGradientDescent<LineSearchStrategy::FIXED,double,uint,3> minimizer;
	minimizer.setMaxIter(20000);
	minimizer.setMaxEpoch(100);
	minimizer.setSizeBatch(50);
	minimizer.setNSamples(N_images);
	minimizer.setLineSearchMethod(line_search);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	minimizer.setSeed(seed);
	minimizer.minimize(cost,weights);
	
	MLMatrix< double > weight_matrix = Eigen::Map< MLMatrix< double > >( weights.data(),N_hidden,1024 );
	weight_matrix.transposeInPlace();

	for (uint i = 0; i < N_hidden; ++i){

		eigen_image = Eigen::Map<MLMatrix<double>>(weight_matrix.data()+i*1024,32,32);
		eigen_image /= std::sqrt(eigen_image.array().abs2().sum());
		eigen2cv(eigen_image,image_to_eigen_gray);
		normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
		imshow( "First Layer Activation - visualize", image_gray );
		cv::waitKey();
	
	}
	
	destroyWindow("First Layer Activation - visualize");

  	std::ofstream myfile;
  	myfile.open(FILENAME, std::ofstream::out | std::ofstream::trunc);
  	myfile << weights;
 	myfile.close();


	

 	return 0;
}