#include <iostream>
#include <string>
#include <fstream>

#include <MLearn/Core>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/NeuralNets/FeedForwardNets/Common/FeedForwardBase.h>
#include <MLearn/NeuralNets/FeedForwardNets/AutoEncoder/AutoEncoder.h>
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
	uint N_hidden = 600;

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
	MLVector<double> weights = 0.05*MLVector<double>::Random( layers.head(N_layers-1).dot(layers.tail(N_layers-1)) + layers.tail(N_layers-1).array().sum() );
	weights.array() -= weights.array().mean();
	
	MLMatrix<uint> sh(2,0);
 	MLMatrix<uint> tr_sh(2,1);
 	tr_sh(0,0) = 0;
 	tr_sh(1,0) = 1;
 	
	AutoEncoder<double,uint,hidden_act,output_act> net(layers);
	net.setLayers(layers);
	net.setWeights(weights);
	net.setSharedWeights(sh);
	net.setTransposedSharedWeights(tr_sh);

	LineSearch< LineSearchStrategy::FIXED,double,uint > line_search(0.015);
	Optimization::StochasticGradientDescent<LineSearchStrategy::FIXED,double,uint,3> minimizer;
	minimizer.setMaxIter(20000);
	minimizer.setMaxEpoch(100);
	minimizer.setSizeBatch(50);
	minimizer.setNSamples(N_images);
	minimizer.setLineSearchMethod(line_search);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	minimizer.setSeed(seed);
	
	RegularizerOptions<double> options;
	options._l2_param = 0.005;
	options._l1_param = 0.005;

	net.train< FFNetTrainingMode::BATCH, reg, loss >(samples,minimizer,options);
	weights = net.getWeights();

	namedWindow( "Activation - visualize", WINDOW_OPENGL );
	MLMatrix< double > weight_matrix = Eigen::Map< MLMatrix< double > >( weights.data(),N_hidden,1024 );
	weight_matrix.transposeInPlace();

	for (uint i = 0; i < N_hidden; ++i){

		eigen_image = Eigen::Map<MLMatrix<double>>(weight_matrix.data()+i*1024,32,32);
		eigen_image /= std::sqrt(eigen_image.array().abs2().sum());
		eigen2cv(eigen_image,image_to_eigen_gray);
		normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
		imshow( "Activation - visualize", image_gray );
		cv::waitKey();
	
	}

	destroyWindow("Activation - visualize");
	std::ofstream myfile;
  	myfile.open(FILENAME, std::ofstream::out | std::ofstream::trunc);
  	myfile << weights;
 	myfile.close();


 	return 0;
}