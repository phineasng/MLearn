#include <iostream>
#include <string>
#include <fstream>

#include <MLearn/Core>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/NeuralNets/FeedForwardNets/Common/FeedForwardBase.h>
#include <MLearn/NeuralNets/FeedForwardNets/MultiLayerPerceptron/MultiLayerPerceptron.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>
#include <MLearn/Optimization/GradientDescent.h>
#include <MLearn/Optimization/Differentiation/Differentiator.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#define MNIST_TRAIN_IMAGES_PATH "../../datasets/MNIST/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS_PATH "../../datasets/MNIST/train-labels-idx1-ubyte"
#define OUTPUT_WEIGHTS_PATH "./trained_weights.txt"

#define FLOAT_TYPE double
#define INT_TYPE uint
#define VISUALIZE_FLAG false

using namespace MLearn;
using namespace NeuralNets;
using namespace FeedForwardNets;
using namespace Optimization;
using namespace cv;
using namespace std;

// import MNIST - for INTEL processor (or other little-endian processors)
void importMNIST( MLMatrix< FLOAT_TYPE >& images, MLMatrix< FLOAT_TYPE >& output ){

	// read images file
	ifstream imageFile;
	ifstream labelFile;
	
	imageFile.open( MNIST_TRAIN_IMAGES_PATH, ios::binary );
	labelFile.open( MNIST_TRAIN_LABELS_PATH, ios::binary );
	// check if open succeded
	if (imageFile.fail() || labelFile.fail()){
		throw "Error opening the file!";
	}

	// useful variables
	unsigned char byte;
	
	// read out magic numbers
	uint32_t magic_number = 0;
	uint32_t magic_number_label = 0;
	imageFile.read((char*)&byte,sizeof(byte));
	magic_number |= ( static_cast<uint32_t>(byte) << 24 );
	imageFile.read((char*)&byte,sizeof(byte));
	magic_number |= ( static_cast<uint32_t>(byte) << 16 );
	imageFile.read((char*)&byte,sizeof(byte));
	magic_number |= ( static_cast<uint32_t>(byte) << 8 );
	imageFile.read((char*)&byte,sizeof(byte));
	magic_number |= ( static_cast<uint32_t>(byte) );
	labelFile.read((char*)&byte,sizeof(byte));
	magic_number_label |= ( static_cast<uint32_t>(byte) << 24 );
	labelFile.read((char*)&byte,sizeof(byte));
	magic_number_label |= ( static_cast<uint32_t>(byte) << 16 );
	labelFile.read((char*)&byte,sizeof(byte));
	magic_number_label |= ( static_cast<uint32_t>(byte) << 8 );
	labelFile.read((char*)&byte,sizeof(byte));
	magic_number_label |= ( static_cast<uint32_t>(byte) );
	
	cout << "Sanity check: Magic number = " << magic_number << std::endl;
	cout << "Sanity check: Magic number label = " << magic_number_label << std::endl;
	if ( (magic_number != 2051) || (magic_number_label != 2049) ){
		throw "Error: magic number not correct!";
	}

	// get number of images
	uint32_t N_images = 0;
	uint32_t N_labels = 0;
	imageFile.read((char*)&byte,sizeof(byte));
	N_images |= ( static_cast<uint32_t>(byte) << 24 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_images |= ( static_cast<uint32_t>(byte) << 16 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_images |= ( static_cast<uint32_t>(byte) << 8 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_images |= ( static_cast<uint32_t>(byte) );
	cout << "Number of images detected = " << N_images << endl;
	labelFile.read((char*)&byte,sizeof(byte));
	N_labels |= ( static_cast<uint32_t>(byte) << 24 );
	labelFile.read((char*)&byte,sizeof(byte));
	N_labels |= ( static_cast<uint32_t>(byte) << 16 );
	labelFile.read((char*)&byte,sizeof(byte));
	N_labels |= ( static_cast<uint32_t>(byte) << 8 );
	labelFile.read((char*)&byte,sizeof(byte));
	N_labels |= ( static_cast<uint32_t>(byte) );
	cout << "Number of labels detected = " << N_labels << endl;

	if (N_images != N_labels){
		throw "Error: different number of training examples detected!";
	}

	// get number of rows and cols
	uint32_t N_rows = 0;
	imageFile.read((char*)&byte,sizeof(byte));
	N_rows |= ( static_cast<uint32_t>(byte) << 24 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_rows |= ( static_cast<uint32_t>(byte) << 16 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_rows |= ( static_cast<uint32_t>(byte) << 8 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_rows |= ( static_cast<uint32_t>(byte) );
	uint32_t N_cols = 0;
	imageFile.read((char*)&byte,sizeof(byte));
	N_cols |= ( static_cast<uint32_t>(byte) << 24 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_cols |= ( static_cast<uint32_t>(byte) << 16 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_cols |= ( static_cast<uint32_t>(byte) << 8 );
	imageFile.read((char*)&byte,sizeof(byte));
	N_cols |= ( static_cast<uint32_t>(byte) );

	cout << "Image dimensions: "<<N_rows<<"x"<<N_cols<<endl;

	images.resize(N_rows*N_cols,N_images);
	output = MLMatrix<FLOAT_TYPE>::Constant(10,N_images,0);

	// use openCV matrix
	Mat image = Mat::zeros(N_rows,N_cols,CV_8UC1);
	Mat_<double> image_to_eigen_gray;
	MLMatrix<double> eigen_image(N_rows,N_cols);
	Eigen::Map< MLVector<double> > view(eigen_image.data(), N_rows*N_cols);

	if (VISUALIZE_FLAG){
		namedWindow( "Current digit", WINDOW_OPENGL );
	}


	for (uint32_t i = 0; i < N_images; ++i){

		for (uint32_t r = 0; r < N_rows; ++r){

			for (uint32_t c = 0; c < N_cols; ++c){

				imageFile.read((char*)&byte,sizeof(byte));
				image.at<uchar>(r,c) = byte;
				if (imageFile.fail()){
					throw "Error reading the file!";
				}
			
			}

		}

		// read label
		labelFile.read((char*)&byte,sizeof(byte));
		if (labelFile.fail()){
			throw "Error reading the file!";
		}
		output.col(i)[unsigned(byte)] = 1;
		if (VISUALIZE_FLAG){
			cout << unsigned(byte) << " -> " << output.col(i).transpose() << std::endl;
		}

		// transform the image in the range 0 - 1
		normalize(image,image_to_eigen_gray,0,1,NORM_MINMAX,CV_64FC1);
		cv2eigen(image_to_eigen_gray,eigen_image);
		images.col(i) = view;

		if (VISUALIZE_FLAG){
			imshow( "Current digit", image );
			waitKey();
		}



	}

	if (VISUALIZE_FLAG){
		destroyWindow("Current digit");
	}

	imageFile.close();
	labelFile.close();

	return;

}


int main(){

	// import dataset
	MLMatrix< FLOAT_TYPE > images;
	MLMatrix< FLOAT_TYPE > class_assignments;

	importMNIST(images,class_assignments);

	uint N_images = images.cols();

	// Setup MLP
	// -- layers
	uint N_layers = 5;
	uint N_hidden_1 = 175;
	uint N_hidden_2 = 100;
	uint N_hidden_3 = 70;
	MLVector<INT_TYPE> layers(N_layers);
	layers << images.rows(), N_hidden_1,N_hidden_2,N_hidden_3,class_assignments.rows();
	// -- activation
	constexpr ActivationType hidden_act = ActivationType::LOGISTIC;
	constexpr ActivationType output_act = ActivationType::LINEAR;
	// -- loss
	constexpr LossType LOSS = LossType::SOFTMAX_CROSS_ENTROPY;
	// -- regularization
	constexpr Regularizer REG = Regularizer::NONE;
	RegularizerOptions< FLOAT_TYPE > options;
	options._l2_param = 0.0005;
	options._l1_param = 0.00005;
	// -- minimizer
	LineSearch< LineSearchStrategy::FIXED,double,uint > line_search(0.5);
	Optimization::StochasticGradientDescent<LineSearchStrategy::FIXED,double,uint,0> minimizer;
	minimizer.setMaxIter(1200000);
	minimizer.setMaxEpoch(1);
	minimizer.setSizeBatch(20);
	minimizer.setNSamples(N_images);
	minimizer.setLineSearchMethod(line_search);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	minimizer.setSeed(seed);
	// -- minimizer_2
	Optimization::GradientDescent<DifferentiationMode::ANALYTICAL,LineSearchStrategy::FIXED,double,uint,1> minimizer_2;
	minimizer_2.setMaxIter(1000);
	minimizer_2.setLineSearchMethod(line_search);
	// -- intial weights
	srand((unsigned int) time(0));
	MLVector<double> weights = 0.005*MLVector<double>::Random( layers.head(N_layers-1).dot(layers.tail(N_layers-1)) + layers.tail(N_layers-1).array().sum() );
	weights.array() -= weights.array().mean();

	// MLP
	MultiLayerPerceptron<double,uint,hidden_act,output_act> net(layers);
	net.setWeights(weights);

	//std::cout << "Gradient Check: ERROR = " << net.gradient_check_implementation< REG, LOSS, DifferentiationMode::NUMERICAL_CENTRAL >(images,class_assignments,options,grad_opt) << std::endl;

	
	// TRAIN!
	//net.train< FFNetTrainingMode::BATCH, REG, LOSS >(images,class_assignments,minimizer,options);
	//net.train< FFNetTrainingMode::ONLINE, REG, LOSS >(images,class_assignments,minimizer_2,options);

	// Visualize and save
		namedWindow( "Activation - visualize", WINDOW_OPENGL );

	for (uint k = 0; k < 2000; ++k){
		std::cout << "Epoch " << k << std::endl;
		net.train< FFNetTrainingMode::BATCH, REG, LOSS >(images,class_assignments,minimizer,options);
		weights = net.getWeights();
		MLMatrix< double > weight_matrix = Eigen::Map< MLMatrix< double > >( weights.data(),N_hidden_1,28*28 );
		weight_matrix.transposeInPlace();
		MLMatrix<double> eigen_image(28,28);
		Mat image_gray;
		Mat_<double> image_to_eigen_gray;

		for (uint i = 0; i < N_hidden_1; ++i){

			eigen_image = Eigen::Map<MLMatrix<double>>(weight_matrix.data()+i*28*28,28,28);
			eigen_image /= std::sqrt(eigen_image.array().abs2().sum());
			eigen2cv(eigen_image,image_to_eigen_gray);
			normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
			imshow( "Activation - visualize", image_gray );
			cv::waitKey(1);
		
		}	
	}

	destroyWindow("Activation - visualize");
	std::ofstream myfile;
  	myfile.open(OUTPUT_WEIGHTS_PATH, std::ofstream::out | std::ofstream::trunc);
  	myfile << weights;
 	myfile.close();
	
	return 0;

}