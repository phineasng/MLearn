#include <iostream>
#include <string>
#include <fstream>

#include <MLearn/Core>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/NeuralNets/FeedForwardNets/Common/FeedForwardBase.h>
#include <MLearn/NeuralNets/FeedForwardNets/MultiLayerPerceptron/MultiLayerPerceptron.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#define MNIST_TEST_IMAGES_PATH "../../datasets/MNIST/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS_PATH "../../datasets/MNIST/t10k-labels-idx1-ubyte"
#define INPUT_WEIGHTS_PATH "./trained_weights.txt"

#define FLOAT_TYPE double
#define INT_TYPE uint
#define VISUALIZE_FLAG false
#define INTERACTIVE_FLAG false

using namespace MLearn;
using namespace NeuralNets;
using namespace FeedForwardNets;
using namespace Optimization;
using namespace cv;
using namespace std;

void importMNIST_test( MLMatrix< FLOAT_TYPE >& images, MLVector< INT_TYPE >& output ){

	// read images file
	ifstream imageFile;
	ifstream labelFile;
	
	imageFile.open( MNIST_TEST_IMAGES_PATH, ios::binary );
	labelFile.open( MNIST_TEST_LABELS_PATH, ios::binary );
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
	output = MLVector<INT_TYPE>::Constant(N_images,10);

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
		output[i] = unsigned(byte);
		if (VISUALIZE_FLAG){
			cout << unsigned(byte) << " -> " << output.col(i).transpose() << std::endl;
		}

		// transform the image in the range 0 - 1
		normalize(image,image_to_eigen_gray,-1,1,NORM_MINMAX,CV_64FC1);
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
	MLVector< INT_TYPE > class_assignments;

	importMNIST_test(images,class_assignments);

	uint N_images = images.cols();

	// Setup MLP
	// -- layers
	uint N_layers = 3;
	uint N_hidden_1 = 175;
	MLVector<INT_TYPE> layers(N_layers);
	layers << images.rows(), N_hidden_1,10;
	// -- activation
	constexpr ActivationType hidden_act = ActivationType::HYPER_TAN;
	constexpr ActivationType output_act = ActivationType::LINEAR;
	// -- weights
	MLVector<double> weights( layers.head(N_layers-1).dot(layers.tail(N_layers-1)) + layers.tail(N_layers-1).array().sum() );
	ifstream input_weights(INPUT_WEIGHTS_PATH,ios::in);
	if (!input_weights.is_open()){
		throw "Error opening weights file!";
	}
	int i = 0;
	while( i < weights.size() ){
		input_weights >> weights[i];
		++i;
	}

	// Visualize
	MLMatrix< double > weight_matrix = Eigen::Map< MLMatrix< double > >( weights.data(),N_hidden_1,28*28 );
	weight_matrix.transposeInPlace();
	MLMatrix<double> eigen_image(28,28);
	Mat image_gray;
	Mat_<double> image_to_eigen_gray;

	/*namedWindow( "Activation - visualize", WINDOW_OPENGL );
	for (uint i = 0; i < N_hidden_1; ++i){

		eigen_image = Eigen::Map<MLMatrix<double>>(weight_matrix.data()+i*28*28,28,28);
		eigen_image /= std::sqrt(eigen_image.array().abs2().sum());
		eigen2cv(eigen_image,image_to_eigen_gray);
		normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
		imshow( "Activation - visualize", image_gray );
		cv::waitKey();
	
	}

	destroyWindow("Activation - visualize");*/

	// MLP
	MultiLayerPerceptron<double,uint,hidden_act,output_act> net(layers);
	net.setWeights(weights);

	auto net_classification = net.classify<INT_TYPE>(images);

	double classification_error = 0.;

	if (INTERACTIVE_FLAG){
		namedWindow( "Classification", WINDOW_OPENGL );
	}

	for (uint idx = 0; idx < N_images; ++idx){

		if (INTERACTIVE_FLAG){

			eigen_image = Eigen::Map<MLMatrix<double>>(images.col(idx).data(),28,28);
			eigen2cv(eigen_image,image_to_eigen_gray);
			normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
			imshow( "Classification", image_gray );
			std::cout << "Digit classified as: "<< net_classification[idx] << ". True digit: " << class_assignments[idx] << endl; 
			cv::waitKey();

		}
		if ( net_classification[idx] != class_assignments[idx] ){
			classification_error += 1.0;
		}

	}

	std::cout << "Classification Error = " << classification_error/((FLOAT_TYPE) N_images) << std::endl;

	if (INTERACTIVE_FLAG){
		destroyWindow( "Classification");
	}



	return 0;

}