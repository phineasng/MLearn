#include <iostream>
#include <string>
#include <fstream>

#include <MLearn/Core>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/NeuralNets/EBM/RBM/RBMSampler.h>
#include <MLearn/NeuralNets/EBM/RBM/RBMCost.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>
#include <MLearn/Optimization/AdaGrad.h>
#include <MLearn/Optimization/AdaDelta.h>
#include <MLearn/Optimization/Momentum.h>
#include <MLearn/Optimization/GradientDescent.h>
#include <MLearn/Optimization/Differentiation/Differentiator.h>
#include <MLearn/Utility/MemoryPool/MLVectorPool.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#define MNIST_TRAIN_IMAGES_PATH "../../datasets/MNIST/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS_PATH "../../datasets/MNIST/train-labels-idx1-ubyte"
#define OUTPUT_WEIGHTS_PATH "./trained_weights.txt"

#define FLOAT_TYPE double
#define INT_TYPE uint
#define VISUALIZE_FLAG false
#define WINDOW_NAME "Activation - visualize"
#define WINDOW_NAME_SAMPLING "Sample"

#define MIN_GAP_SIZE 2
#define N_IMG_COLS 10

using namespace MLearn;
using namespace NeuralNets;
using namespace Optimization;
using namespace cv;
using namespace std;
using namespace RBMSupport;
using namespace Utility::MemoryPool;

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
	output = MLMatrix<FLOAT_TYPE>::Constant(10,N_images,-1);

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

void show_filters( const MLMatrix< FLOAT_TYPE >& weights, int n_hidden ){

	MLMatrix< FLOAT_TYPE > weight_matrix = Eigen::Map< const MLMatrix< FLOAT_TYPE > >( weights.data(),n_hidden,28*28 );
	weight_matrix.transposeInPlace();
	MLMatrix<FLOAT_TYPE> eigen_image(28,28);
	Mat image_gray;
	Mat_<FLOAT_TYPE> image_to_eigen_gray;
	int n_rows = ceil((double)n_hidden/(double)N_IMG_COLS);

	cv::Mat result = cv::Mat::zeros(n_rows*28 + (n_rows+1)*MIN_GAP_SIZE,
                                    N_IMG_COLS*28 + (N_IMG_COLS+1)*MIN_GAP_SIZE, image_gray.type());

	size_t i = 0;
    int current_height = MIN_GAP_SIZE;
    int current_width = MIN_GAP_SIZE;
    for ( int y = 0; y < n_rows; y++ ) {
        for ( int x = 0; x < N_IMG_COLS; x++ ) {
			cv::Mat to(result,
                       cv::Range(current_height, current_height + 28),
                       cv::Range(current_width, current_width + 28));
			eigen_image = Eigen::Map<MLMatrix<double>>(weight_matrix.data()+i*28*28,28,28);
			eigen_image /= std::sqrt(eigen_image.array().abs2().sum());
			eigen2cv(eigen_image,image_to_eigen_gray);
			normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
			image_gray.copyTo(to);
			current_width += 28 + MIN_GAP_SIZE;
			++i;
		}
		current_width = MIN_GAP_SIZE;
        current_height += 28 + MIN_GAP_SIZE;
	}
	imshow( WINDOW_NAME, result );
	cv::waitKey(1);
}
int main(){

	// import dataset
	MLMatrix< FLOAT_TYPE > images;
	MLMatrix< FLOAT_TYPE > class_assignments;

	importMNIST(images,class_assignments);

	uint N_images = images.cols();
	uint N_vis = images.rows();
	uint N_hid = 100u;
	uint N_params = N_vis*N_hid + N_vis + N_hid;

	constexpr RBMUnitType visible = RBMUnitType::BERNOULLI;
	constexpr RBMUnitType hidden = RBMUnitType::BERNOULLI;
	constexpr Regularizer reg = Regularizer::NONE;   
	constexpr RBMTrainingMode mode = RBMTrainingMode::CONTRASTIVE_DIVERGENCE;
	MLVector<FLOAT_TYPE> grad_tmp(N_params);
	RegularizerOptions<FLOAT_TYPE> opt;
	opt._l2_param = 0.005;
	opt._l1_param = 0.001;
	RBMSampler<FLOAT_TYPE,visible,hidden> sampler(N_vis,N_hid);
	MLVector<FLOAT_TYPE> params = 0.00005*MLVector<FLOAT_TYPE>::Random(N_params);
	sampler.attachParameters(params);
	RBMCost< reg, RBMSampler<FLOAT_TYPE,visible,hidden>, mode,1 > cost(sampler,images,opt,grad_tmp);

	//sampler.setVisibleDistributionParameters(5);
	//sampler.setHiddenDistributionParameters(20);
	
	LineSearch< LineSearchStrategy::FIXED,double,uint > line_search(0.1);
	//Optimization::StochasticGradientDescent<LineSearchStrategy::FIXED,double,uint,0> minimizer;
	//Optimization::AdaGrad<LineSearchStrategy::FIXED,double,uint,0> minimizer;
	//Optimization::AdaDelta<double,uint,0> minimizer;
	Optimization::Momentum<LineSearchStrategy::FIXED,double,uint,0> minimizer;
	minimizer.setMaxIter(600);
	minimizer.setMaxEpoch(1);
	minimizer.setSizeBatch(100);
	minimizer.setNSamples(N_images);
	minimizer.setLineSearchMethod(line_search);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	minimizer.setSeed(seed);
	//std::cout << "Start) Average Free energy = " << cost.evaluate(params) << std::endl;
	namedWindow( WINDOW_NAME, WINDOW_OPENGL );
	show_filters( params, N_hid );

	for ( uint i = 1; i <= 20; ++i ){
		//auto start_time = chrono::high_resolution_clock::now();
		minimizer.minimize(cost,params);
		minimizer.setInitializedFlag(false); // for AdaGrad, AdaDelta and Momentum
		//auto end_time = chrono::high_resolution_clock::now();
		//std::cout << std::chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << ":"<< std::endl;
		if (!(i%5))
			std::cout << i <<")Average Free energy = " << cost.evaluate(params) << std::endl;
		else
			std::cout << i << std::endl;
		show_filters( params, N_hid );
	}
	std::cout << "Press a key" << std::endl;
	cv::waitKey();

	namedWindow(WINDOW_NAME_SAMPLING, WINDOW_OPENGL);

	MLMatrix<double> eigen_image(28,28);
	Mat image_gray,image_blurred;
	Mat_<double> image_to_eigen_gray;
	sampler.attachParameters(params);

	while(1){
		// sample
		sampler.sampleHFromV();
		sampler.sampleVFromH();
		eigen_image = Eigen::Map< const MLMatrix<FLOAT_TYPE> >(sampler.getVisibleUnits().data(),28,28);
		eigen2cv(eigen_image,image_to_eigen_gray);
		normalize(image_to_eigen_gray,image_gray,0,255,NORM_MINMAX,CV_8UC1);
		medianBlur(image_gray,image_blurred,3);
		imshow( WINDOW_NAME_SAMPLING, image_blurred );
		cv::waitKey(1);
	}

	destroyWindow(WINDOW_NAME);
	destroyWindow(WINDOW_NAME_SAMPLING);
	return 0;

}