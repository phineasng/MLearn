#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <random>

#include <MLearn/Core>
#include <MLearn/NeuralNets/layers/fc_layer.h>
#include <MLearn/NeuralNets/neural_nets.h>
#include <MLearn/Optimization/StochasticGradientDescent.h>
#include <MLearn/Optimization/AdaGrad.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <CImg.h>

using namespace std;
using namespace MLearn;
using namespace nn;
using namespace Optimization;
namespace fs = boost::filesystem;
using namespace cimg_library;
typedef double float_type;

// import MNIST - for INTEL processor (or other little-endian processors)
int importMNIST( const fs::path& images_path, const fs::path& labels_path, MLMatrix< float_type >& images, MLMatrix< float_type >& output ){

	ifstream imageFile;
	ifstream labelFile;

	// read images file
	imageFile.open( images_path.string(), ios::binary );
	labelFile.open( labels_path.string(), ios::binary );
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
	output = MLMatrix<float_type>::Constant(10,N_images,0);

	// use openCV matrix
/*	Mat image = Mat::zeros(N_rows,N_cols,CV_8UC1);
	Mat_<double> image_to_eigen_gray;
	MLMatrix<double> eigen_image(N_rows,N_cols);
	Eigen::Map< MLVector<double> > view(eigen_image.data(), N_rows*N_cols);*/


	for (uint32_t i = 0; i < N_images; ++i){

		for (uint32_t r = 0; r < N_rows; ++r){

			for (uint32_t c = 0; c < N_cols; ++c){

				imageFile.read((char*)&byte,sizeof(byte));
				//image.at<uchar>(r,c) = byte;
				images(c*N_rows + r, i) = float_type(byte);
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

		// transform the image in the range 0 - 1
		//normalize(image,image_to_eigen_gray,0,1,NORM_MINMAX,CV_64FC1);
		//cv2eigen(image_to_eigen_gray,eigen_image);
		//images.col(i) = view;
	}
	images.array() /= 255.0;


	imageFile.close();
	labelFile.close();

	return N_rows;

}



int main(int argc, char* argv[]){
	std::srand((unsigned int) time(0));

	namespace po = boost::program_options;
	typedef MLMatrix<float_type> Matrix;
	typedef MLVector<float_type> Vector;
	constexpr ActivationType type = ActivationType::TANH;
	constexpr LossType loss_t = LossType::SOFTMAX_CROSS_ENTROPY;
	typedef FCLayer<float_type, type> layer_t;

	// Create command line options
	po::options_description 
	desc("This is a demo showing a simple FC neural net trained on MNIST.");

	desc.add_options()
		("help", "Show the help")
		("data_folder", po::value<string>(), "Folder where to find the uncompressed data.")
		("visualize,v", po::bool_switch()->default_value(false), "Visualize test samples with classification.");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm); 

	if (vm.count("help")) {
		std::cout << desc << "\n";
		return 1;
	}
	string data_folder;
	if (vm.count("data_folder")){
		data_folder = vm["data_folder"].as<string>();
	}else{
		throw "Invalid data folder";
		return 1;
	}
	fs::path data_folder_path(data_folder);
	bool visualize = vm["visualize"].as<bool>();

	Matrix train_images;
	Matrix test_images;
	Matrix train_labels;
	Matrix test_labels;
	Matrix labels;
	Vector temp;


	int image_rows = importMNIST(data_folder_path/fs::path("train-images.idx3-ubyte"), 
								 data_folder_path/fs::path("train-labels.idx1-ubyte"),
								 train_images, train_labels);
	int image_cols = train_images.rows()/image_rows;
	importMNIST(data_folder_path/fs::path("t10k-images.idx3-ubyte"), data_folder_path/fs::path("t10k-labels.idx1-ubyte"),
				test_images, test_labels);


	Vector real_labels = Vector::Zero(test_labels.cols());
	Vector hat_labels = Vector::Zero(test_labels.cols());

	for (int idx = 0; idx < test_labels.cols(); ++idx){
		temp = test_labels.col(idx);
		temp.maxCoeff(&real_labels[idx]);
	}


	auto network = make_network<float_type, loss_t>(layer_t(100), layer_t(50), layer_t(train_labels.rows()));
	network.set_input_dim(train_images.rows());
	Vector weights = Vector::Random(network.get_n_parameters())*2.5;
	network.set_weights(weights.data(), true);

	LineSearch< LineSearchStrategy::FIXED,float_type,uint > line_search(0.3);
	Optimization::AdaGrad<LineSearchStrategy::FIXED,float_type,uint,0> minimizer;
	minimizer.setMaxIter(10);
	minimizer.setMaxEpoch(1);
	minimizer.setSizeBatch(500);
	minimizer.setNSamples(train_images.cols());
	minimizer.setLineSearchMethod(line_search);
	minimizer.setSeed(std::chrono::system_clock::now().time_since_epoch().count());
	std::cout << "Initial loss: " << network.evaluate(test_images, test_labels) << std::endl;
	int n_epochs = 0;

	random_device rand_dev;
	mt19937 generator(rand_dev());
	uniform_int_distribution<int> dist(0, test_images.cols());
	char title[50];
	CImgDisplay main_disp(800, 800,"Test samples", 3, false, true);
	while (true){
		++n_epochs;
		network.fit(train_images, train_labels, minimizer);
		labels = network.forward_pass(test_images);
		float_type accuracy = 0;
		for (int idx = 0; idx < test_labels.cols(); ++idx){
			temp = labels.col(idx);
			temp.maxCoeff(&hat_labels[idx]);
			accuracy += float_type( int(hat_labels[idx]) == int(real_labels[idx]) );
		}
		float_type loss = network.evaluate(test_images, test_labels);
		accuracy /= float_type(test_labels.cols());
		float_type error_rate = 1.0 - accuracy;
		std::cout << "Loss: " << loss;
		std::cout << " Error rate: " << error_rate << std::endl;
		if (visualize && (n_epochs % 10 == 0)){
				int n_images_to_show_per_dim = 10; // This shows n*n images
				int frame_width = 2; // surround images with a frame

			    // Create empty yellow image
			    CImg<float_type> image(n_images_to_show_per_dim*(image_cols + 2*frame_width), 
			    					   n_images_to_show_per_dim*(image_rows + 2*frame_width), 1, 3);
			    image = 0.0;
			    // fill the image
			    float red[]  = { 1.0,0,0 };
			    float green[]  = { 0.0,1.0,0 };
			    for (int i_grid = 0; i_grid < n_images_to_show_per_dim; ++i_grid){
			    	for (int j_grid = 0; j_grid < n_images_to_show_per_dim; ++j_grid){
			    		int sample_idx = dist(generator);

			    		// Draw sample
			    		int top_left_col = j_grid*(2*frame_width + image_cols) + 2;
			    		int top_left_row = i_grid*(2*frame_width + image_rows) + 2;
			    		for (int i = 0; i < image_rows; ++i){
			    			for (int j = 0; j < image_cols; ++j){
			    				int pixel_row = top_left_row + i;
			    				int pixel_col = top_left_col + j;

			    				image(pixel_row, pixel_col, 0, 0) = test_images(i*image_rows + j, sample_idx);
			    				image(pixel_row, pixel_col, 0, 1) = test_images(i*image_rows + j, sample_idx);
			    				image(pixel_row, pixel_col, 0, 2) = test_images(i*image_rows + j, sample_idx);

			    			}
			    		}
	    				if (hat_labels[sample_idx] == real_labels[sample_idx]){
	    					image.draw_text(top_left_row, top_left_col, std::to_string(int(hat_labels[sample_idx])).c_str(), green);
	    				}else{
	    					image.draw_text(top_left_row, top_left_col, std::to_string(int(hat_labels[sample_idx])).c_str(), red);
	    				}
			    	}
			    }
			    
			    image.resize(800, 800);
			    main_disp.display(image);
			    sprintf(title, "Test samples - Loss: %.3f - Error rate: %.3f", loss, error_rate);
			    main_disp.set_title(title);
			    main_disp.show();
		}
	}

	return 0;
}