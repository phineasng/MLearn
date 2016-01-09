#include <iostream>
#include <string>
#include <fstream>

#include <MLearn/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#define FILENAME "trained_weights.txt"
#define SAVE_PATH "./"

int main(int argc, char* argv[]){

	using namespace MLearn;
	using namespace cv;

	uint N_layers = 3;
	uint N_hidden = 600;

	// Setting layers
	MLVector<uint> layers(N_layers);
	layers << 1024,N_hidden,1024;

	// import weights
  	std::ifstream myfile;
  	myfile.open(FILENAME, std::ifstream::in );
  	MLVector< double > trained_weights(layers.head(N_layers-1).dot(layers.tail(N_layers-1)) + layers.tail(N_layers-1).array().sum());
  	if (myfile.is_open()){
  		double num;
  	  	int i = 0;
  	  	while ( myfile >> num ){
  	  		trained_weights[i++] = num;
  	  	}
  	}else{
  		std::cerr << "Problem in opening the trained_weights file. Check the path to the file." << std::endl;
  		return -1;
  	}
  	myfile.close();
	

	// windows
	namedWindow( "Features", WINDOW_OPENGL );

	MLMatrix< double > weight_matrix = Eigen::Map< MLMatrix< double > >( trained_weights.data(),N_hidden,1024 );
	weight_matrix.transposeInPlace();

	Mat_<double> image_from_eigen;
	MLMatrix<double> eigen_image(32,32);
	Mat image;

	//std::string save_path(SAVE_PATH);
	//std::string extension(".jpg");

	for (uint i = 0; i < N_hidden; ++i){

		eigen_image = Eigen::Map<MLMatrix<double>>(weight_matrix.data()+i*1024,32,32);
		eigen_image /= eigen_image.array().abs2().sum();
		eigen2cv(eigen_image,image_from_eigen);
		normalize(image_from_eigen,image,0,255,NORM_MINMAX,CV_8UC1);
		imshow( "Features", image );
		//imwrite( save_path + std::to_string(i) + extension ,image);
		cv::waitKey();
	
	}

    destroyWindow("Features");
	

 	return 0;
}