#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <MLearn/Core>
#include <MLearn/Clustering/KMeans/KMeans.h>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,Eigen::DontAlignCols,
    ", ", "\n");

// generate N random 2d points into circle with diameter d = 100
Eigen::MatrixXf PointsInCircle(int N)
{
  Eigen::MatrixXf points(2,N);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> rand_angle(0.0,2.0*M_PI);
  std::uniform_real_distribution<double> rand_radius(0.0,50.0);
  for(size_t i = 0; i < N; ++i)
  {
    double angle = rand_angle(generator);
    double radius = rand_radius(generator);
    points.col(i) << (float)(50 + radius * cos(angle)),
        (float)(50 + radius * sin(angle));
  }
  std::cout<<points.rows()<<", "<<points.cols()<<std::endl;
  return points;
}

// print clusters to file
void printToFile(const Eigen::MatrixXf& data,
    const std::vector<int>& labels, const Eigen::MatrixXf& centroids)
{
  std::ofstream file("clustering.csv");
  if(file.is_open())
  {
    // n_points, n_dims (features), K
    file << data.cols() <<"," <<data.rows()<<","<<centroids.cols()<<std::endl;
    file << data.format(CSVFormat) << std::endl;
    for(int i = 0; i < labels.size()-1; ++i)
      file << labels[i] << ",";
    file << labels.back() << std::endl;
    file <<centroids.format(CSVFormat) <<std::endl;
    file.close();
  }
}

int main(int argc, char* argv[]){

	using namespace MLearn;
	using namespace Clustering;

	if (argc < 3) { 
		// 3 arguments: program name, source file and destination file
		std::cerr << "Usage: " << argv[0] << " N_POINTS N_CLUSTERS "
		 << std::endl;
		return 1;
	}
  
  // 1) generate random points into circle
  int n_points = std::atoi(argv[1]);
  int K = std::atoi(argv[2]);
  Eigen::MatrixXf points = PointsInCircle(n_points);
  // 2) cluster them using kmeans
  KMeans clustering(K, n_points, points.rows(), 100);
  clustering.run(points, 10);
  // 3) display results and stats (e.g. time)
  printToFile(points,clustering.getLabels(),
      clustering.getClusterCentroids());
  return 0;	
}
