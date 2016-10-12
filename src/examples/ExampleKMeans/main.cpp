#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <MLearn/Core>
#include <MLearn/Clustering/KMeans/KMeans.h>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,Eigen::DontAlignCols,
    ", ", "\n");

// generate N random 2d points into circle with diameter d = 100
Eigen::MatrixXd PointsInCircle(int N)
{
  Eigen::MatrixXd points(2,N);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> rand_angle(0.0,2.0 * M_PI);
  std::uniform_real_distribution<double> rand_radius(0.0,50.0);
  for(int i = 0; i < N; ++i)
  {
    double angle = rand_angle(generator);
    double radius = rand_radius(generator);
    points.col(i) << (double)(50 + radius * cos(angle)),
        (double)(50 + radius * sin(angle));

  }
  return points;
}

// print clusters to file
void printToFile(const Eigen::MatrixXd& data,
    const std::vector<int>& labels, const Eigen::MatrixXd& centroids)
{
  std::ofstream file("clustering.csv");
  if(file.is_open())
  {
    // n_points, n_dims (features), K
    file << data.cols() <<"," <<data.rows()<<","<<centroids.cols()<<std::endl;
    file << data.format(CSVFormat) << std::endl;
    for(unsigned int i = 0; i < labels.size()-1; ++i)
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
  Eigen::MatrixXd points = PointsInCircle(n_points);
 
  // 2) cluster them using kmeans
  KMeans<double,3> clustering(1e3);
  
  // option A) run 10 times with kmeans ++ init, keep the best result
  clustering.run(points,K,10,true);
  
  // or B) manually set centroids to the borders of the circle and run once
  // Eigen::MatrixXd centers(2,K);
  // double radius = 50;
  // for(int i = 0; i < centers.cols(); ++i)
  // {
    // double angle = 2.0 * M_PI *(double)i/(double)centers.cols();
    // centers.col(i) << (double)(50 + radius * cos(angle)),
        // (double)(50 + radius * sin(angle));
  // }
  // clustering.initialize(centers);
  // clustering.runAfterInitialization(points);
 
  // 3) print to csv for visualization with python
  printToFile(points,clustering.getLabels(),
      clustering.getClusterCentroids());
  return 0;	
}