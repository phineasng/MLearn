#ifndef MLEARN_KMEANS_CLASS_
#define MLEARN_KMEANS_CLASS_

#include <iostream>
#include <math.h>
#include <cassert>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <MLearn/Core>
#include <MLearn/Utility/VerbosityLogger.h>

// squared euclidian distance between two points
float getDistance(const Eigen::VectorXf& a, const Eigen::VectorXf& b)
{
  return (a - b).squaredNorm();
}

namespace MLearn{

  namespace Clustering{

    /*!
     *  \brief    KMeans class
    *   \details  k-means implementation with k-means ++ initialization.
    *   \author   frenaut
    *
    */

    class KMeans
    {
    private:
      const int K_; // number of clusters
      const int n_dims_, n_points_, max_iterations_;
      std::vector<int> labels_;
      Eigen::MatrixXf centroids_;

      // return ID of nearest centroid 
      int getIDNearestCentroid(const Eigen::VectorXf& point, const int& k_max)
      {
        float dist;
        float min_dist = INFINITY;
        int id_cluster_center = 0;
        //assert(k_max <= K_);
        for(size_t k = 0; k < k_max; ++k)
        {	
          dist = getDistance(point,centroids_.col(k));
          if(dist < min_dist)
          {
            min_dist = dist;
            id_cluster_center = k;
          }
        }
        return id_cluster_center;
      }

      // initialize centroids with kmeans ++
      void initializeCentroids(const Eigen::MatrixXf& points)
      {
        std::fill(labels_.begin(), labels_.end(), -1);
        centroids_.setZero();

        // choose one center uniformly at random from among the data points
        centroids_.col(0) = points.col((rand() % n_points_));

        for(size_t k = 1; k < K_; ++k)
        {	
          // for each point, compute squared distance and put as probability weight
          std::vector<float> weight_sum(n_points_);
          
          float weight_sum_i = 0;
          for(size_t i = 0; i < n_points_; ++i)
          {
            // get squared dist between point and nearest center
            int nearest_centroid_id = getIDNearestCentroid(points.col(i), k);
            weight_sum_i += getDistance(points.col(i),
                          centroids_.col(nearest_centroid_id));
            weight_sum[i] = weight_sum_i;
          }
          // generate a random float from a 
          // uniform distribution from 0 to weights_sum(end)
          float p_rand = static_cast <float> (rand()) / (static_cast <float>
                      (RAND_MAX / weight_sum[n_points_ - 1]));
                      
          // lowest idx such that p_rand < weight_sum[idx]
          int idx = std::upper_bound(weight_sum.begin(), weight_sum.end(), p_rand)
            - weight_sum.begin();
          // take the point chosen with the probability distribution as cluster seed
          centroids_.col(k) = points.col(int(idx));
        }	
      }

      // update labels to nearest centroid
      void updateLabels(const Eigen::MatrixXf& points)
      {
        for(size_t i = 0; i < n_points_; ++i)
          labels_[i] = getIDNearestCentroid(points.col(i),K_);
      }

      // update centroid positions according to labels
      void updateCentroids(const Eigen::MatrixXf& points)
      {
        Eigen::MatrixXf centroids_ = Eigen::MatrixXf::Zero(n_dims_, K_);

        std::vector<int> counter(K_,0);
        for(size_t i = 0; i < n_points_; ++i)
        {
          int k = labels_[i];
          centroids_.col(k) += points.col(i);
          counter[k] ++; 
        }
        for(size_t k = 0; k < K_; ++k)
        {
          assert(counter[k] > 0);
          centroids_.col(k) /= counter[k];
        }
      }

      // run k-means algorithm after k-means ++ initialization
      void runAfterInitialization(const Eigen::MatrixXf& points)
      {	
        int iter = 1;
        bool change = true;
        while(change && iter < max_iterations_)
        {	
          change = false;	
          for (size_t i = 0; i < n_points_; ++i)
          {
            int new_label = getIDNearestCentroid(points.col(i),K_);
            if (new_label != labels_[i])
            {
              labels_[i] = new_label;
              change = true; 
            }
          }
          if (change == true)
            updateCentroids(points);
          iter++;
        }
      }


    public:

      // constructor
      KMeans(const int& K, const int& n_points, const int& n_dims, 
      const int& max_iterations):
      K_(K), n_points_(n_points), n_dims_(n_dims), max_iterations_(max_iterations)
      {
        assert(K_ <= n_points);
        centroids_ = Eigen::MatrixXf(n_dims_, K_);
        labels_ = std::vector<int>(n_points_, -1);
      }


      // get inertia to estimate how good clustering performs
      float getInertia(const Eigen::MatrixXf& points)
      {
        float sum = 0;
        for(size_t i = 0; i < n_points_; ++i)
          sum += getDistance(points.col(i),centroids_.col(labels_[i]));
        return sum;
      }

      // run N times with different centroid seeds and keep the best result
      void run(const Eigen::MatrixXf& points, int N = 1)
      {
        Eigen::MatrixXf best_centroids(K_, n_dims_);
        float min_inertia = INFINITY;
        // Utility::VerbosityLogger<1,VERBOSITY_REF>::log(
            // "====== STARTING: KMeans Clustering  ======\n" );
// 
        // Utility::VerbosityLogger<1,VERBOSITY_REF>::log(" Running KMeans ");
        // Utility::VerbosityLogger<1,VERBOSITY_REF>::log(N);
        // Utility::VerbosityLogger<1,VERBOSITY_REF>::log(" times.\n");
// 
        for(size_t n = 0; n < N; ++n)
        {
          // initialize using k++
          initializeCentroids(points);
          // run k means
          runAfterInitialization(points);
          // if inertia is min, store final centroids
          float inertia = getInertia(points);
          // Utility::VerbosityLogger<2,VERBOSITY_REF>::log( n );
          // Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Final inertia =  " );
          // Utility::VerbosityLogger<2,VERBOSITY_REF>::log( inertia );
          // Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );
// 
          if(inertia < min_inertia)
          {
            min_inertia = inertia;
            best_centroids = getClusterCentroids();
          }
        }
        // save centroids which gave the lowest inertia
        centroids_ = best_centroids;
        // update labels accordingly
        updateLabels(points);
        // Utility::VerbosityLogger<1,VERBOSITY_REF>::log( 
            // "====== DONE: KMeans Clustering  ======\n" );

      }

      // get cluster label for each sample
      std::vector<int> getLabels()
      {
        return(labels_);
      }
      
      // get cluster centroids
      Eigen::MatrixXf getClusterCentroids()
      {
        return(centroids_);
      }

    };// class KMeans

  }// End Clustering namespace

}// End MLearn namespace

#endif
