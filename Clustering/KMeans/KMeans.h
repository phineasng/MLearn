#ifndef MLEARN_KMEANS_CLASS_
#define MLEARN_KMEANS_CLASS_

#include <iostream>
#include <math.h>
#include <time.h>
#include <cassert>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <type_traits>

#include <MLearn/Core>
#include <MLearn/Utility/VerbosityLogger.h>
#include <MLearn/Clustering/DistanceFunction.h>

namespace MLearn{

  namespace Clustering{

    /*!
     *  \brief    KMeans class
    *   \details  k-means implementation with k-means ++ initialization.
    *   \author   frenaut
    *
    */
    template < typename SCALAR_TYPE, ushort VERBOSITY_REF = 0, 
             typename DistanceFunction = SquaredEuclidianDistance >
    class KMeans
    {
    private:

      const int max_iterations_;
      std::vector<int> labels_;
      MLMatrix < SCALAR_TYPE> centroids_;
      bool CENTROIDS_SET_ = false;

    public:

      // constructor
      KMeans(int max_iterations = 1e3):
      max_iterations_(max_iterations)
      {
        srand(time(NULL)); // provide seed to randomizer in k-means++
      }

    private:

      // return ID of nearest centroid 
      int getIDNearestCentroid(const Eigen::Ref< const MLVector< SCALAR_TYPE > > point,
          const int& k_max)
      {
        SCALAR_TYPE dist;
        SCALAR_TYPE min_dist = INFINITY;
        int id_cluster_center = 0;
        //assert(k_max <= centroids_.cols());
        for(int k = 0; k < k_max; ++k)
        {	
          dist = DistanceFunction::compute(point,centroids_.col(k));
          if(dist < min_dist)
          {
            min_dist = dist;
            id_cluster_center = k;
          }
        }
        return id_cluster_center;
      }

      // initialize K random centroids
      void initializeRandom(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input, 
          int K)
      {
        std::fill(labels_.begin(), labels_.end(), -1);
        centroids_.setZero();

        // choose k centers uniformly at random from among the data points
        for(int k = 0; k < K; ++k)
          centroids_.col(k) = input.col((rand() % labels_.size()));

        CENTROIDS_SET_ = true;
      }
      
      // initialize centroids with kmeans ++
      void initializeKPP(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input)
      {
        std::fill(labels_.begin(), labels_.end(), -1);
        centroids_.setZero();

        // choose one center uniformly at random from among the data points
        centroids_.col(0) = input.col((rand() % labels_.size()));

        for(unsigned int k = 1; k < centroids_.cols(); ++k)
        {	
          // for each point, compute squared distance and put as probability weight
          std::vector<SCALAR_TYPE> weight_sum(labels_.size());
          
          SCALAR_TYPE weight_sum_i = 0;
          for(unsigned int i = 0; i < labels_.size(); ++i)
          {
            // get squared dist between point and nearest center
            int nearest_centroid_id = getIDNearestCentroid(input.col(i), k);
            weight_sum_i += DistanceFunction::compute(input.col(i),
                          centroids_.col(nearest_centroid_id));
            weight_sum[i] = weight_sum_i;
          }
          // generate a random float from a 
          // uniform distribution from 0 to weights_sum(end)
          SCALAR_TYPE p_rand = static_cast <SCALAR_TYPE> (rand()) / (static_cast <SCALAR_TYPE>
                      (RAND_MAX / weight_sum[labels_.size() - 1]));
                      
          // lowest idx such that p_rand < weight_sum[idx]
          int idx = std::upper_bound(weight_sum.begin(), weight_sum.end(), p_rand)
            - weight_sum.begin();
          // take the point chosen with the probability distribution as cluster seed
          centroids_.col(k) = input.col(int(idx));
        }	

        CENTROIDS_SET_ = true;
      }

      // update labels to nearest centroid
      void updateLabels(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input)
      {
        for(unsigned int i = 0; i < labels_.size(); ++i)
          labels_[i] = getIDNearestCentroid(input.col(i),centroids_.cols());
      }

      // update centroid positions according to labels
      void updateCentroids(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input)
      {
        centroids_.setZero();
        std::vector<int> counter(centroids_.cols(),0);
        for(unsigned int i = 0; i < labels_.size(); ++i)
        {
          int k = labels_[i];
          centroids_.col(k) += input.col(i);
          counter[k] ++; 
        }
        for(int k = 0; k < centroids_.cols(); ++k)
        {
          MLEARN_ASSERT(counter[k] > 0,"Empty cluster!");
          centroids_.col(k) /= counter[k];
        }
      }


    public:

      // get inertia to estimate how good clustering performs
      SCALAR_TYPE getInertia(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input)
      {
        SCALAR_TYPE sum = 0;
        for(unsigned int i = 0; i < labels_.size(); ++i)
          sum += DistanceFunction::compute(input.col(i),centroids_.col(labels_[i]));
        return sum;
      }

       // manually initialize centroids with indexes or directly assign matrix
      void initialize(const std::vector<int>& idx,
          const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input)
      {
       centroids_.resize(input.rows(),idx.size());
      
       for (unsigned int i = 0; i < idx.size(); ++i)
         centroids_.col(i) = input.col(idx[i]);

       CENTROIDS_SET_ = true;
      }

      void initialize(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > centroids)
      {
        centroids_ = centroids;
        CENTROIDS_SET_ = true;
      }


      // run k-means algorithm with initialized centroids
      void runAfterInitialization(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input)
      {	
        labels_ = std::vector<int>(input.cols(), -1);
        MLEARN_ASSERT(CENTROIDS_SET_ = true, "Cluster centroids are not initialized!");
        int iter = 1;
        bool change = true;
        while(change && iter < max_iterations_)
        {	
          change = false;	
          for (unsigned int i = 0; i < labels_.size(); ++i)
          {
            int new_label = getIDNearestCentroid(input.col(i),centroids_.cols());
            if (new_label != labels_[i])
            {
              labels_[i] = new_label;
              change = true; 
            }
          }
          if (change == true)
            updateCentroids(input);
          iter++;
        }
        Utility::VerbosityLogger<2,VERBOSITY_REF>::log(
            "KMeans Clustering converged after " );
        Utility::VerbosityLogger<2,VERBOSITY_REF>::log(iter);
        Utility::VerbosityLogger<2,VERBOSITY_REF>::log(" iterations" );

      }

      // run N times with different centroid seeds and keep the best result
      void run(const Eigen::Ref<const MLMatrix <SCALAR_TYPE> > input, int K, 
          int N = 10, bool USE_KMEANS_PP = true)
      {
        MLEARN_ASSERT(K <= input.cols(), 
            "The number of clusters has to be smaller than the amount of points.");
        centroids_.resize(input.rows(), K);
        labels_ = std::vector<int>(input.cols(), -1);
 
        MLMatrix< SCALAR_TYPE > best_centroids(centroids_.rows(),centroids_.cols());
        SCALAR_TYPE min_inertia = INFINITY;
        Utility::VerbosityLogger<1,VERBOSITY_REF>::log(
            "====== STARTING: KMeans Clustering  ======\n" );

        Utility::VerbosityLogger<1,VERBOSITY_REF>::log(" Running KMeans ");
        Utility::VerbosityLogger<1,VERBOSITY_REF>::log(N);
        Utility::VerbosityLogger<1,VERBOSITY_REF>::log(" times.\n");

        for(int n = 0; n < N; ++n)
        {
          CENTROIDS_SET_ = false;

          // initialize using kmeans++ (default) or random
          if(USE_KMEANS_PP) 
            initializeKPP(input);
          else
            initializeRandom(input,K);

          // run k means and get inertia
          Utility::VerbosityLogger<2,VERBOSITY_REF>::log( n );
          Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") " );
          runAfterInitialization(input);
          SCALAR_TYPE inertia = getInertia(input);
          Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ", Final inertia =  " );
          Utility::VerbosityLogger<2,VERBOSITY_REF>::log( inertia );
          Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );

          // if inertia is min, store final centroids
          if(inertia < min_inertia)
          {
            min_inertia = inertia;
            best_centroids = getClusterCentroids();
          }
        }
        // save centroids which gave the lowest inertia
        centroids_ = best_centroids;
        // update labels accordingly
        updateLabels(input);
        Utility::VerbosityLogger<1,VERBOSITY_REF>::log( 
            "====== DONE: KMeans Clustering  ======\n" );

      }
      
      // get cluster label for each sample
      std::vector<int> getLabels()
      {
        return(labels_);
      }
      
      // get cluster centroids
      const MLMatrix< SCALAR_TYPE >& getClusterCentroids()
      {
        return(centroids_);
      }

    };// class KMeans

  }// End Clustering namespace

}// End MLearn namespace

#endif
