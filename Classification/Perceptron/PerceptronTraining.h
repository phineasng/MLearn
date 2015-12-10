#ifndef MLEARN_PERCEPTRON_TRAINING_ALGS_
#define MLEARN_PERCEPTRON_TRAINING_ALGS_

// MLearn Base structs and funcs
#include <MLearn/Core>

// Perceptron Helper Structures and Routines
#include "PerceptronHelper.h"
#include "Perceptron.h"

namespace MLearn{

	namespace Classification{

		// Enum class for training algorithms
		enum class PerceptronTraining {
			BATCH,
			BATCH_AVERAGE,	
			ONLINE
		};

		// Concrete Algorithm
		template < PerceptronTraining MODE >
		struct PerceptronAlgorithm{
			template < typename WeightType, typename ClassType >
			static void train( const MLMatrix< WeightType >& dataMatrix, const MLVector< ClassType >& labels, PerceptronInfo< WeightType >& p ){}
		};

		// BATCH algorithm
		template <> template < typename WeightType, typename ClassType >
		void PerceptronAlgorithm< PerceptronTraining::BATCH >::train( const MLMatrix< WeightType >& dataMatrix, const MLVector< ClassType >& labels, PerceptronInfo< WeightType >& p ) { 
			// DEBUG mode: check if the input data, labels and weights vector have consistent and meaningful dimensions 
			MLEARN_ASSERT( dataMatrix.cols() == labels.size(), "dimensions not consistent. Check the data matrix and the associated labels vector!" ); 
			MLEARN_ASSERT( dataMatrix.rows() > 0 , "dimensions not valid. Data matrix must have more than 0 rows!" );
			MLEARN_ASSERT( p.weights.size() == (dataMatrix.rows() + 1), "dimensions not consistent. Check the data matrix and the weights vector size!" ); 

			p.weights = std::move(MLVector<WeightType>::Zero(p.weights.size()));

			auto N_samples = dataMatrix.cols();
			WeightType inv_N_samples = WeightType(1)/((WeightType)N_samples);
			auto dim_data = dataMatrix.rows();
			ClassType predicted_label; 
			WeightType update_factor = 1.;
			MLVector< WeightType > delta(MLVector<WeightType>::Constant(p.weights.size(),p.tolerance));

			for (uint32_t i = 0; (i < p.max_iter) && (delta.norm()>p.tolerance); ++i){
				delta = std::move(MLVector<WeightType>::Zero(p.weights.size()));
				for ( decltype(N_samples) currIdx = 0 ; currIdx < N_samples ; ++currIdx ){

					predicted_label = ml_zero_one_sign< ClassType, WeightType >( p.weights[0] + dataMatrix.col( currIdx ).dot( p.weights.tail( dim_data ) ) );
					// if misclassification, correct
					if ( predicted_label != labels[currIdx] ){
						update_factor = ((WeightType)predicted_label) - ((WeightType)labels[currIdx]);
						delta[0] += update_factor;
						delta.tail(dim_data) += update_factor*dataMatrix.col(currIdx);
					}

				}
				delta *= inv_N_samples;
				p.weights -= p.learningRate*delta;
			}

		} /* END implementation of the batch algorithm */

		// Averaged BATCH algorithm 
		template <> template < typename WeightType, typename ClassType >
		void PerceptronAlgorithm< PerceptronTraining::BATCH_AVERAGE >::train( const MLMatrix< WeightType >& dataMatrix, const MLVector< ClassType >& labels, PerceptronInfo< WeightType >& p ) {
			// DEBUG mode: check if the input data, labels and weights vector have consistent and meaningful dimensions 
			MLEARN_ASSERT( dataMatrix.cols() == labels.size(), "dimensions not consistent. Check the data matrix and the associated labels vector!" ); 
			MLEARN_ASSERT( dataMatrix.rows() > 0 , "dimensions not valid. Data matrix must have more than 0 rows!" );
			MLEARN_ASSERT( p.weights.size() == (dataMatrix.rows() + 1), "dimensions not consistent. Check the data matrix and the weights vector size!" ); 

			p.weights = std::move(MLVector<WeightType>::Zero(p.weights.size()));
			auto cached_weights = p.weights;

			auto N_samples = dataMatrix.cols();
			auto dim_data = dataMatrix.rows();
			ClassType predicted_label; 
			WeightType update_factor = 1.;

			int32_t total_steps = p.max_iter*N_samples;
			int32_t step = total_steps;

			for (uint32_t i = 0; i < p.max_iter; ++i){
				for ( decltype(N_samples) currIdx = 0 ; currIdx < N_samples ; ++currIdx ){

					predicted_label = ml_zero_one_sign< ClassType, WeightType >( p.weights[0] + dataMatrix.col( currIdx ).dot( p.weights.tail( dim_data ) ) );
					// if misclassification, correct
					if ( predicted_label != labels[currIdx] ){
						update_factor = (( ((WeightType)labels[currIdx]) - ((WeightType)predicted_label) ));
						p.weights[0] += update_factor;
						p.weights.tail(dim_data) += update_factor*dataMatrix.col(currIdx);
						cached_weights[0] += ((WeightType)step)*update_factor;
						cached_weights.tail(dim_data) += ((WeightType)step)*update_factor*dataMatrix.col(currIdx);
					}
					--step;
				}
			}
			p.weights = cached_weights/((WeightType)total_steps);
		} /* END implementation of the averaged online algorithm */

		// Standard ONLINE algorithm - can be updated with new data
		template <> template < typename WeightType, typename ClassType >
		void PerceptronAlgorithm< PerceptronTraining::ONLINE >::train( const MLMatrix< WeightType >& dataMatrix, const MLVector< ClassType >& labels, PerceptronInfo< WeightType >& p ) {
			// DEBUG mode: check if the input data, labels and weights vector have consistent and meaningful dimensions 
			MLEARN_ASSERT( dataMatrix.cols() == labels.size(), "dimensions not consistent. Check the data matrix and the associated labels vector!" ); 
			MLEARN_ASSERT( dataMatrix.rows() > 0 , "dimensions not valid. Data matrix must have more than 0 rows!" );
			MLEARN_ASSERT( p.weights.size() == (dataMatrix.rows() + 1), "dimensions not consistent. Check the data matrix and the weights vector size!" ); 

			auto N_samples = dataMatrix.cols();
			auto dim_data = dataMatrix.rows();
			ClassType predicted_label; 
			WeightType update_factor = 1.;

			for ( decltype(N_samples) currIdx = 0 ; currIdx < N_samples ; ++currIdx ){

				predicted_label = ml_zero_one_sign< ClassType, WeightType >( p.weights[0] + dataMatrix.col( currIdx ).dot( p.weights.tail( dim_data ) ) );
				// if misclassification, correct
				if ( predicted_label != labels[currIdx] ){
					update_factor = p.learningRate*(( ((WeightType)labels[currIdx]) - ((WeightType)predicted_label) ));
					p.weights[0] += update_factor;
					p.weights.tail(dim_data) += update_factor*dataMatrix.col(currIdx);
				}

			}

		} /* END implementation of the standard online algorithm */

	}

}

#endif