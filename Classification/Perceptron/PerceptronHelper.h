#ifndef MLEARN_PERCEPTRON_HELPER_
#define MLEARN_PERCEPTRON_HELPER_

// Core functionalities and definitions
#include <MLearn/Core>

namespace MLearn{

	namespace Classification {

		// Helper Class to pass infos from
		// the perceptron class to the concrete 
		// implementations of the training algorithm  
		template < typename WeightType >
		class PerceptronInfo{
		public:
			PerceptronInfo( MLVector< WeightType >& perceptron_weights, const WeightType& perceptron_tolerance, const WeightType& perceptron_learning_rate, const uint32_t& perceptron_max_iter ): 
				weights(perceptron_weights), 
				tolerance(perceptron_tolerance),
				learningRate(perceptron_learning_rate),
				max_iter(perceptron_max_iter)
				{}
			MLVector< WeightType >& weights;
			const WeightType& 		tolerance;
			const WeightType& 		learningRate;
			const uint32_t&			max_iter;
		};



	}

}

#endif