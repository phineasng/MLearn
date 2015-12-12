#ifndef MLEARN_PERCEPTRON_CLASS_
#define MLEARN_PERCEPTRON_CLASS_

// Core functionalities and definitions
#include <MLearn/Core>

// Perceptron utilities
#include "PerceptronHelper.h"
#include "PerceptronTraining.h"

// STL includes
#include <type_traits>
#include <cassert>
#include <cmath>

namespace MLearn{

	namespace Classification{

		/*!
		*
		*	\brief 		Perceptron class.
		*	\details 	Implementation of the perceptron for binary classification 
		*				of linearly separable classes.
		*				Implementation based on "http://www.cs.utah.edu/~piyush/teaching/8-9-print.pdf"
		* 	\author 	phineasng
		*
		*/
		template < 	typename WeightType, 
					typename ClassType, 
					typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type,
					typename = typename std::enable_if< std::is_integral<ClassType>::value && std::is_unsigned<ClassType>::value , ClassType >::type >
		class Perceptron {
		protected:
			MLVector< WeightType > 	_weights;
			WeightType 				_tolerance 			= 1e-5;
			WeightType 				_learning_rate		= 1.;
			uint32_t				_max_iter			= 1000;
		public:
			
			// CONSTRUCTORS
			// -- default: default construct everything
			Perceptron() = default;
			// -- integer argument: number of features (without accounting for the bias)
			template< typename U_INT, typename = typename std::enable_if< std::is_integral< U_INT >::value && std::is_unsigned< U_INT >::value, U_INT>::type >
			explicit Perceptron( const U_INT& N ): _weights(N+1) {}
			// -- const reference MLVector
			explicit Perceptron( const MLVector< WeightType >& refWeights ): _weights(refWeights) {}
			// -- rvalue MLVector
			explicit Perceptron( MLVector< WeightType >&& refWeights ): _weights(std::move(refWeights)) {}
			// -- copy constructor
			Perceptron( const Perceptron<WeightType,ClassType>& refPerceptron ): _weights(refPerceptron._weights), _tolerance(refPerceptron._tolerance), _learning_rate(refPerceptron._learning_rate), _max_iter(refPerceptron._max_iter) {}
			// -- move constructor
			Perceptron( Perceptron<WeightType,ClassType>&& refPerceptron ): _weights(std::move(refPerceptron._weights)), _tolerance(refPerceptron._tolerance), _learning_rate(refPerceptron._learning_rate), _max_iter(refPerceptron._max_iter) {}
			
			// ASSIGNMENT OPERATORS
			// -- copy assignment
			Perceptron& operator=( const Perceptron<WeightType,ClassType>& refPerceptron) { _weights = refPerceptron._weights; _tolerance = refPerceptron._tolerance; _learning_rate = refPerceptron._learning_rate; _max_iter = refPerceptron._max_iter; return *this; }
			// -- move assignment
			Perceptron& operator=( Perceptron<WeightType,ClassType>&& refPerceptron) { _weights = std::move(refPerceptron._weights); _tolerance = refPerceptron._tolerance; _learning_rate = refPerceptron._learning_rate; _max_iter = refPerceptron._max_iter; return *this; }

			// MODIFIERS
			void setWeights( const MLVector< WeightType >& refWeights ) { _weights = refWeights; }
			void setWeights( MLVector< WeightType >&& refWeights ) { _weights = std::move(refWeights); }
			template< typename U_INT, typename = typename std::enable_if< std::is_integral< U_INT >::value && std::is_unsigned< U_INT >::value, U_INT>::type >
			void setSize( const U_INT& N ) { _weights.resize(N+1); }
			template< typename U_INT, typename = typename std::enable_if< std::is_integral< U_INT >::value && std::is_unsigned< U_INT >::value, U_INT>::type >
			void setMaxIter( const U_INT& N ) { _max_iter = (uint32_t)N; }
			void initializeZero() { _weights = std::move(MLVector< WeightType >::Zero(_weights.size())); }
			void initializeRandom( WeightType scaleFactor = WeightType(1) ) { _weights = std::move( scaleFactor*MLVector< WeightType >::Random(_weights.size()) ); }
			void setTolerance( WeightType new_tolerance ) { MLEARN_ASSERT( !std::signbit(new_tolerance), "invalid value. Tolerance must be positive!" ); _tolerance = new_tolerance; }
			void setLearningRate( WeightType new_learning_rate ) { MLEARN_ASSERT( !std::signbit(new_learning_rate), "invalid value. Learning rate must be positive!" ); _learning_rate = new_learning_rate; }

			// OBSERVERS
			const MLVector< WeightType >& getWeights() const { return _weights; }
			decltype( _weights.size() ) getWeightsSize() const { return _weights.size(); }
			WeightType getTolerance() const { return _tolerance; }
			WeightType getLearningRate() const { return _learning_rate; }
			uint32_t getMaxIter() const { return _max_iter; }

			// TRAINING
			/*!
			*	\brief 		Training algorithm for perceptrons
			*	\details 	Different algorithms are implemented. 
			*				Check PerceptronTraining.h to see which ones are available
			*				Given N samples, the data sample is represented by a 
			*				column of the data matrix. Thus, given that the data is represented
			*				by R^M vectors, the data matrix is MxN. 
			*				'Labels' is a R^N vector storing the classification of the above mentioned data.
			*				The labels are assumed to have labels 0 (negative examples) and 1 (positive examples).
			*				The BATCH algorithm use the whole dataset to learn the parameters, while the ONLINE algorithms
			*				process each sample (row) once and once at a time.
			*				NOTE: the algorithms will work on the last value of the weights vector. To reset the perceptron,
			*						use the reset functions.
			*				 
			*/
			template < PerceptronTraining MODE >
			void train( const MLMatrix< WeightType >& dataMatrix, const MLVector< ClassType >& labels ){
				PerceptronInfo< WeightType > infos(_weights,_tolerance,_learning_rate,_max_iter); 
				PerceptronAlgorithm<MODE>::train(dataMatrix, labels, infos); 
			}

			template < PerceptronTraining MODE >
			void train( const MLMatrix< WeightType >& dataMatrix, ClassType label ){
				MLVector< ClassType > labels(1);
				labels[0] = label;
				PerceptronInfo< WeightType > infos(_weights,_tolerance,_learning_rate,_max_iter);
				PerceptronAlgorithm<MODE>::train(dataMatrix, labels, infos); 
			}

			/*!
			*	\brief 		Classification function.
			*	\details 	Used the trained perceptron, to predict the label
			*
			*/
			// TODO: find a nice way to optimize it
			MLVector<ClassType> classify( const MLMatrix< WeightType >& dataToClassify) const{
				MLVector<ClassType> labels(dataToClassify.rows());
				auto dim_data = dataToClassify.cols();
				for ( decltype(labels.size()) idx = 0; idx < labels.size(); ++idx ){
					labels[idx] = ml_zero_one_sign< ClassType, WeightType >( _weights[0] + dataToClassify.row(idx).dot(_weights.tail(dim_data)) );
				}
				return labels;
			}

		private:
		};

	} // End Classification namespace

} // End MLearn namespace

#endif 