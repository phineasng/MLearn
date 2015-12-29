#ifndef MLEARN_FULLY_CONNECTED_NET_EXPLORER_INCLUDE
#define MLEARN_FULLY_CONNECTED_NET_EXPLORER_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include "../../ActivationFunction.h"

// STL
#include <random>
#include <chrono>

namespace MLearn{

	namespace NeuralNets{

		namespace FeedForwardNets{

			/*!
			*	\brief		Helper class for performing dropout
			*	\author 	phineasng
			*
			*/
			template < bool DROPOUT = false >
			struct UnitProcessor{
				template < 	ActivationType TYPE,
							typename WeightType,
							typename RNG,
							typename DERIVED,
							typename DERIVED_BOOL,
							typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type >
				static inline void process( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, Eigen::VectorBlock<DERIVED_BOOL> dropped, RNG& rng, std::bernoulli_distribution& b_dist ){
					activations = pre_activations.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<TYPE>::evaluate));
				}
				template < 	typename WeightType,
							typename DERIVED_BOOL,
							typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type >
				static inline void process_derivatives( MLVector<WeightType> derivatives ,Eigen::VectorBlock<DERIVED_BOOL> dropped){}
			};
			template <>
			struct UnitProcessor<true>{
				template < 	ActivationType TYPE,
							typename WeightType,
							typename RNG,
							typename DERIVED,
							typename DERIVED_BOOL,
							typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type >
				static inline void process( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, Eigen::VectorBlock<DERIVED_BOOL> dropped, RNG& rng, std::bernoulli_distribution& b_dist ){
					for ( decltype(activations.size()) idx = 0; idx < activations.size(); ++idx ){
						if ( b_dist(rng) ){
							activations[idx] = WeightType(0);
							pre_activations[idx] = WeightType(0);
							dropped[idx] = true;
						}else{
							activations[idx] = ActivationFunction<TYPE>::evaluate(pre_activations[idx]);
							dropped[idx] = false;
						}
					}
				}
				template < 	typename WeightType,
							typename DERIVED_BOOL,
							typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type >
				static inline void process_derivatives( MLVector<WeightType> derivatives ,Eigen::VectorBlock<DERIVED_BOOL> dropped){
					for ( decltype(derivatives.size()) idx = 0; idx < derivatives.size(); ++idx ){
						if ( dropped[idx] ){
							derivatives[idx] = WeightType(0);
						}
					}
				}
			};

			/*!
			*	\brief		Helper class to compute the derivative of the activation functions.
			*	\details	Useful for the logistic activation since it 
			*				can be computed efficiently without calling 
			*				the activation function derivative
			*				(but knowing the activation function evaluation)
			*	\author 	phineasng
			*
			*/
			template < ActivationType TYPE >
			struct ActivationDerivativeWrapper{
				template < 	typename WeightType,
							typename DERIVED,
							typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type >
				static inline void derive( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, MLVector<WeightType>& derivatives ){
					derivatives = pre_activations.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<TYPE>::first_derivative));
				}
			};
			template <>
			struct ActivationDerivativeWrapper<ActivationType::LOGISTIC>{
				template < 	typename WeightType,
							typename DERIVED,
							typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type >
				static inline void derive( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, MLVector<WeightType>& derivatives ){
					derivatives = activations - activations.cwiseProduct(activations);
				}
			};

			/*! 
			*	\brief		Supporting class to perform the common feedforward fully connected neural nets
			*	\details  	Class implementing the feedforward pass and the backpropagation algorithm for 
			*				feedforward fully connected neural nets.
			*				The weights (and biases) are considered to be stored in a vector, in a 
			*				row major way and sequentially from input to output 
			*				An array of integers is used to describe the layers and interpret the weights vector.
			*	\author 	phineasng
			*/
			template < 	typename WeightType, 
						typename IndexType,
						ActivationType HiddenLayerActivation = ActivationType::LOGISTIC,
						ActivationType OutputLayerActivation = ActivationType::LINEAR,
						typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value , IndexType >::type >
			class FCNetsExplorer{
			public:
				// TYPEDEFS
				typedef WeightType 	Scalar_T;
				typedef IndexType 	Index_T;
				static inline constexpr ActivationType getHiddenLayerType() { return HiddenLayerActivation; }
				static inline constexpr ActivationType getOutputLayerType() { return OutputLayerActivation; }
				// CONSTRUCTOR
				FCNetsExplorer(const MLVector< IndexType >& refLayers):
					layers(refLayers),
					activations( refLayers.array().sum() ),
					pre_activations( refLayers.array().sum() ),
					dropped( refLayers.array().sum() ),
					rng(static_cast<std::mt19937::result_type>(std::chrono::system_clock::now().time_since_epoch().count())),
					b_dist(0.5) {
						MLEARN_ASSERT( layers.size() > 2, "Not a deep architecture: at least 1 hidden layer is needed for the algorithm to work!" );
						MLEARN_WARNING( ( layers.array() > 0 ).all(), "Some layers have zero units!" );
					}
				// FORWARD PASS
				template< 	bool DROPOUT = false,
							typename DERIVED,
							typename DERIVED_2,
							typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value , typename DERIVED::Scalar >::type,
							typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar,WeightType>::value , typename DERIVED::Scalar >::type,
							typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1, DERIVED >::type,
							typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1, DERIVED_2 >::type >
				void forwardpass( const Eigen::MatrixBase< DERIVED >& weights, const Eigen::MatrixBase< DERIVED_2 >& input ){
					MLEARN_ASSERT( ( layers.head(layers.size()-1).dot(layers.tail(layers.size()-1)) + layers.tail(layers.size()-1).array().sum()) == weights.size(), "Number of weights is not consistent with number of units in the layers!" );
					MLEARN_ASSERT( input.size() == layers[0] , "Input should be of the declared size of the first layer!" );
					
					pre_activations.segment(0,layers[0]) = input;
					UnitProcessor<DROPOUT>::template process<ActivationType::LINEAR,WeightType,RNG_TYPE>(pre_activations.segment(0,layers[0]),activations.segment(0,layers[0]),dropped.segment(0,layers[0]),rng,b_dist);

					decltype(activations.size()) offset = layers[0];
					decltype(weights.size()) offset_weights = 0;
					decltype(weights.size()) tmp;
					decltype(activations.size()) idx;
					decltype(idx) idx_m1;

					for ( idx = 1; idx < (layers.size() - 1); ++idx ){

						idx_m1 = idx - 1;
						tmp = layers[idx_m1]*layers[idx];
						pre_activations.segment(offset,layers[idx]) = Eigen::Map<const MLMatrix<WeightType>>( weights.segment(offset_weights,tmp).data(), layers[idx], layers[idx_m1] )*activations.segment(offset-layers[idx_m1],layers[idx_m1]);
						offset_weights += tmp;
						pre_activations.segment(offset,layers[idx]) += weights.segment(offset_weights,layers[idx]);	
						UnitProcessor<DROPOUT>::template process<HiddenLayerActivation,WeightType,RNG_TYPE>(pre_activations.segment(offset,layers[idx]),activations.segment(offset,layers[idx]),dropped.segment(offset,layers[idx]),rng,b_dist);
						offset_weights += layers[idx];
						offset += layers[idx];  
					}
					
					idx_m1 = idx - 1;
					tmp = layers[idx_m1]*layers[idx];
					pre_activations.segment(offset,layers[idx]) = Eigen::Map<const MLMatrix<WeightType>>( weights.segment(offset_weights,tmp).data(), layers[idx], layers[idx_m1] )*activations.segment(offset-layers[idx_m1],layers[idx_m1]);
					offset_weights += tmp;
					pre_activations.segment(offset,layers[idx]) += weights.segment(offset_weights,layers[idx]);	
					activations.segment(offset,layers[idx]) = pre_activations.segment(offset,layers[idx]).unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OutputLayerActivation>::evaluate));
					
				}
				// BACKPROPAGATION	
				/*!
				*	\brief 		Backpropagation algorithm
				*	\details	It assumes that a forward pass has already been performed.
				*/
				template< 	bool DROPOUT = false, 
							typename DERIVED,
							typename DERIVED_2,
							typename DERIVED_3,
							typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value , typename DERIVED::Scalar >::type,
							typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar,WeightType>::value , typename DERIVED::Scalar >::type,
							typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar,typename DERIVED_3::Scalar>::value , typename DERIVED::Scalar >::type,
							typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1, DERIVED >::type,
							typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1, DERIVED_2 >::type,
							typename = typename std::enable_if< DERIVED_3::ColsAtCompileTime == 1, DERIVED_3 >::type >
				void backpropagate( const Eigen::MatrixBase<DERIVED>& weights, const Eigen::MatrixBase<DERIVED_2>& gradient_output, Eigen::MatrixBase<DERIVED_3>& gradient_weights ){
					MLEARN_ASSERT( ( layers.head(layers.size()-1).dot(layers.tail(layers.size()-1)) + layers.tail(layers.size()-1).array().sum()) == weights.size(), "Number of weights is not consistent with number of units in the layers!" );
					MLEARN_ASSERT( gradient_output.size() == layers[layers.size()-1], "Output layer not consistent with the given gradient!" );
					MLEARN_ASSERT( gradient_weights.size() == weights.size(), "Weights size and respective gradient size not compatible!" );
				
					decltype(activations.size()) idx = layers.size()-1;
					decltype(activations.size()) idx_m1 = idx-1;
					decltype(activations.size()) offset = layers.head(idx).array().sum();
					decltype(weights.size()) offset_weights = weights.size() - layers[idx];
					decltype(weights.size()) tmp = layers[idx_m1]*layers[idx];

					// compute delta output layer and assign it to the gradientof the last biases
					ActivationDerivativeWrapper<OutputLayerActivation>::template derive<WeightType>( pre_activations.segment(offset,layers[idx]),activations.segment(offset,layers[idx]),derivatives );
					gradient_weights.segment(offset_weights,layers[idx]) = gradient_output.cwiseProduct(derivatives);
					// ... and get a view of the last delta
					Eigen::Map< const MLVector<WeightType> > delta_p1( gradient_weights.segment(offset_weights,layers[idx]).data(), layers[idx] );

					// update offsets
					offset_weights -= tmp;
					offset -= layers[idx_m1];
					
					// get appropriate view of the weight matrices
					Eigen::Map<MLMatrix<WeightType>> diff_weight_matrix(gradient_weights.segment(offset_weights,tmp).data(),layers[idx],layers[idx_m1]);
					Eigen::Map<const MLMatrix<WeightType>> weight_matrix(weights.segment(offset_weights,tmp).data(),layers[idx],layers[idx_m1]);
					
					// update gradient of last weight matrix
					diff_weight_matrix.noalias() = delta_p1*activations.segment(offset,layers[idx_m1]).transpose();
					
					// update offset weights 
					--idx;

					for ( ; idx > 0; --idx ){
						
						offset_weights -= layers[idx_m1];
						--idx_m1;

						tmp = layers[idx_m1]*layers[idx];

						ActivationDerivativeWrapper<HiddenLayerActivation>::template derive<WeightType>( pre_activations.segment(offset,layers[idx]),activations.segment(offset,layers[idx]),derivatives );
						UnitProcessor<DROPOUT>::template process_derivatives<WeightType>(derivatives,dropped.segment(offset,layers[idx]));
						gradient_weights.segment(offset_weights,layers[idx]) = (weight_matrix.transpose()*delta_p1).cwiseProduct( derivatives );
						new (&delta_p1) Eigen::Map< const MLVector<WeightType> >( gradient_weights.segment(offset_weights,layers[idx]).data(), layers[idx] );
						
						offset_weights -= tmp;
						offset -= layers[idx_m1];

						new (&diff_weight_matrix) Eigen::Map<MLMatrix<WeightType>> (gradient_weights.segment(offset_weights,tmp).data(),layers[idx],layers[idx_m1]);
						new (&weight_matrix) Eigen::Map<const MLMatrix<WeightType>> (weights.segment(offset_weights,tmp).data(),layers[idx],layers[idx_m1]);

						diff_weight_matrix.noalias() = delta_p1*activations.segment(offset,layers[idx_m1]).transpose();
						
					}

				}
				// MODIFIERS
				void setDropoutProbability( WeightType prob ){
					b_dist.param( std::bernoulli_distribution::param_type(prob) );
				}
				// UPDATE INTERNAL MODEL ( external modification of the layers )
				void internal_resize(){
					activations.resize(layers.array().sum());
					pre_activations.resize(layers.array().sum());
					dropped.resize(layers.array().sum());
				}
				// OBSERVERS
				WeightType getDropoutProbability() const{
					return WeightType(b_dist.p());
				}
				// get activations
				const MLVector< WeightType >& getActivations() const{
					return activations;
				}
				// 
			protected:
			private:
				typedef std::mt19937 RNG_TYPE;

				const MLVector< IndexType >& layers;
				MLVector< WeightType > activations;
				MLVector< WeightType > pre_activations;
				MLVector< bool > dropped;
				MLVector<WeightType> derivatives;
				RNG_TYPE rng;
				std::bernoulli_distribution b_dist;
			};

		}

	}

}

#endif