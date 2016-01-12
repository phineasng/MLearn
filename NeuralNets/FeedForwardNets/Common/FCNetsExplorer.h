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
							typename DERIVED_BOOL >
				static inline void process( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, Eigen::VectorBlock<DERIVED_BOOL> dropped, RNG& rng, std::bernoulli_distribution& b_dist, const WeightType& inv_p  ){
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value,"The scalar type has to be floating point!");
					static_assert( std::is_floating_point<WeightType>::value,"The declared scalar type and the scalar type of the input vector have to be the same!");
					activations = pre_activations.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<TYPE>::evaluate));
				}
				template < 	typename WeightType,
							typename DERIVED_BOOL >
				static inline void process_derivatives( MLVector<WeightType>& derivatives ,Eigen::VectorBlock<DERIVED_BOOL> dropped){static_assert( std::is_floating_point<WeightType>::value,"The scalar type has to be floating point!");}
			};
			template <>
			struct UnitProcessor<true>{
				template < 	ActivationType TYPE,
							typename WeightType,
							typename RNG,
							typename DERIVED,
							typename DERIVED_BOOL >
				static inline void process( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, Eigen::VectorBlock<DERIVED_BOOL> dropped, RNG& rng, std::bernoulli_distribution& b_dist, const WeightType& inv_p ){
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value,"The scalar type has to be floating point!");
					static_assert( std::is_floating_point<WeightType>::value,"The declared scalar type and the scalar type of the input vector have to be the same!");
					for ( decltype(activations.size()) idx = 0; idx < activations.size(); ++idx ){
						if ( b_dist(rng) ){
							activations[idx] = ActivationFunction<TYPE>::evaluate(pre_activations[idx])*inv_p;
							dropped[idx] = false;
						}else{
							activations[idx] = WeightType(0);
							pre_activations[idx] = WeightType(0);
							dropped[idx] = true;
						}
					}
				}
				template < 	typename WeightType,
							typename DERIVED_BOOL >
				static inline void process_derivatives( MLVector<WeightType>& derivatives ,Eigen::VectorBlock<DERIVED_BOOL> dropped){
					static_assert( std::is_floating_point<WeightType>::value,"The scalar type has to be floating point!");
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
				template < 	typename DERIVED >
				static inline void derive( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, MLVector<typename DERIVED::Scalar>& derivatives ){
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value,"The scalar type has to be floating point!");
					derivatives = pre_activations.unaryExpr(std::pointer_to_unary_function<typename DERIVED::Scalar,typename DERIVED::Scalar>(ActivationFunction<TYPE>::first_derivative));
				}
			};
			template <>
			struct ActivationDerivativeWrapper<ActivationType::LOGISTIC>{
				template < 	typename DERIVED >
				static inline void derive( Eigen::VectorBlock<DERIVED> pre_activations, Eigen::VectorBlock<DERIVED> activations, MLVector<typename DERIVED::Scalar>& derivatives ){
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value,"The scalar type has to be floating point!");
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
						ActivationType OutputLayerActivation = ActivationType::LINEAR >
			class FCNetsExplorer{
			public:
				static_assert( std::is_floating_point<WeightType>::value, "The weights type has to be floating point!" );
				static_assert( std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, "The index type has to be integer and unsigned!" );
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
						MLEARN_ASSERT( layers.size() > 1, "Not a deep architecture: at least 2 layers are needed for the algorithm to work!" );
						MLEARN_WARNING( ( layers.array() > 0 ).all(), "Some layers have zero units!" );
						inv_p = WeightType(1)/b_dist.p();
					}
				// FORWARD PASS
				template< 	bool DROPOUT = false,
							typename DERIVED,
							typename DERIVED_2 >
				void forwardpass( const Eigen::MatrixBase< DERIVED >& weights, const Eigen::MatrixBase< DERIVED_2 >& input ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar >::value, "The vectors given as input has to have same scalar type!" );
					static_assert( std::is_same<typename DERIVED::Scalar,WeightType >::value, "The vectors' scalar type has to be consistent to the declared weight type!" );
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "The inputs have to be column vectors (or consistent structures)!");
					MLEARN_ASSERT( ( layers.head(layers.size()-1).dot(layers.tail(layers.size()-1)) + layers.tail(layers.size()-1).array().sum()) == weights.size(), "Number of weights is not consistent with number of units in the layers!" );
					MLEARN_ASSERT( input.size() == layers[0] , "Input should be of the declared size of the first layer!" );
					
					pre_activations.segment(0,layers[0]) = input;
					UnitProcessor<DROPOUT>::template process<ActivationType::LINEAR,WeightType,RNG_TYPE>(pre_activations.segment(0,layers[0]),activations.segment(0,layers[0]),dropped.segment(0,layers[0]),rng,b_dist,inv_p);
					
					// declare refs to derived: if DERIVED_* are expressions then they will be evaluated otherwise we can access its data without copies
					const Eigen::Ref<const Eigen::Matrix< typename DERIVED::Scalar, DERIVED::RowsAtCompileTime, DERIVED::ColsAtCompileTime > >& ref_weights(weights);
					
					decltype(activations.size()) offset = layers[0];
					decltype(weights.size()) offset_weights = 0;
					decltype(weights.size()) tmp;
					decltype(activations.size()) idx;
					decltype(idx) idx_m1;

					for ( idx = 1; idx < (layers.size() - 1); ++idx ){

						idx_m1 = idx - 1;
						tmp = layers[idx_m1]*layers[idx];
						pre_activations.segment(offset,layers[idx]) = Eigen::Map<const MLMatrix<WeightType>>( ref_weights.data() + offset_weights, layers[idx], layers[idx_m1] )*activations.segment(offset-layers[idx_m1],layers[idx_m1]);
						offset_weights += tmp;
						pre_activations.segment(offset,layers[idx]) += ref_weights.segment(offset_weights,layers[idx]);	
						UnitProcessor<DROPOUT>::template process<HiddenLayerActivation,WeightType,RNG_TYPE>(pre_activations.segment(offset,layers[idx]),activations.segment(offset,layers[idx]),dropped.segment(offset,layers[idx]),rng,b_dist,inv_p);
						offset_weights += layers[idx];
						offset += layers[idx];  
					}
					
					idx_m1 = idx - 1;
					tmp = layers[idx_m1]*layers[idx];
					pre_activations.segment(offset,layers[idx]) = Eigen::Map<const MLMatrix<WeightType>>( ref_weights.data() + offset_weights, layers[idx], layers[idx_m1] )*activations.segment(offset-layers[idx_m1],layers[idx_m1]);
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
							typename DERIVED_3 >
				void backpropagate( const Eigen::MatrixBase<DERIVED>& weights, const Eigen::MatrixBase<DERIVED_2>& gradient_output, Eigen::MatrixBase<DERIVED_3>& gradient_weights ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar >::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_3::Scalar >::value, "The vectors given as input has to have same scalar type!" );
					static_assert( std::is_same<typename DERIVED::Scalar,WeightType >::value, "The vectors' scalar type has to be consistent to the declared weight type!" );
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1) && (DERIVED_3::ColsAtCompileTime == 1), "The inputs have to be column vectors (or consistent structures)!");
					MLEARN_ASSERT( ( layers.head(layers.size()-1).dot(layers.tail(layers.size()-1)) + layers.tail(layers.size()-1).array().sum()) == weights.size(), "Number of weights is not consistent with number of units in the layers!" );
					MLEARN_ASSERT( gradient_output.size() == layers[layers.size()-1], "Output layer not consistent with the given gradient!" );
					MLEARN_ASSERT( gradient_weights.size() == weights.size(), "Weights size and respective gradient size not compatible!" );

					// declare refs to derived: if DERIVED_* are expressions then they will be evaluated otherwise we can access its data without copies
					const Eigen::Ref<const Eigen::Matrix< typename DERIVED::Scalar, DERIVED::RowsAtCompileTime, DERIVED::ColsAtCompileTime > >& ref_weights(weights);
					
					decltype(activations.size()) idx = layers.size()-1;
					decltype(activations.size()) idx_m1 = idx-1;
					decltype(activations.size()) offset = layers.head(idx).array().sum();
					decltype(ref_weights.size()) offset_weights = ref_weights.size() - layers[idx];
					decltype(ref_weights.size()) tmp = layers[idx_m1]*layers[idx];

					// compute delta output layer and assign it to the gradientof the last biases
					ActivationDerivativeWrapper<OutputLayerActivation>::derive( pre_activations.segment(offset,layers[idx]),activations.segment(offset,layers[idx]),derivatives );
					gradient_weights.segment(offset_weights,layers[idx]) = gradient_output.cwiseProduct(derivatives);
					// ... and get a view of the last delta
					Eigen::Map< const MLVector<WeightType> > delta_p1( gradient_weights.segment(offset_weights,layers[idx]).data(), layers[idx] );

					// update offsets
					offset_weights -= tmp;
					offset -= layers[idx_m1];
					
					// get appropriate view of the weight matrices
					Eigen::Map<MLMatrix<WeightType>> diff_weight_matrix(gradient_weights.segment(offset_weights,tmp).data(),layers[idx],layers[idx_m1]);
					Eigen::Map<const MLMatrix<WeightType>> weight_matrix(ref_weights.data()+offset_weights,layers[idx],layers[idx_m1]);
					
					// update gradient of last weight matrix
					diff_weight_matrix.noalias() = delta_p1*activations.segment(offset,layers[idx_m1]).transpose();
					
					// update offset weights 
					--idx;

					for ( ; idx > 0; --idx ){
						
						offset_weights -= layers[idx_m1];
						--idx_m1;

						tmp = layers[idx_m1]*layers[idx];

						ActivationDerivativeWrapper<HiddenLayerActivation>::derive( pre_activations.segment(offset,layers[idx]),activations.segment(offset,layers[idx]),derivatives );
						UnitProcessor<DROPOUT>::template process_derivatives<WeightType>(derivatives,dropped.segment(offset,layers[idx]));
						gradient_weights.segment(offset_weights,layers[idx]) = (weight_matrix.transpose()*delta_p1).cwiseProduct( derivatives );
						new (&delta_p1) Eigen::Map< const MLVector<WeightType> >( gradient_weights.segment(offset_weights,layers[idx]).data(), layers[idx] );
						
						offset_weights -= tmp;
						offset -= layers[idx_m1];

						new (&diff_weight_matrix) Eigen::Map<MLMatrix<WeightType>> (gradient_weights.segment(offset_weights,tmp).data(),layers[idx],layers[idx_m1]);
						new (&weight_matrix) Eigen::Map<const MLMatrix<WeightType>> (ref_weights.data()+offset_weights,layers[idx],layers[idx_m1]);

						diff_weight_matrix.noalias() = delta_p1*activations.segment(offset,layers[idx_m1]).transpose();
						
					}

				}
				// MODIFIERS
				void setDropoutProbability( WeightType prob ){
					MLEARN_WARNING( ( (prob >= 0) && (prob <= 1) ), "Probability value not valid!"  );
					b_dist.param( std::bernoulli_distribution::param_type(prob) );
					inv_p = WeightType(1)/prob;
				}
				// UPDATE INTERNAL MODEL ( external modification of the layers )
				void internal_resize(){
					MLEARN_ASSERT( layers.size() > 1, "Not a deep architecture: at least 2 layers are needed for the algorithm to work!" );
					MLEARN_WARNING( ( layers.array() > 0 ).all(), "Some layers have zero units!" );
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
				WeightType inv_p;
			};

		}

	}

}

#endif