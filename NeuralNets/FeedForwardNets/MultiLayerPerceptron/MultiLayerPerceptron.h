#ifndef MLEARN_MULTI_LAYER_PERCEPTRON_CLASS_
#define MLEARN_MULTI_LAYER_PERCEPTRON_CLASS_

// MLearn
#include <MLearn/Core>
#include "../Common/FeedForwardBase.h"
#include "../../ActivationFunction.h"

namespace MLearn{

	namespace NeuralNets{

		namespace FeedForwardNets{
				
			/*!
			*
			*	\brief		Autoencoder class.
			*	\details	Implementation of the autoencoder class.
			*	\author		phineasng
			*/
			template< 	typename WeightType, 
						typename IndexType,
						ActivationType HiddenLayerActivation = ActivationType::LOGISTIC,
						ActivationType OutputLayerActivation = ActivationType::LINEAR >
			class MultiLayerPerceptron: public FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, MultiLayerPerceptron< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation > >{
			public:
				// Static asserts
				static_assert( std::is_floating_point<WeightType>::value, "The weights type has to be floating point!" );
				static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value , "The index type has to be unsigned integer!" );
				// CONSTRUCTOR
				MultiLayerPerceptron(): BaseClass(){}
				MultiLayerPerceptron( const MLVector< IndexType >& refLayers ): 
						BaseClass(refLayers) {
						}
				MultiLayerPerceptron( const MultiLayerPerceptron< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation >& refEncoder ): 
						BaseClass(refEncoder) {}
				MultiLayerPerceptron( MultiLayerPerceptron< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation >&& refEncoder ): 
						BaseClass(std::move(refEncoder)) {}
				// TRAIN
				template < 	FFNetTrainingMode MODE,
							Regularizer REG,
							LossType LOSS,
							typename MINIMIZER,
							typename EIGEN_DERIVED >
				void train( const Eigen::MatrixBase< EIGEN_DERIVED >& input, const Eigen::MatrixBase< EIGEN_DERIVED >& output, MINIMIZER& minimizer, const RegularizerOptions< WeightType >& reg_options = RegularizerOptions< WeightType >() ){
					this->template train_base_implementation<MODE,REG,LOSS>(input,output,minimizer,reg_options);
				}
				template < typename EIGEN_DERIVED >
				MLMatrix< typename EIGEN_DERIVED::Scalar > forwardpass( const Eigen::MatrixBase< EIGEN_DERIVED >& input ){
					return this->forwardpass_implementation(input);
				}
				template < 	typename INDEX_RETURN_TYPE,
							typename EIGEN_DERIVED >
				MLVector< INDEX_RETURN_TYPE > classify( const Eigen::MatrixBase< EIGEN_DERIVED >& input ){
					static_assert( std::is_integral<INDEX_RETURN_TYPE>::value, "For classification, only integral types are supported as return type!" );
					MLVector< INDEX_RETURN_TYPE > classification(input.cols());

					IndexType output_dimension = this->layers[ this->layers.size() - 1 ];
					for ( decltype(input.cols()) idx = 0; idx < input.cols(); ++idx ){
						this->explorer.forwardpass(this->weights,input.col(idx));
						this->explorer.getActivations().tail(output_dimension).maxCoeff(&classification[idx]);
					}

					return classification;
				}
			protected:	
			public:	
				void setLayers_implementation(const MLVector< IndexType >& new_layers){
					static_cast<BaseClass*>(this)->setLayers_implementation(new_layers);
				}
				void setLayersAndWeights_implementation(const MLVector< IndexType >& new_layers, const MLVector< WeightType >& new_weights){
					static_cast<BaseClass*>(this)->setLayersAndWeights_implementation(new_layers,new_weights);
				}	
			private:
				typedef FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, MultiLayerPerceptron< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation > > BaseClass;
			};
		}
	}

}


#endif