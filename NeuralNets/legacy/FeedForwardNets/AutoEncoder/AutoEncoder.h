#ifndef MLEARN_AUTOENCODER_CLASS_
#define MLEARN_AUTOENCODER_CLASS_

// MLearn
#include <MLearn/Core>
#include "../Common/FeedForwardBase.h"
#include "../../ActivationFunction.h"

// STL
#include <type_traits>

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
			class AutoEncoder: public FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, AutoEncoder< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation > >{
			public:
				// Static asserts
				static_assert( std::is_floating_point<WeightType>::value, "The weights type has to be floating point!" );
				static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value , "The index type has to be unsigned integer!" );
				// CONSTRUCTOR
				AutoEncoder(): BaseClass(){}
				AutoEncoder( const MLVector< IndexType >& refLayers ): 
						BaseClass(refLayers) {
							MLEARN_ASSERT( refLayers[0] == refLayers[ refLayers.size() - 1 ], "An autoencoder has to have input and output layer of the same dimension!" );
						}
				AutoEncoder( const AutoEncoder< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation >& refEncoder ): 
						BaseClass(refEncoder) {}
				AutoEncoder( AutoEncoder< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation >&& refEncoder ): 
						BaseClass(std::move(refEncoder)) {}
				// TRAIN
				template < 	FFNetTrainingMode MODE,
							Regularizer REG,
							LossType LOSS,
							typename MINIMIZER,
							typename EIGEN_DERIVED >
				void train( const Eigen::MatrixBase< EIGEN_DERIVED >& input, MINIMIZER& minimizer, const RegularizerOptions< WeightType >& reg_options = RegularizerOptions< WeightType >() ){
					this->template train_base_implementation<MODE,REG,LOSS>(input,input,minimizer,reg_options);
				}
				template < typename EIGEN_DERIVED >
				MLMatrix< typename EIGEN_DERIVED::Scalar > autoencode( const Eigen::MatrixBase< EIGEN_DERIVED >& input ){
					return this->forwardpass_implementation(input);
				}
			protected:	
			public:	
				void setLayers_implementation(const MLVector< IndexType >& new_layers){
					MLEARN_ASSERT( new_layers[0] == new_layers[ new_layers.size() - 1 ], "An autoencoder has to have input and output layer of the same dimension!" );
					static_cast<BaseClass*>(this)->setLayers_implementation(new_layers);
				}
				void setLayersAndWeights_implementation(const MLVector< IndexType >& new_layers, const MLVector< WeightType >& new_weights){
					MLEARN_ASSERT( new_layers[0] == new_layers[ new_layers.size() - 1 ], "An autoencoder has to have input and output layer of the same dimension!" );
					static_cast<BaseClass*>(this)->setLayersAndWeights_implementation(new_layers,new_weights);
				}	
			private:
				typedef FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, AutoEncoder< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation > > BaseClass;
			};
		}
	}

}

#endif /* MLEARN_AUTOENCODER_CLASS_ */