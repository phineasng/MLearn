#ifndef MLEARN_MULTI_LAYER_PERCEPTRON_CLASS_
#define MLEARN_MULTI_LAYER_PERCEPTRON_CLASS_

// MLearn
#include <MLearn/Core>
#include "FCNetsExplorer.h"
#include "FCCostFunction.h"
#include "../../ActivationFunction.h"

// STL
#include <type_traits>

namespace MLearn{

	namespace NeuralNets{

		namespace FeedForwardNets{
	
			/*!
			*	\brief		MultiLayer Perceptron class
			*	\author		phineasng
			*/
			template <	typename WeightType,
						typename IndexType,
						ActivationType HiddenLayerActivation = ActivationType::LOGISTIC,
						ActivationType OutputLayerActivation = ActivationType::LINEAR,
						typename DERIVED = void >
			class FeedForwardBase{
			public:
				// Static asserts
				static_assert( std::is_floating_point<WeightType>::value, "The weights type has to be floating point!" );
				static_assert( std::is_floating_point<WeightType>::value, "The index type has to be unsigned integer!" );
				// Constructors
				FeedForwardBase() = delete;
				FeedForwardBase( const MLVector< IndexType >& refLayers ): 
						layers(refLayers), 
						weights( ( refLayers.head(refLayers.size()-1).dot(refLayers.tail(refLayers.size()-1)) + refLayers.tail(refLayers.size()-1).array().sum())),
						explorer(layers) {}
				FeedForwardBase( const FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& refBase ): 
						layers(refBase.layers), 
						weights(refBase.weights),
						explorer(layers) {}
				FeedForwardBase( FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >&& refBase ): 
						layers(std::move(refBase.layers)), 
						weights(std::move(refBase.weights)),
						explorer(layers) {}
				// Assignment operator
				FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& operator=( const FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& refBase ){
					layers = refBase.layers;
					weights = refBase.weights;
					explorer.internal_resize();
					return *this;
				}
				FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& operator=( const FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >&& refBase ){
					layers = std::move(refBase.layers);
					weights = std::move(refBase.weights);
					explorer.internal_resize();
					return *this;
				}
				// Modifiers
				// -- Set Layers
				template < typename Enable = void >
				void setLayers(const MLVector< IndexType >& new_layers){
					static_cast<DERIVED*>(this)->setLayers(new_layers);
				}
				template < typename = typename std::enable_if< !std::is_same< DERIVED, void >::value >::type >
				void setLayers(const MLVector< IndexType >& new_layers){
					setLayers_base_implementation(new_layers);
				}
			protected:
				MLVector< IndexType > layers;
				MLVector< WeightType > weights;
				FCNetsExplorer< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation > explorer;
				// base implementations
				void setLayers_base_implementation(){
	
				}
			};
		}
	}

}


#endif