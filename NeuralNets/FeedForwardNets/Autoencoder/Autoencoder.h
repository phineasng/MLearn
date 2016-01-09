#ifndef MLEARN_AUTOENCODER_CLASS_
#define MLEARN_AUTOENCODER_CLASS_

// Core functionalities and definitions
#include <MLearn/Core>

namespace MLearn{

	namespace NeuralNets{

		/*!
		*
		*	\brief		Autoencoder class.
		*	\details	Implementation of the autoencoder class as a feedforward fully connected neural net.
		*	\author		phineasng
		*/
		template< 	typename WeightType, 
					typename IndexType,
					typename = typename std::enable_if< std::is_floating_point<WeightType>::value , WeightType >::type,
					typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value , IndexType >::type >
		class Autoencoder{
		public:
			// CONSTRUCTORS
			Autoencoder() = default;
			// -- const reference layers vector 
			explicit Autoencoder( const MLVector<IndexType>& refLayers ): 
				_layers(refLayers),
				_weights(refLayers.head(refLayers.size()-1).dot(refLayers.tail(refLayers.size()-1)) + refLayers.array().sum())
			{
				MLEARN_WARNING( refLayers.size() > 1 , "Not a deep architecture: implemented training algorithm may throw an error!");
				MLEARN_ASSERT( refLayers[0] == refLayers[refLayers.size() - 1], "Autoencoders have to have the same number of units in the input and the output layers!" );
			}
			// -- copy constructor
			Autoencoder( const Autoencoder<WeightType,IndexType>& refAutoencoder ): 
				_layers(refAutoencoder._layers),
				_weights(refAutoencoder._weights){}
			// -- move constructor
			Autoencoder( Autoencoder<WeightType,IndexType>&& refAutoencoder ): 
				_layers(std::move(refAutoencoder._layers)),
				_weights(std::move(refAutoencoder._weights)){}
			// ASSIGNMENT OPERATOR
			// -- copy assignment
			Autoencoder& operator=( const Autoencoder<WeightType,IndexType>& refAutoencoder ){
				_layers = refAutoencoder._layers;
				_weights = refAutoencoder._weights;
				_activations = refAutoencoder._activations;
				_pre_activations = refAutoencoder._pre_activations;
				return *this;
			}
			// -- move assignment
			Autoencoder& operator=( Autoencoder<WeightType,IndexType>&& refAutoencoder ){
				_layers = std::move(refAutoencoder._layers);
				_weights = std::move(refAutoencoder._weights);
				_activations = std::move(refAutoencoder._activations);
				_pre_activations = std::move(refAutoencoder._pre_activations);
				return *this;
			}
			// OBSERVERS
			const MLVector<IndexType>& getLayers() const { return _layers; }
			const MLVector<WeightType>& getWeights() const { return _weights; }
			// MODIFIERS
			void setLayers( const MLVector<IndexType>& refLayers ){
				MLEARN_WARNING( refLayers.size() > 1 , "Not a deep architecture: implemented training algorithm may throw an error!");
				MLEARN_ASSERT( refLayers[0] == refLayers[refLayers.size() - 1], "Autoencoders have to have the same number of units in the input and the output layers!" );
				_layers = refLayers;
				_weights.conservativeResize(refLayers.head(refLayers.size()-1).dot(refLayers.tail(refLayers.size()-1)) + refLayers.array().sum());
				_activations.conservativeResize(refLayers.array().sum()-refLayers[0]);
				_pre_activations.conservativeResize(refLayers.array().sum());
			}
			void setWeights( const MLVector<WeightType>& refWeights ){
				MLEARN_ASSERT( refWeights.size() ==  _layers.head(_layers.size()-1).dot(_layers.tail(_layers.size()-1)) + _layers.array().sum(),"Weights vector size not compatible with the declared layers. Please use the function setLayersAndWeights() to set consistent vectors at the same time!");
				_weights = refWeights;
			}
			void setLayersAndWeights( const MLVector<IndexType>& refLayers, const MLVector<WeightType>& refWeights ){
				MLEARN_WARNING( refLayers.size() > 1 , "Not a deep architecture: implemented training algorithm may throw an error!");
				MLEARN_ASSERT( refLayers[0] == refLayers[refLayers.size() - 1], "Autoencoders have to have the same number of units in the input and the output layers!" );
				MLEARN_ASSERT( refWeights.size() ==  refLayers.head(refLayers.size()-1).dot(refLayers.tail(refLayers.size()-1)) + refLayers.array().sum(),"Weights and layers vectors size not compatible!");
				_layers = refLayers;
				_weights = refWeights;
				_activations.conservativeResize(refLayers.array().sum()-refLayers[0]);
				_pre_activations.conservativeResize(refLayers.array().sum());
			}
			// TRAINING ALGORITHM
			void initialize(){
				_weights = std::move(MLVector<WeightType>::Random(_weights.size()));
			}
			template < typename  >
			void train(  ){

			}
			// ENCODING/DECODING
			MLVector<WeightType> autoencode( const MLVector<WeightType>& input ){
				return this->encode(input,_layers[_layers.size()-1]);
			}
			// --- give the encoded vector at layer i
			MLVector<WeightType> encode( const MLVector<WeightType>& input, IndexType i );
			// --- decode vector from layer i
			MLVector<WeightType> decode( const MLVector<WeightType>& encodedInput, IndexType i );
		protected:
			MLVector< IndexType > _layers;				 
			MLVector< WeightType > _weights;
		private:
		};

	}

}

#endif /* MLEARN_AUTOENCODER_CLASS_ */