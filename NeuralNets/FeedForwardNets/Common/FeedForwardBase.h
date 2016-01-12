#ifndef MLEARN_FEED_FORWARD_BASE_CLASS_
#define MLEARN_FEED_FORWARD_BASE_CLASS_

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

			enum class FFNetTrainingMode{
				ONLINE,
				BATCH
			};

			/*!
			*	\brief		Feedforward base class
			*	\details	Class for generic feedforward nets functionalities.
			*	\author		phineasng
			*/
			template <	typename WeightType,
						typename IndexType,
						ActivationType HiddenLayerActivation,
						ActivationType OutputLayerActivation,
						typename DERIVED >
			class FeedForwardBase{
			public:
				// Static asserts
				static_assert( std::is_floating_point<WeightType>::value, "The weights type has to be floating point!" );
				static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value , "The index type has to be unsigned integer!" );
				// Constructors
				FeedForwardBase(): explorer(layers){}
				FeedForwardBase( const MLVector< IndexType >& refLayers ): 
						layers(refLayers), 
						weights( ( refLayers.head(refLayers.size()-1).dot(refLayers.tail(refLayers.size()-1)) + refLayers.tail(refLayers.size()-1).array().sum())),
						explorer(layers) {}
				FeedForwardBase( const FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& refBase ): 
						layers(refBase.layers), 
						weights(refBase.weights),
						explorer(layers),
						shared_weights(refBase.shared_weights),
						tr_shared_weights(refBase.tr_shared_weights) {}
				FeedForwardBase( FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >&& refBase ): 
						layers(std::move(refBase.layers)), 
						weights(std::move(refBase.weights)),
						explorer(layers),
						shared_weights(std::move(refBase.shared_weights)),
						tr_shared_weights(std::move(refBase.tr_shared_weights)) {}
				// Assignment operator
				FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& operator=( const FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& refBase ){
					layers = refBase.layers;
					weights = refBase.weights;
					explorer.internal_resize();
					shared_weights = refBase.shared_weights;
					tr_shared_weights = refBase.tr_shared_weights;
					return *this;
				}
				FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >& operator=( const FeedForwardBase< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation, DERIVED >&& refBase ){
					layers = std::move(refBase.layers);
					weights = std::move(refBase.weights);
					explorer.internal_resize();
					shared_weights = std::move(refBase.shared_weights);
					tr_shared_weights = std::move(refBase.tr_shared_weights);
					return *this;
				}
				// MODIFIERS
				// -- Set Layers
				void setLayers(const MLVector< IndexType >& new_layers){
					static_cast<DERIVED*>(this)->setLayers_implementation(new_layers);
				}
				// -- Set Weights
				void setWeights(const MLVector< WeightType >& new_layers){
					static_cast<DERIVED*>(this)->setWeights_implementation(new_layers);
				}
				// -- Set Layers and Weights
				void setLayersAndWeights(const MLVector< IndexType >& new_layers,const MLVector< WeightType >& new_weights){
					static_cast<DERIVED*>(this)->setLayersAndWeights_implementation(new_layers,new_weights);
				}
				// -- Set Shared Weights
				void setSharedWeights(const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& new_sh_weights){
					static_cast<DERIVED*>(this)->setSharedWeights_implementation(new_sh_weights);
				}
				// -- Set Transposed Shared Weights
				void setTransposedSharedWeights(const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& new_tr_sh_weights){
					static_cast<DERIVED*>(this)->setTransposedSharedWeights_implementation(new_tr_sh_weights);
				}
				// OBSERVERS
				const MLVector< IndexType >& getLayers() const{ return layers; }
				const MLVector< WeightType >& getWeights() const{ return weights; }
				const MLMatrix<WeightType>& getSharedWeights() const{ return shared_weights; }
				const MLMatrix<WeightType>& getTransposedSharedWeights() const{ return tr_shared_weights; }
			protected:
				MLVector< IndexType > layers;
				MLVector< WeightType > weights;
				Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign > shared_weights;
				Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign > tr_shared_weights;
				FCNetsExplorer< WeightType, IndexType, HiddenLayerActivation, OutputLayerActivation > explorer;
				// pre-allocation 
				MLVector< WeightType > gradient_tmp;
				MLVector< WeightType > gradient_tmp_output;
			public:
				// base implementations
				void setLayers_implementation(const MLVector< IndexType >& new_layers){
					MLEARN_ASSERT( layers.size() > 1, "Not a deep architecture: at least 2 layers are needed for the algorithm to work!" );
					MLEARN_WARNING( ( new_layers.array() > 0 ).all(), "Some layers have zero units!" );
					layers = new_layers;
					weights.resize( layers.head(layers.size()-1).dot(layers.tail(layers.size()-1)) + layers.tail(layers.size()-1).array().sum() );
					explorer.internal_resize();
					shared_weights.resize(2,0);
					tr_shared_weights.resize(2,0);
				}
				void setWeights_implementation(const MLVector< WeightType >& new_weights){
					MLEARN_ASSERT( ( layers.head(layers.size()-1).dot(layers.tail(layers.size()-1)) + layers.tail(layers.size()-1).array().sum()) == new_weights.size(), "Number of weights is not consistent with number of units in the layers!" );
					weights = new_weights;
				}
				void setLayersAndWeights_implementation(const MLVector< IndexType >& new_layers, const MLVector< WeightType >& new_weights){
					MLEARN_ASSERT( layers.size() > 1, "Not a deep architecture: at least 2 layers are needed for the algorithm to work!" );
					MLEARN_WARNING( ( new_layers.array() > 0 ).all(), "Some layers have zero units!" );
					MLEARN_ASSERT( ( new_layers.head(new_layers.size()-1).dot(new_layers.tail(new_layers.size()-1)) + new_layers.tail(new_layers.size()-1).array().sum()) == new_weights.size(), "Number of weights is not consistent with number of units in the layers!" );
					layers = new_layers;
					weights = new_weights;
					explorer.internal_resize();
					shared_weights.resize(2,0);
					tr_shared_weights.resize(2,0);
				}
				void setSharedWeights_implementation(const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& new_shared_weights){
					if (new_shared_weights.size() > 0){
						MLEARN_ASSERT( new_shared_weights.maxCoeff() < (layers.size() - 1), "Input shared weights matrix contain a non acceptable layer index!" );
						MLEARN_ASSERT( new_shared_weights.rows() == 2, "Invalid shared weights matrix format!" );
					}
					shared_weights = new_shared_weights;
				}
				void setTransposedSharedWeights_implementation(const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& new_tr_shared_weights){
					if (new_tr_shared_weights.size() > 0){
						MLEARN_ASSERT( new_tr_shared_weights.maxCoeff() < (layers.size() - 1), "Input shared weights matrix contain a non acceptable layer index!" );
						MLEARN_ASSERT( new_tr_shared_weights.rows() == 2, "Invalid shared weights matrix format!" );
					}
					tr_shared_weights = new_tr_shared_weights;
				}
				template < 	FFNetTrainingMode MODE,
							Regularizer REG,
							LossType LOSS,
							typename MINIMIZER,
							typename EIGEN_DERIVED,
							typename EIGEN_DERIVED_2 >
				typename std::enable_if< MODE == FFNetTrainingMode::ONLINE >::type train_base_implementation( const Eigen::MatrixBase< EIGEN_DERIVED >& input, const Eigen::MatrixBase< EIGEN_DERIVED_2 >& output, MINIMIZER& minimizer, const RegularizerOptions< WeightType >& reg_options = RegularizerOptions< WeightType >() ){
					static_assert( std::is_same<typename EIGEN_DERIVED::Scalar,typename EIGEN_DERIVED_2::Scalar>::value, "Not compatible input and output scalar type!");
					static_assert( std::is_same<typename EIGEN_DERIVED::Scalar,WeightType>::value, "Input not compatible with the weights type!");
					MLEARN_ASSERT( input.cols() == output.cols(), "Not compatible input and output dimensions!");
					
					if ( input.cols() == 0 ){
						return;
					}

					gradient_tmp.resize( weights.size() );
					gradient_tmp_output.resize( output.rows() );

					Eigen::Map< const MLMatrix< WeightType > > single_input( input.col(0).data(), input.rows(), 1 );
					Eigen::Map< const MLMatrix< WeightType > > single_output( output.col(0).data(), output.rows(), 1 );

					for ( decltype( input.cols() ) i = 0; i < input.cols(); ++i ){

						new (&single_input) Eigen::Map< const MLVector< WeightType > >( input.col(i).data(), input.rows() );
						new (&single_output) Eigen::Map< const MLVector< WeightType > >( output.col(i).data(), output.rows() );

						TEMPLATED_FC_NEURAL_NET_COST_CONSTRUCTION_WITH_SHARED_WEIGHTS( LOSS,REG,layers,single_input,single_output,explorer,reg_options,gradient_tmp_output,gradient_tmp,shared_weights,tr_shared_weights,cost_function );
						minimizer.minimize(cost_function,weights);

					}

				}
				template < 	FFNetTrainingMode MODE,
							Regularizer REG,
							LossType LOSS,
							typename MINIMIZER,
							typename EIGEN_DERIVED,
							typename EIGEN_DERIVED_2 >
				typename std::enable_if< MODE == FFNetTrainingMode::BATCH >::type train_base_implementation( const Eigen::MatrixBase< EIGEN_DERIVED >& input, const Eigen::MatrixBase< EIGEN_DERIVED_2 >& output, MINIMIZER& minimizer, const RegularizerOptions< WeightType >& reg_options = RegularizerOptions< WeightType >() ){
					static_assert( std::is_same<typename EIGEN_DERIVED::Scalar,typename EIGEN_DERIVED_2::Scalar>::value, "Not compatible input and output scalar type!");
					static_assert( std::is_same<typename EIGEN_DERIVED::Scalar,WeightType>::value, "Input not compatible with the weights type!");
					MLEARN_ASSERT( input.cols() == output.cols(), "Not compatible input and output dimensions!");

					gradient_tmp.resize( weights.size() );
					gradient_tmp_output.resize( output.rows() );

					FCCostFunction< LOSS,
									REG,
									HiddenLayerActivation,
									OutputLayerActivation,
									IndexType,
									EIGEN_DERIVED,
									EIGEN_DERIVED_2 > cost_function(layers,input,output,explorer,reg_options,gradient_tmp_output,gradient_tmp,shared_weights,tr_shared_weights);
					minimizer.minimize(cost_function,weights);
				}
				template < typename EIGEN_DERIVED >
				MLMatrix< typename EIGEN_DERIVED::Scalar > forwardpass_implementation( const Eigen::MatrixBase< EIGEN_DERIVED >& input ){
					IndexType output_dimension = layers[ layers.size() - 1 ];
					MLMatrix< typename EIGEN_DERIVED::Scalar > output( output_dimension,input.cols());

					for ( decltype(input.cols()) idx = 0; idx < input.cols(); ++idx ){
						explorer.forwardpass(weights,input.col(idx));
						output.col(idx) = explorer.getActivations().tail( output_dimension );
					}

					return output;

				}
			};

		}
	}


}


#endif