#ifndef MLEARN_RECURRENT_LAYER_NET_EXPLORER_INCLUDE
#define MLEARN_RECURRENT_LAYER_NET_EXPLORER_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include "../../ActivationFunction.h"
#include "RecurrentCellType.h"

// STL includes 
#include <type_traits>

namespace MLearn{

	namespace NeuralNets{

		namespace RecurrentNets{

			template < 	typename WeightType,
						typename IndexType, 
						RNNType CELLTYPE = RnnType::LSTM,
						ActivationType ACTIVATION = ActivationType::LOGISTIC >
			class RecurrentLayer{
				// static asserts
				static_assert( std::is_floating_point< WeightType >::value, "The weights type has to be floating point!" );
				static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
			public:
				// typedefs
				typedef WeightType 				scalar_t;
				typedef WeightType 				index_t;
				static const ActivationType 	Activation = ACTIVATION;
				static const RnnType 			CellType = CELLTYPE;
				// Constructors
				template < typename UINT >
				RecurrentLayer( UINT input_sz, UINT hidden_sz, UINT output_sz ):
					input_size(input_sz),
					output_size(output_sz),
					hidden_size(hidden_sz),
					inputs(input_size,100),
					internal_state( hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states*100 ),
					internal_state_pre_activation( hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states*100 ),
					outputs(output_size,100),
					outputs_pre_activation(output_size,100)
				{
					static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
				}
				// Modifiers
				void resetHiddenState() { internal_state = MLMatrix< WeightType >::Zero( internal_state.rows(),internal_state.cols() ); }
				void attachWeightsToCell( Eigen::Ref< const MLVector< WeightType > > weights ) { cell.attachWeights(weights, input_size,hidden_size,output_size); }
				// Observers
				IndexType getInputSize() const { return input_size; }
				IndexType getOutputSize() const { return output_size; }
				IndexType getHiddenSize() const { return hidden_size; }
				// Forward passing routines
				void forwardpass_step_input_output( Eigen::Ref< const MLVector< WeightType > > weights, Eigen::Ref< const MLVector< WeightType > > input, Eigen::Ref< MLVector<WeightType> > output ){
					
					
					
				}
			private:
				// sizes
				const IndexType input_size, output_size, hidden_size;
				// internal state storage
				MLMatrix< WeightType > internal_state;
				MLMatrix< WeightType > internal_state_pre_activation;
				// input and output state storage (for training phase)
				MLMatrix< WeightType > outputs;
				MLMatrix< WeightType > outputs_pre_activation;
				MLMatrix< WeightType > inputs;
				// Temporaries
				MLVector< WeightType > grad_temp;
				// Cell - for specialized routines
				RNNCell< WeightType,CellType,Activation > cell;

			};

		}

	}

}

#endif