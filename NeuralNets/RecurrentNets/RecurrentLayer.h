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
						ActivationType HIDDEN_ACTIVATION = ActivationType::LOGISTIC,
						ActivationType OUTPUT_ACTIVATION = ActivationType::LOGISTIC >
			class RecurrentLayer{
				// static asserts
				static_assert( std::is_floating_point< WeightType >::value, "The weights type has to be floating point!" );
				static_assert( std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, "Size values must be unsigned integers!" );
			public:
				// typedefs
				typedef WeightType 				scalar_t;
				typedef WeightType 				index_t;
				static const ActivationType 	HiddenActivation = HIDDEN_ACTIVATION;
				static const ActivationType 	OutputActivation = OUTPUT_ACTIVATION;
				static const RnnType 			CellType = CELLTYPE;
				// Constructors
				template < typename U_INT >
				RecurrentLayer( U_INT input_sz, U_INT hidden_sz, U_INT output_sz, U_INT n_states_alloc = 100u ):
					input_size(input_sz),
					output_size(output_sz),
					hidden_size(hidden_sz),
					inputs(input_size,n_states_alloc),
					internal_state( hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states*n_states_alloc ),
					internal_state_pre_activation( hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states*n_states_alloc ),
					outputs(output_size,n_states_alloc),
					outputs_pre_activation(output_size,n_states_alloc),
					grad_temp(cell.computeNWeights(input_sz,hidden_sz,output_sz)),
					grad_hidden(hidden_sz)
				{
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Size values must be unsigned integers!" );
				}
				// Modifiers
				void resetHiddenState() { internal_state = MLMatrix< WeightType >::Zero( internal_state.rows(),internal_state.cols() ); }
				void attachWeightsToCell( Eigen::Ref< const MLVector< WeightType > > weights ) { cell.attachWeights(weights, input_size,hidden_size,output_size); }
				void setHiddenState( const Eigen::Ref< MLMatrix<WeightType> >& ref_hidden_state ) {
					MLEARN_ASSERT( (ref_hidden_state.cols() ==  RNNCellTraits<CELLTYPE>::N_internal_states) &&  (ref_hidden_state.rows() == hidden_size), "Wrong size for the hidden state!" );
					internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states) = ref_hidden_state;
				}
				template < typename U_INT >
				void resizeInputsAllocation(U_INT size_alloc){
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Size value must be unsigned integers!" );
					inputs.conservativeResize(input_size,size_alloc);
				}
				template < typename U_INT >
				void resizeOutputsAllocation(U_INT size_alloc){
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Size value must be unsigned integers!" );
					outputs.conservativeResize(output_size,size_alloc);
					outputs_pre_activation.conservativeResize(output_size,size_alloc);
				}
				template < typename U_INT >
				void resizeHiddenAllocation(U_INT size_alloc){
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Size value must be unsigned integers!" );
					internal_state.conservativeResize(hidden_size,size_alloc*RNNCellTraits<CELLTYPE>::N_internal_states);
					internal_state_pre_activation.conservativeResize(hidden_size,size_alloc*RNNCellTraits<CELLTYPE>::N_internal_states);
				}
				// Observers
				IndexType getInputSize() const { return input_size; }
				IndexType getOutputSize() const { return output_size; }
				IndexType getHiddenSize() const { return hidden_size; }
				// Forward passing routines
				void forwardpass_step( const Eigen::Ref< const MLVector< WeightType > > input ){
					MLEARN_ASSERT( input.size() == input_size, "Input have the wrong size!" );
					inputs.col(0) = input;
					cell.step_input( input, internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states));
				}
				void forwardpass_step(){
					cell.step_hidden( internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states));
				}
				Eigen::Ref<MLVector<WeightType>> forwardpass_step_output( const Eigen::Ref< const MLVector< WeightType > > input ){
					MLEARN_ASSERT( input.size() == input_size, "Input have the wrong size!" );
					inputs.col(0) = input;
					cell.step_input_output( input, outputs.col(0), internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states));
					return outputs.col(0);
				}
				Eigen::Ref<MLVector<WeightType>> forwardpass_step_output(){
					cell.step_output( outputs.col(0), internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states));
					return outputs.col(0);
				}
				void forwardpass_step_unroll( const Eigen::Ref< const MLMatrix< WeightType > > input, size_t delay, size_t output_steps, bool reset = false ){
					MLEARN_ASSERT( input.rows() == input_size, "Input have the wrong size!" );
					
					if (input.cols() > inputs.cols()){
						resizeInputsAllocation(input.cols());
					}
					inputs.leftCols(input.cols()) = input;

					if (output_steps > outputs.cols()){
						resizeOutputsAllocation(output_steps);
					}
					unrolled_output_steps = output_steps;

					IndexType curr_step = 0;
					IndexType N_steps_only_input ;
					N_steps_only_input = std::min(input.cols(),delay);
					unrolled_steps = delay + output_steps;
					if ( (unrolled_steps+1)*RNNCellTraits<CELLTYPE>::N_internal_states > internal_state.cols() ){
						resizeHiddenAllocation((unrolled_steps+1)*RNNCellTraits<CELLTYPE>::N_internal_states);
					}
					if (reset){
						resetHiddenState();
					}
					// unrolling steps with only inputs (and hidden states)
					for (; curr_step < N_steps_only_input; ++curr_step){
						IndexType start_col_old = curr_step*RNNCellTraits<CELLTYPE>::N_internal_states;
						IndexType start_col = start_col_old + RNNCellTraits<CELLTYPE>::N_internal_states;
						cell.step_input_to_hidden_unroll( 	input.col(curr_step), 
															internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
					}
					// unrolling steps with only hidden states
					for (; curr_step < delay; ++curr_step){
						IndexType start_col = (curr_step+1)*RNNCellTraits<CELLTYPE>::N_internal_states;
						cell.step_hidden_to_hidden_unroll( 	internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
					}
					// unrolling hidden states and outputs
					IndexType offset = delay+1;
					for (IndexType curr_output_step = 0; curr_output_step < output_steps; ++curr_output_step){
						IndexType start_col = (curr_output_step+offset)*RNNCellTraits<CELLTYPE>::N_internal_states;
						cell.step_hidden_to_output_unroll( 	internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															outputs_pre_activation.col(curr_output_step),
															outputs.col(curr_output_step));
					}
				}
				// NOTE: valid use only after unrolling
				const Eigen::Ref< const MLVector<WeightType>> getLastOutput() const{
					MLEARN_ASSERT( unrolled_output_steps > 0, "RNN was not unrolled enough before this query!" );
					MLEARN_WARNING( "This function returns a valid result only if unrolling has been performed first!" );
					return outputs.col(unrolled_output_steps-1);
				}
				// NOTE: valid use only after unrolling
				const Eigen::Ref< const MLMatrix<WeightType>> getAllOutputs() const{
					MLEARN_ASSERT( unrolled_output_steps > 0, "RNN was not unrolled enough before this query!" );
					MLEARN_WARNING( "This function returns a valid result only if unrolling has been performed first!" );
					return outputs.leftCols(unrolled_output_steps-1);
				}
				// NOTE: valid use only after unrolling
				template < typename U_INT >
				const Eigen::Ref< const MLMatrix<WeightType>> getOutput(U_INT step_idx) const{
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Step index must be unsigned integer!" );
					MLEARN_ASSERT( unrolled_output_steps > 0, "RNN was not unrolled enough before this query!" );
					MLEARN_ASSERT( unrolled_output_steps > step_idx, "RNN was not unrolled enough before this query!" );
					MLEARN_WARNING( "This function returns a valid result only if unrolling has been performed first!" );
					return outputs.col(step_idx);
				}
				// NOTE: valid use only after unrolling
				const Eigen::Ref< const MLVector<WeightType>> getLastHiddenState() const{
					MLEARN_ASSERT( unrolled_steps > 0, "RNN was not unrolled before this query!" );
					MLEARN_WARNING( "This function returns a valid result only if unrolling has been performed first!" );
					IndexType start_col = unrolled_steps*RNNCellTraits<CELLTYPE>::N_internal_states;
					return cell.block( 0,start_col, hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states );
				}
				// NOTE: valid use only after unrolling
				template < typename U_INT >
				const Eigen::Ref< const MLVector<WeightType>> getHiddenState(U_INT state_idx) const{
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Step index must be unsigned integer!" );
					MLEARN_ASSERT( unrolled_steps > 0, "RNN was not unrolled before this query!" );
					MLEARN_ASSERT( unrolled_steps > state_idx, "RNN was not unrolled enough before this query!" );
					MLEARN_WARNING( "This function returns a valid result only if unrolling has been performed first!" );
					IndexType start_col = (state_idx+1)*RNNCellTraits<CELLTYPE>::N_internal_states;
					return cell.block( 0,start_col, hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states );
				}
			private:
				// sizes
				const IndexType input_size, output_size, hidden_size;
				IndexType unrolled_steps=0;
				IndexType unrolled_output_steps=0;
				// internal state storage
				MLMatrix< WeightType > internal_state;
				MLMatrix< WeightType > internal_state_pre_activation;
				// input and output state storage (for training phase)
				MLMatrix< WeightType > outputs;
				MLMatrix< WeightType > outputs_pre_activation;
				MLMatrix< WeightType > inputs;
				// Temporaries
				MLVector< WeightType > grad_temp;
				MLVector< WeightType > grad_hidden;
				// Cell - for specialized routines
				RNNCell< WeightType,CellType,HiddenActivation,OutputActivation > cell;

			};

		}

	}

}

#endif