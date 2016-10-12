#ifndef MLEARN_RECURRENT_LAYER_NET_EXPLORER_INCLUDE
#define MLEARN_RECURRENT_LAYER_NET_EXPLORER_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include "../ActivationFunction.h"
#include "RecurrentCellType.h"

// STL includes 
#include <type_traits>

namespace MLearn{

	namespace NeuralNets{

		namespace RecurrentNets{

			template < 	typename WeightType,
						typename IndexType, 
						RNNType CELLTYPE = RNNType::LSTM,
						ActivationType HIDDEN_ACTIVATION = ActivationType::LOGISTIC,
						ActivationType OUTPUT_ACTIVATION = ActivationType::LOGISTIC >
			class RecurrentLayer{
				// static asserts
				static_assert( std::is_floating_point< WeightType >::value, "The weights type has to be floating point!" );
				static_assert( std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, "Size values must be unsigned integers!" );
			public:
				// typedefs
				typedef WeightType 				scalar_t;
				typedef IndexType 				index_t;
				static const ActivationType 	HiddenActivation = HIDDEN_ACTIVATION;
				static const ActivationType 	OutputActivation = OUTPUT_ACTIVATION;
				static const RNNType 			CellType = CELLTYPE;
				// Constructors
				template < typename U_INT >
				RecurrentLayer( U_INT input_sz, U_INT hidden_sz, U_INT output_sz, U_INT n_states_alloc = 100u ):
					input_size(input_sz),
					output_size(output_sz),
					hidden_size(hidden_sz),
					internal_state( hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states*n_states_alloc ),
					internal_state_pre_activation( hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states*n_states_alloc ),
					outputs(output_size,n_states_alloc),
					outputs_pre_activation(output_size,n_states_alloc),
					inputs(input_size,n_states_alloc),
					grad_temp(cell.computeNWeights(input_sz,hidden_sz,output_sz)),
					grad_hidden(hidden_sz,size_t(RNNCellTraits<CELLTYPE>::N_cell_gradients))
				{
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Size values must be unsigned integers!" );
				}
				// Modifiers
				void resetHiddenState() { 
					internal_state = MLMatrix< WeightType >::Zero( internal_state.rows(),internal_state.cols() ); 
					internal_state_pre_activation = MLMatrix< WeightType >::Zero( internal_state.rows(),internal_state.cols() ); 
					unrolled_steps = 0;
				}
				void attachWeightsToCell( Eigen::Ref< const MLVector< WeightType > > weights ) { cell.attachWeights(weights, input_size,hidden_size,output_size); }
				void setHiddenState( const Eigen::Ref< MLMatrix<WeightType> >& ref_hidden_state ) {
					MLEARN_ASSERT( (ref_hidden_state.cols() ==  RNNCellTraits<CELLTYPE>::N_internal_states) &&  (ref_hidden_state.rows() == hidden_size), "Wrong size for the hidden state!" );
					resetHiddenState();
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
					unrolled_steps = 0;
				}
				void forwardpass_step(){
					cell.step_hidden( internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states));
					unrolled_steps = 0;
				}
				Eigen::Ref<MLVector<WeightType>> forwardpass_step_output( const Eigen::Ref< const MLVector< WeightType > > input ){
					MLEARN_ASSERT( input.size() == input_size, "Input have the wrong size!" );
					inputs.col(0) = input;
					cell.step_input_output( input, outputs.col(0), internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states));
					unrolled_steps = 0;
					return outputs.col(0);
				}
				Eigen::Ref<MLVector<WeightType>> forwardpass_step_output(){
					cell.step_output( outputs.col(0), internal_state.leftCols(RNNCellTraits<CELLTYPE>::N_internal_states));
					unrolled_steps = 0;
					return outputs.col(0);
				}
				void forwardpass_unroll( const Eigen::Ref< const MLMatrix< WeightType > > input, size_t delay, size_t output_steps, bool reset = true ){
					MLEARN_ASSERT( input.rows() == input_size, "Input have the wrong size!" );
					MLEARN_WARNING( delay + output_steps >= IndexType(input.cols()) , "Not all the inputs will be used!" );
					
					if (input.cols() > inputs.cols()){
						resizeInputsAllocation(size_t(input.cols()));
					}
					inputs.leftCols(input.cols()) = input;

					if (output_steps > size_t(outputs.cols())){
						resizeOutputsAllocation(output_steps);
					}

					unrolled_output_steps = output_steps;
					unrolled_steps = delay + output_steps;
					unrolled_input_steps = input.cols();

					IndexType N_steps_only_input ;
					N_steps_only_input = std::min(size_t(input.cols()),delay);
					if ( (unrolled_steps+1)*RNNCellTraits<CELLTYPE>::N_internal_states > internal_state.cols() ){
						resizeHiddenAllocation((unrolled_steps+1)*RNNCellTraits<CELLTYPE>::N_internal_states);
					}
					if (reset){
						resetHiddenState();
						unrolled_steps = delay + output_steps;
					} 	  

					IndexType curr_step = 0;
					IndexType start_col_old = 0;
					IndexType start_col = RNNCellTraits<CELLTYPE>::N_internal_states;
					// unrolling steps with only inputs (and hidden states)
					for (; curr_step < N_steps_only_input; ++curr_step){
						cell.step_input_to_hidden_unroll( 	input.col(curr_step), 
															internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
						start_col_old = start_col;
						start_col += RNNCellTraits<CELLTYPE>::N_internal_states;
					}
					IndexType curr_output_step = 0;
					if ( delay > size_t(input.cols()) ){
						// unrolling steps with only hidden states
						for (; curr_step < delay; ++curr_step){
							cell.step_hidden_to_hidden_unroll( 	internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
																internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
																internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
							start_col_old = start_col;
							start_col += RNNCellTraits<CELLTYPE>::N_internal_states;
						}
					}else{
						// unrolling steps with input, hidden and output
						for (; (curr_step < input.cols()) && (curr_step < output_steps); ++curr_step){
							cell.step_input_to_hidden_unroll( 	input.col(curr_step), 
																internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
																internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
																internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
							cell.step_hidden_to_output_unroll( 	internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
																outputs_pre_activation.col(curr_output_step),
																outputs.col(curr_output_step));
							++curr_output_step;
							start_col_old = start_col;
							start_col += RNNCellTraits<CELLTYPE>::N_internal_states;
						}
					}
					// unrolling hidden states and outputs
					for (; curr_output_step < output_steps; ++curr_output_step){
						cell.step_hidden_to_hidden_unroll( 	internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
						cell.step_hidden_to_output_unroll( 	internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states),
															outputs_pre_activation.col(curr_output_step),
															outputs.col(curr_output_step));
						start_col_old = start_col;
						start_col += RNNCellTraits<CELLTYPE>::N_internal_states;
					}

				}
				// NOTE: valid use only after unrolling
				void bptt( Eigen::Ref< MLVector<WeightType> > _grad_weights, const Eigen::Ref< const MLMatrix<WeightType> > grad_output ){
					MLEARN_ASSERT( grad_output.cols() == unrolled_output_steps, "Expected a different number of output gradients! (It might be that you haven't unrolled the network for the current data point)" );
					MLEARN_ASSERT( grad_temp.size() == _grad_weights.size(), "Expected gradient vector of different size" );
					// initialize to zero
					grad_temp.setZero();
					_grad_weights.setZero();
					cell.attachGradWeights(grad_temp, input_size,hidden_size,output_size);
					// 
					int outer_iteration = 0;
					for ( int out_step = unrolled_output_steps - 1; out_step >= 0; --out_step ){

						IndexType hidden_step = unrolled_steps - outer_iteration;
						IndexType hidden_step_old = hidden_step - 1;
						IndexType start_col = hidden_step*RNNCellTraits<CELLTYPE>::N_internal_states;
						IndexType start_col_old = start_col - RNNCellTraits<CELLTYPE>::N_internal_states;

						// backpropagate the output error
						cell.compute_grad_hidden_output( 	grad_output.col(out_step), 
															outputs_pre_activation.col(out_step), 
															outputs.col(out_step), 
															internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states) );
						
						cell.updateGradientOutputWeights(_grad_weights,grad_temp);
						cell.compute_hidden_gradient_from_output(grad_hidden,internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));

						// backpropagate through unrolled steps with only hidden states
						while ((hidden_step > unrolled_input_steps) && (hidden_step>0)){

							cell.compute_grad_hidden_hidden( 	grad_hidden, 
																internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states), 
																internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states), 
																internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states) );
							cell.updateGradientHiddenWeights(_grad_weights,grad_temp);
							cell.compute_hidden_gradient_from_hidden(grad_hidden,internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
							hidden_step = hidden_step_old;
							--hidden_step_old;
							start_col = start_col_old;
							start_col_old -= RNNCellTraits<CELLTYPE>::N_internal_states;
						}	
						
						// backpropagate through unrolled steps with also input steps
						while ( hidden_step > 0 ){

							cell.compute_grad_hidden_hidden( 	grad_hidden, 
																internal_state_pre_activation.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states), 
																internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states), 
																internal_state.block(0,start_col_old,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states) );
							cell.compute_grad_input_hidden( inputs.col(hidden_step_old) );
							cell.updateGradientHiddenWeights(_grad_weights,grad_temp);
							cell.updateGradientInputWeights(_grad_weights,grad_temp);
							cell.compute_hidden_gradient_from_hidden(grad_hidden,internal_state.block(0,start_col,hidden_size,RNNCellTraits<CELLTYPE>::N_internal_states));
							
							hidden_step = hidden_step_old;
							--hidden_step_old;
							start_col = start_col_old;
							start_col_old -= RNNCellTraits<CELLTYPE>::N_internal_states;
						}

						++outer_iteration;
					}
				}
				// NOTE: valid use only after unrolling
				const Eigen::Ref< const MLVector<WeightType>> getLastOutput() const{
					MLEARN_ASSERT( unrolled_output_steps > 0, "RNN was not unrolled enough before this query!" );
					//MLEARN_WARNING_MESSAGE( "This function returns a valid result only if unrolling has been performed first!" );
					return outputs.col(unrolled_output_steps-1);
				}
				// NOTE: valid use only after unrolling
				const Eigen::Ref< const MLMatrix<WeightType>> getAllOutputs() const{

					MLEARN_ASSERT( unrolled_output_steps > 0, "RNN was not unrolled enough before this query!" );
					//MLEARN_WARNING_MESSAGE( "This function returns a valid result only if unrolling has been performed first!" );
					return outputs.leftCols(unrolled_output_steps);
				}
				// NOTE: valid use only after unrolling
				template < typename U_INT >
				const Eigen::Ref< const MLMatrix<WeightType>> getOutput(U_INT step_idx) const{
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Step index must be unsigned integer!" );
					MLEARN_ASSERT( unrolled_output_steps > 0, "RNN was not unrolled enough before this query!" );
					MLEARN_ASSERT( unrolled_output_steps > step_idx, "RNN was not unrolled enough before this query!" );
					//MLEARN_WARNING_MESSAGE( "This function returns a valid result only if unrolling has been performed first!" );
					return outputs.col(step_idx);
				}
				// NOTE: valid use only after unrolling
				const Eigen::Ref< const MLMatrix<WeightType>> getLastHiddenState() const{
					MLEARN_ASSERT( unrolled_steps > 0, "RNN was not unrolled before this query!" );
					//MLEARN_WARNING_MESSAGE( "This function returns a valid result only if unrolling has been performed first!" );
					IndexType start_col = unrolled_steps*RNNCellTraits<CELLTYPE>::N_internal_states;
					return internal_state.block( 0,start_col, hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states );
				}
				// NOTE: valid use only after unrolling
				template < typename U_INT >
				const Eigen::Ref< const MLMatrix<WeightType>> getHiddenState(U_INT state_idx) const{
					static_assert( std::is_integral<U_INT>::value && std::is_unsigned<U_INT>::value, "Step index must be unsigned integer!" );
					MLEARN_ASSERT( unrolled_steps > 0, "RNN was not unrolled before this query!" );
					MLEARN_ASSERT( unrolled_steps > state_idx, "RNN was not unrolled enough before this query!" );
					//MLEARN_WARNING_MESSAGE( "This function returns a valid result only if unrolling has been performed first!" );
					IndexType start_col = (state_idx+1)*RNNCellTraits<CELLTYPE>::N_internal_states;
					return internal_state.block( 0,start_col, hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states );
				}
				const Eigen::Ref< const MLMatrix<WeightType>> getAllHiddenStates() const{
					MLEARN_ASSERT( unrolled_steps > 0, "RNN was not unrolled before this query!" );
					//MLEARN_WARNING_MESSAGE( "This function returns a valid result only if unrolling has been performed first!" );
					return internal_state.block( 0,0, hidden_size, (unrolled_steps+1)*RNNCellTraits<CELLTYPE>::N_internal_states );
				}
				const Eigen::Ref< const MLMatrix<WeightType>> getHiddenState() const{
					//MLEARN_WARNING_MESSAGE( "This function returns a valid result only if unrolling has been performed first!" );
					return internal_state.block( 0,0, hidden_size, RNNCellTraits<CELLTYPE>::N_internal_states );
				}
				IndexType getNWeights() const{
					return cell.computeNWeights(input_size, hidden_size, output_size);
				}
				const MLMatrix<WeightType>& getLastHiddenGradient() const{
					return grad_hidden;
				}
			private:
				// sizes
				const IndexType input_size, output_size, hidden_size;
				IndexType unrolled_input_steps=0;
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
				MLMatrix< WeightType > grad_hidden;
				// Cell - for specialized routines
				RecurrentImpl::RNNCell< WeightType,CELLTYPE,HIDDEN_ACTIVATION,OUTPUT_ACTIVATION > cell;

			};

		}

	}

}

#endif