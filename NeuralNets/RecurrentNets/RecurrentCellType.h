#ifndef RECURRENT_CELL_TYPE_H_FILE_
#define RECURRENT_CELL_TYPE_H_FILE_

//MLearn includes
#include <MLearn/Core>
#include "../../ActivationFunction.h"
#include "../../CommonImpl.h"

namespace MLearn{

	namespace NeuralNets{

		namespace RecurrentNets{

			enum class RNNType{
				VANILLA,
				LSTM,
				GRU
			};

			template < RNNType TYPE >
			struct RNNCellTraits{
				static const uint N_internal_states = 1; 	// denote the number of internal states useful for cell computations, 
															// e.g. LSTM has 5 (hidden, preprocessed cell state, processed cell state, input gate result, forget gate result)
			};

			template <>
			struct RNNCellTraits< RNNType::LSTM >{
				static const uint N_internal_states = 6;
			};

			namespace RecurrentImpl{

				template < 	typename WeightType,
							RNNType TYPE = RNNType::VANILLA, 
							ActivationType HIDDEN_ACTIVATION = ActivationType::LOGISTIC, 
							ActivationType OUTPUT_ACTIVATION = ActivationType::LOGISTIC >
				class RNNCell{
					static_assert(std::is_floating_point< WeightType >::value, "The weights type has to be floating point!"); 
					// Constructor
					RNNCell():
						W_in_hid(NULL,0,0),
						W_hid_hid(NULL,0,0),
						W_hid_out(NULL,0,0),
						b_hid(NULL,0),
						b_out(NULL,0),
						grad_W_in_hid(NULL,0,0),
						grad_W_hid_hid(NULL,0,0),
						grad_W_hid_out(NULL,0,0),
						grad_b_hid(NULL,0),
						grad_b_out(NULL,0)
					{
						MLEARN_WARNING_MESSAGE( "Cell constructed but no weights attached yet!" );
					}
					// utils
					template < typename UINT >
					static UINT computeNWeights( UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						return input_sz*hidden_sz + hidden_sz + hidden_sz*hidden_sz + hidden_sz*output_sz + output_sz;
					}

					template < typename UINT >
					void attachWeights( Eigen::Ref< MLVector<WeightType> > weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( weights.size() >= computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" )

						UINT curr_index = 0;

						// input to hidden 
						new (&W_in_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// hidden to hidden 
						new (&W_hid_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						new (&b_hid) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz;

						// hidden to output 
						new (&W_hid_out) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&b_out) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, output_sz );

					}
					template < typename UINT >
					void attachGradWeights( Eigen::Ref< MLVector<WeightType> > grad_weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( weights.size() >= computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" )

						UINT curr_index = 0;

						// input to hidden 
						new (&grad_W_in_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// hidden to hidden 
						new (&grad_W_hid_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						new (&grad_b_hid) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz;

						// hidden to output 
						new (&grad_W_hid_out) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&grad_b_out) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, output_sz );

					}
					// compute - forward
					void step_input_output( Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLVector<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.rows() == output.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += W_in_hid*input + b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						output.noalias() = W_hid_out*hidden + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));
					}

					void step_input( Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLVector<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.rows() == output.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += W_in_hid*input + b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden( Eigen::Ref< MLVector<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.rows() == output.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_output( Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLVector<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.rows() == output.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						output.noalias() = W_hid_out*hidden + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}

					void step_input_to_hidden_unroll( 	Eigen::Ref< const MLVector<WeightType> > input, 
														Eigen::Ref< const MLVector<WeightType> > old_hidden, 
														Eigen::Ref< MLVector<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden_pre_activation.size(), "Hidden (pre activation) size not consistent with current cell setting!" );

						hidden_pre_activation.noalias() = W_hid_hid*old_hidden + W_in_hid*input + b_hid;
						hidden = hidden_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden_to_hidden_unroll( 	Eigen::Ref< const MLVector<WeightType> > old_hidden, 
														Eigen::Ref< MLVector<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_hid_out.cols() == hidden_pre_activation.size(), "Hidden (pre activation) size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden_pre_activation.noalias() = W_hid_hid*old_hidden + b_hid;
						hidden.noalias() = hidden_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden_to_output_unroll( 	Eigen::Ref< const MLVector<WeightType> > hidden, 
														Eigen::Ref< MLVector<WeightType> > output_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > output ) const{

						MLEARN_ASSERT( W_hid_out.rows() == output_pre_activation.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.rows() == output.size(), "Output (pre activation) size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						output_pre_activation.noalias() = W_hid_out*hidden + b_out;
						output.noalias() = output_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}


					// compute - gradient backprop
					// NOTE: for the correct use of this function, attachWeights and attachGradWeights have to be called first
					void compute_grad_hidden_output( 	Eigen::Ref< const MLVector<WeightType> > grad_output, 
														Eigen::Ref< const MLVector<WeightType> > output_pre_activation, 
														Eigen::Ref< const MLVector<WeightType> > output, 
														Eigen::Ref< const MLVector<WeightType> > hidden ){

						InternalImpl::DerivativeWrapper<OUTPUT_ACTIVATION>::template derive<WeightType>(output_pre_activation,output,grad_b_out);
						grad_b_out = grad_output.cwiseProduct( grad_b_out );
						grad_W_hid_out = grad_b_out*hidden.transpose(); 

					}

					void compute_grad_hidden_hidden( 	Eigen::Ref< const MLVector<WeightType> > grad_hidden, 
														Eigen::Ref< const MLVector<WeightType> > hidden_pre_activation, 
														Eigen::Ref< const MLVector<WeightType> > hidden, 
														Eigen::Ref< const MLVector<WeightType> > hidden_previous ){

						InternalImpl::DerivativeWrapper<HIDDEN_ACTIVATION>::template derive<WeightType>(hidden_pre_activation,hidden,grad_b_hid);
						grad_b_hid = grad_hidden.cwiseProduct( grad_b_hid );
						grad_W_hid_hid = grad_b_hid*hidden_previous.transpose(); 

					}

					void compute_grad_input_hidden( 	Eigen::Ref< const MLVector<WeightType> > grad_hidden, 
														Eigen::Ref< const MLVector<WeightType> > input ){

						grad_W_in_hid = grad_b_hid*input.transpose(); 

					}

					void compute_grad_input_hidden_no_hid_precomp( 	Eigen::Ref< const MLVector<WeightType> > grad_hidden, 
																	Eigen::Ref< const MLVector<WeightType> > hidden_pre_activation, 
																	Eigen::Ref< const MLVector<WeightType> > hidden,
																	Eigen::Ref< const MLVector<WeightType> > input ){

						InternalImpl::DerivativeWrapper<HIDDEN_ACTIVATION>::template derive<WeightType>(hidden_pre_activation,hidden,grad_b_hid);
						grad_b_hid = grad_hidden.cwiseProduct( grad_b_hid );
						grad_W_in_hid = grad_b_hid*input.transpose(); 

					}

					// mapped weights
					Eigen::Map< MLMatrix<WeightType> > W_in_hid;
					Eigen::Map< MLMatrix<WeightType> > W_hid_hid;
					Eigen::Map< MLMatrix<WeightType> > W_hid_out;
					Eigen::Map< MLVector<WeightType> > b_hid;
					Eigen::Map< MLVector<WeightType> > b_out;
					// mapped gradient weights
					Eigen::Map< MLMatrix<WeightType> > grad_W_in_hid;
					Eigen::Map< MLMatrix<WeightType> > grad_W_hid_hid;
					Eigen::Map< MLMatrix<WeightType> > grad_W_hid_out;
					Eigen::Map< MLVector<WeightType> > grad_b_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_out;
				};

				// LSTM specialization
				// order of layers: hidden, preprocessed cell state, processed cell state, input gate result, forget gate result
				template < 	typename WeightType, 
							ActivationType HIDDEN_ACTIVATION = ActivationType::LOGISTIC, 
							ActivationType OUTPUT_ACTIVATION = ActivationType::LOGISTIC >
				class RNNCell< WeightType,RNNType::LSTM,HIDDEN_ACTIVATION,OUTPUT_ACTIVATION >{
					static_assert(std::is_floating_point< WeightType >::value, "The weights type has to be floating point!"); 
					// Constructor
					RNNCell():
						W_input_in(NULL,0,0),
						W_input_hid(NULL,0,0),
						b_input(NULL,0,0),
						W_forget_in(NULL,0,0),
						W_forget_hid(NULL,0,0),
						b_forget(NULL,0,0),
						W_candidate_in(NULL,0,0),
						W_candidate_hid(NULL,0,0),
						b_candidate(NULL,0,0),
						W_output_in(NULL,0,0),
						W_output_hid(NULL,0,0),
						b_output(NULL,0,0),
						W_out(NULL,0,0),
						b_out(NULL,0,0),
						grad_W_input_in(NULL,0,0),
						grad_W_input_hid(NULL,0,0),
						grad_b_input(NULL,0,0),
						grad_W_forget_in(NULL,0,0),
						grad_W_forget_hid(NULL,0,0),
						grad_b_forget(NULL,0,0),
						grad_W_candidate_in(NULL,0,0),
						grad_W_candidate_hid(NULL,0,0),
						grad_b_candidate(NULL,0,0),
						grad_W_output_in(NULL,0,0),
						grad_W_output_hid(NULL,0,0),
						grad_b_output(NULL,0,0),
						grad_W_out(NULL,0,0),
						grad_b_out(NULL,0,0)
					{
						MLEARN_WARNING_MESSAGE( "Cell constructed but no weights attached yet!" );
					}
					// utils
					template < typename UINT >
					static UINT computeNWeights( UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						return (input_sz + hidden_sz + 1)*hidden_sz*4 + (hidden_sz + 1)*output_sz;
					}

					template < typename UINT >
					void attachWeights( Eigen::Ref< MLVector<WeightType> > weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( weights.size() >= computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" )

						UINT curr_index = 0;

						// input gate - input
						new (&W_input_in) Eigen::Map< MLMatrix< WeightType > >( weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// input gate - hidden
						new (&W_input_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// input gate - bias
						new (&b_input) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// forget gate - input
						new (&W_forget_in) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// forget gate - hidden
						new (&W_forget_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// forget gate - bias
						new (&b_forget) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// candidate gate - input
						new (&W_candidate_in) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// candidate gate - hidden
						new (&W_candidate_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// candidate gate - bias
						new (&b_candidate) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// output gate - input
						new (&W_output_in) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// output gate - hidden
						new (&W_output_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// output gate - bias
						new (&b_output) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// hidden to output 
						new (&W_out) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 

						new (&b_out) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, output_sz );

					}
					template < typename UINT >
					void attachGradWeights( Eigen::Ref< MLVector<WeightType> > grad_weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( weights.size() >= computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" )

						UINT curr_index = 0;

						// input gate - input
						new (&grad_W_input_in) Eigen::Map< MLMatrix< WeightType > >( weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// input gate - hidden
						new (&grad_W_input_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// input gate - bias
						new (&grad_b_input) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// forget gate - input
						new (&grad_W_forget_in) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// forget gate - hidden
						new (&grad_W_forget_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// forget gate - bias
						new (&grad_b_forget) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// candidate gate - input
						new (&grad_W_candidate_in) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// candidate gate - hidden
						new (&grad_W_candidate_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// candidate gate - bias
						new (&grad_b_candidate) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// output gate - input
						new (&grad_W_output_in) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 

						// output gate - hidden
						new (&grad_W_output_hid) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 

						// output gate - bias
						new (&grad_b_output) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 

						// hidden to output 
						new (&grad_W_out) Eigen::Map< MLMatrix< WeightType > >( weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 

						new (&grad_b_out) Eigen::Map< MLVector< WeightType > >( weights.data() + curr_index, output_sz );

					}
					// compute - forward
					void step_input_output( Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLVector<WeightType> > hidden ) const{

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output size not consistent with current cell setting!" );

						Eigen::Map< MLVector<WeightType> > h(hidden.data() + hidden_sz*hidden_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > i(hidden.data() + hidden_sz*input_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > f(hidden.data() + hidden_sz*forget_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_tilde(hidden.data() + hidden_sz*preproc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C(hidden.data() + hidden_sz*proc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > o(hidden.data() + hidden_sz*output_idx,hidden_sz);

						f = W_forget_in*input + W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i = W_input_in*input + W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde = W_candidate_in*input + W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.array() = i.array()*C_tilde.array() + f.array()*C.array();

						o = W_output_in*input + W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						output = W_out*h + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));
						
					}

					void step_input( Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLVector<WeightType> > hidden ) const{

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );

						Eigen::Map< MLVector<WeightType> > h(hidden.data() + hidden_sz*hidden_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > i(hidden.data() + hidden_sz*input_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > f(hidden.data() + hidden_sz*forget_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_tilde(hidden.data() + hidden_sz*preproc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C(hidden.data() + hidden_sz*proc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > o(hidden.data() + hidden_sz*output_idx,hidden_sz);

						f.noalias() = W_forget_in*input + W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i.noalias() = W_input_in*input + W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde.noalias() = W_candidate_in*input + W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.noalias().array() = i.array()*C_tilde.array() + f.array()*C.array();

						o.noalias() = W_output_in*input + W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.noalias().array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden( Eigen::Ref< MLVector<WeightType> > hidden ) const{

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );

						Eigen::Map< MLVector<WeightType> > h(hidden.data() + hidden_sz*hidden_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > i(hidden.data() + hidden_sz*input_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > f(hidden.data() + hidden_sz*forget_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_tilde(hidden.data() + hidden_sz*preproc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C(hidden.data() + hidden_sz*proc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > o(hidden.data() + hidden_sz*output_idx,hidden_sz);

						f.noalias() = W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i.noalias() = W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde.noalias() = W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.noalias().array() = i.array()*C_tilde.array() + f.array()*C.array();

						o.noalias() = W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.noalias().array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_output( Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLVector<WeightType> > hidden ) const{

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output size not consistent with current cell setting!" );

						Eigen::Map< MLVector<WeightType> > h(hidden.data() + hidden_sz*hidden_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > i(hidden.data() + hidden_sz*input_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > f(hidden.data() + hidden_sz*forget_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_tilde(hidden.data() + hidden_sz*preproc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C(hidden.data() + hidden_sz*proc_cell_idx,hidden_sz);
						Eigen::Map< MLVector<WeightType> > o(hidden.data() + hidden_sz*output_idx,hidden_sz);

						f.noalias() = W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i.noalias() = W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde.noalias() = W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.noalias().array() = i.array()*C_tilde.array() + f.array()*C.array();

						o.noalias() = W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.noalias().array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						output.noalias() = W_out*h + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}

					void step_input_to_hidden_unroll( 	Eigen::Ref< const MLVector<WeightType> > input, 
														Eigen::Ref< const MLVector<WeightType> > old_hidden, 
														Eigen::Ref< MLVector<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > hidden ) const{

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == old_hidden.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden_pre_activation.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );
					
						size_t offset;

						offset = hidden_sz*hidden_idx;
						Eigen::Map< MLVector<WeightType> > h(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > h_pre(hidden_pre_activation.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > old_h(old_hidden.data() + offset,hidden_sz);

						offset = hidden_sz*input_idx;
						Eigen::Map< MLVector<WeightType> > i(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > i_pre(hidden_pre_activation.data() + offset,hidden_sz);
						
						offset = hidden_sz*forget_idx;
						Eigen::Map< MLVector<WeightType> > f(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > f_pre(hidden_pre_activation.data() + offset,hidden_sz);

						offset = hidden_sz*preproc_cell_idx;
						Eigen::Map< MLVector<WeightType> > C_tilde(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_tilde_pre(hidden_pre_activation.data() + offset,hidden_sz);

						offset = hidden_sz*proc_cell_idx;
						Eigen::Map< MLVector<WeightType> > C(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_pre(hidden_pre_activation.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > old_C(old_hidden.data() + offset,hidden_sz);

						offset = hidden_sz*output_idx;
						Eigen::Map< MLVector<WeightType> > o(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > o_pre(hidden_pre_activation.data() + offset,hidden_sz);

						f_pre.noalias() = W_forget_in*input + W_forget_hid*h + b_forget;
						f.noalias() = f_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i_pre.noalias() = W_input_in*input + W_input_hid*h + b_input;
						i.noalias() = i_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde_pre.noalias() = W_candidate_in*input + W_candidate_hid*h + b_candidate;
						C_tilde.noalias() = C_tilde_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C_pre.noalias().array() = i.array()*C_tilde.array() + f.array()*old_C.array();
						C.noalias().array() = C_pre.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						o_pre.noalias() = W_output_in*input + W_output_hid*h + b_output;
						o.noalias() = o_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C.array();

					}

					void step_hidden_to_hidden_unroll( 	Eigen::Ref< const MLVector<WeightType> > old_hidden, 
														Eigen::Ref< MLVector<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > hidden ) const{

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == old_hidden.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden_pre_activation.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );
					
						size_t offset;

						offset = hidden_sz*hidden_idx;
						Eigen::Map< MLVector<WeightType> > h(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > h_pre(hidden_pre_activation.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > old_h(old_hidden.data() + offset,hidden_sz);

						offset = hidden_sz*input_idx;
						Eigen::Map< MLVector<WeightType> > i(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > i_pre(hidden_pre_activation.data() + offset,hidden_sz);
						
						offset = hidden_sz*forget_idx;
						Eigen::Map< MLVector<WeightType> > f(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > f_pre(hidden_pre_activation.data() + offset,hidden_sz);

						offset = hidden_sz*preproc_cell_idx;
						Eigen::Map< MLVector<WeightType> > C_tilde(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_tilde_pre(hidden_pre_activation.data() + offset,hidden_sz);

						offset = hidden_sz*proc_cell_idx;
						Eigen::Map< MLVector<WeightType> > C(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > C_pre(hidden_pre_activation.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > old_C(old_hidden.data() + offset,hidden_sz);

						offset = hidden_sz*output_idx;
						Eigen::Map< MLVector<WeightType> > o(hidden.data() + offset,hidden_sz);
						Eigen::Map< MLVector<WeightType> > o_pre(hidden_pre_activation.data() + offset,hidden_sz);

						f_pre.noalias() = W_forget_hid*h + b_forget;
						f.noalias() = f_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i_pre.noalias() = W_input_hid*h + b_input;
						i.noalias() = i_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde_pre.noalias() = W_candidate_hid*h + b_candidate;
						C_tilde.noalias() = C_tilde_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C_pre.array().noalias() = i.array()*C_tilde.array() + f.array()*old_C.array();
						C.array().noalias() = C_pre.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						o_pre.noalias() = W_output_hid*h + b_output;
						o.noalias() = o_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array().noalias() = o.array()*C.array();
					}

					void step_hidden_to_output_unroll( 	Eigen::Ref< const MLVector<WeightType> > hidden, 
														Eigen::Ref< MLVector<WeightType> > output_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > output ) const{

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( W_out.rows() == output_pre_activation.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output (pre activation) size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );

						Eigen::Map< MLVector<WeightType> > h(hidden.data() + hidden_sz*hidden_idx,hidden_sz);
						output_pre_activation.noalias() = W_out*h + b_out;
						output.noalias() = output_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}


					// compute - gradient backprop
					// NOTE: for the correct use of this function, attachWeights and attachGradWeights have to be called first
					void compute_grad_hidden_output( 	Eigen::Ref< const MLVector<WeightType> > grad_output, 
														Eigen::Ref< const MLVector<WeightType> > output_pre_activation, 
														Eigen::Ref< const MLVector<WeightType> > output, 
														Eigen::Ref< const MLVector<WeightType> > hidden ){

						InternalImpl::DerivativeWrapper< OUTPUT_ACTIVATION >::template derive<WeightType>(output_pre_activation,output,grad_b_out);
						grad_b_out = grad_output.cwiseProduct( grad_b_out );

						size_t hidden_sz = W_input_in.rows();
						Eigen::Map< const MLVector<WeightType> > h(hidden.data() + hidden_sz*hidden_idx,hidden_sz);

						grad_W_out = grad_b_out*h.transpose(); 

					}

					void compute_grad_hidden_hidden( 	Eigen::Ref< const MLVector<WeightType> > grad_hidden, 
														Eigen::Ref< const MLVector<WeightType> > hidden_pre_activation, 
														Eigen::Ref< const MLVector<WeightType> > hidden, 
														Eigen::Ref< const MLVector<WeightType> > old_hidden ){

						size_t hidden_sz = W_input_in.rows();

						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == old_hidden.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden_pre_activation.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( hidden_sz*RNNCellTraits< RNNType::LSTM >::N_internal_states == hidden.size(), "Hidden size not consistent with current cell setting!" );
					
						size_t offset;

						offset = hidden_sz*hidden_idx;
						Eigen::Map< const MLVector<WeightType> > h(hidden.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > h_pre(hidden_pre_activation.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > old_h(old_hidden.data() + offset,hidden_sz);

						offset = hidden_sz*input_idx;
						Eigen::Map< const MLVector<WeightType> > i(hidden.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > i_pre(hidden_pre_activation.data() + offset,hidden_sz);
						
						offset = hidden_sz*forget_idx;
						Eigen::Map< const MLVector<WeightType> > f(hidden.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > f_pre(hidden_pre_activation.data() + offset,hidden_sz);

						offset = hidden_sz*preproc_cell_idx;
						Eigen::Map< const MLVector<WeightType> > C_tilde(hidden.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > C_tilde_pre(hidden_pre_activation.data() + offset,hidden_sz);

						offset = hidden_sz*proc_cell_idx;
						Eigen::Map< const MLVector<WeightType> > C(hidden.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > C_pre(hidden_pre_activation.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > old_C(old_hidden.data() + offset,hidden_sz);

						offset = hidden_sz*output_idx;
						Eigen::Map< const MLVector<WeightType> > o(hidden.data() + offset,hidden_sz);
						Eigen::Map< const MLVector<WeightType> > o_pre(hidden_pre_activation.data() + offset,hidden_sz);

						InternalImpl::DerivativeWrapper<ActivationType::LOGISTIC>::template derive<WeightType>(o_pre,o,grad_b_output);
						grad_b_hid = grad_hidden.cwiseProduct( grad_b_hid );
						grad_W_hid_hid = grad_b_hid*hidden_previous.transpose(); 

					}

					void compute_grad_input_hidden( 	Eigen::Ref< const MLVector<WeightType> > grad_hidden, 
														Eigen::Ref< const MLVector<WeightType> > input ){

						grad_W_in_hid = grad_b_hid*input.transpose(); 

					}

					void compute_grad_input_hidden_no_hid_precomp( 	Eigen::Ref< const MLVector<WeightType> > grad_hidden, 
																	Eigen::Ref< const MLVector<WeightType> > hidden_pre_activation, 
																	Eigen::Ref< const MLVector<WeightType> > hidden,
																	Eigen::Ref< const MLVector<WeightType> > input ){

						InternalImpl::DerivativeWrapper<ACTIVATION>::template derive<WeightType>(hidden_pre_activation,hidden,grad_b_out);
						grad_b_hid = grad_hidden.cwiseProduct( grad_b_hid );
						grad_W_in_hid = grad_b_hid*input.transpose(); 

					}

					// convenient typedefs
					typedef Eigen::Matrix< WeightType,-1,4 > HiddenMatrix;
					// state/gates indexing
					static const size_t hidden_idx = 0,	
										preproc_cell_idx = 1,
										proc_cell_idx = 2,
										forget_idx = 3,
										input_idx = 4,
										output_idx = 5;
					// mapped weights
					Eigen::Map< MLMatrix<WeightType> > W_input_in;
					Eigen::Map< MLMatrix<WeightType> > W_input_hid;
					Eigen::Map< MLVector<WeightType> > b_input;
					Eigen::Map< MLMatrix<WeightType> > W_forget_in;
					Eigen::Map< MLMatrix<WeightType> > W_forget_hid;
					Eigen::Map< MLVector<WeightType> > b_forget;
					Eigen::Map< MLMatrix<WeightType> > W_candidate_in;
					Eigen::Map< MLMatrix<WeightType> > W_candidate_hid;
					Eigen::Map< MLVector<WeightType> > b_candidate;
					Eigen::Map< MLMatrix<WeightType> > W_output_in;
					Eigen::Map< MLMatrix<WeightType> > W_output_hid;
					Eigen::Map< MLVector<WeightType> > b_output;
					Eigen::Map< MLMatrix<WeightType> > W_out;
					Eigen::Map< MLVector<WeightType> > b_out;
					// mapped gradient weights
					Eigen::Map< MLMatrix<WeightType> > grad_W_input_in;
					Eigen::Map< MLMatrix<WeightType> > grad_W_input_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_input;
					Eigen::Map< MLMatrix<WeightType> > grad_W_forget_in;
					Eigen::Map< MLMatrix<WeightType> > grad_W_forget_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_forget;
					Eigen::Map< MLMatrix<WeightType> > grad_W_candidate_in;
					Eigen::Map< MLMatrix<WeightType> > grad_W_candidate_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_candidate;
					Eigen::Map< MLMatrix<WeightType> > grad_W_output_in;
					Eigen::Map< MLMatrix<WeightType> > grad_W_output_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_output;
					Eigen::Map< MLMatrix<WeightType> > grad_W_out;
					Eigen::Map< MLVector<WeightType> > grad_b_out;
				};


			}

		}


	}

}

#endif