#ifndef RECURRENT_CELL_TYPE_H_FILE_
#define RECURRENT_CELL_TYPE_H_FILE_

//MLearn includes
#include <MLearn/Core>
#include "../ActivationFunction.h"
#include "../CommonImpl.h"

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
				static const uint N_internal_states; 	// denote the number of internal states useful for cell computations, 
															// e.g. LSTM has 5 (hidden, preprocessed cell state, processed cell state, input gate result, forget gate result)
				static const uint N_cell_gradients; 	// denote the number of gradients to be backpropagated
			};

			template <>
			struct RNNCellTraits< RNNType::VANILLA >{
				static const uint N_internal_states = 1u; 	
				static const uint N_cell_gradients = 1u; 
			};

			template <>
			struct RNNCellTraits< RNNType::GRU >{
				static const uint N_internal_states = 4u;
				static const uint N_cell_gradients = 1u;
			};

			template <>
			struct RNNCellTraits< RNNType::LSTM >{
				static const uint N_internal_states = 6u;
				static const uint N_cell_gradients = 2u;
			};


			namespace RecurrentImpl{

				template < typename WeightType >
				class RNNCellBase{
				public:
					void updateGradientInputWeights( Eigen::Ref< MLVector< WeightType > > grad_weights, const Eigen::Ref< const MLVector< WeightType > > temp_grad_weights ) const{
						grad_weights.segment(offset_input_weights,n_input_weights) += temp_grad_weights.segment(offset_input_weights,n_input_weights);
					}
					void updateGradientHiddenWeights( Eigen::Ref< MLVector< WeightType > > grad_weights, const Eigen::Ref< const MLVector< WeightType > > temp_grad_weights ) const{
						grad_weights.segment(offset_hidden_weights,n_hidden_weights) += temp_grad_weights.segment(offset_hidden_weights,n_hidden_weights);
					}
					void updateGradientOutputWeights( Eigen::Ref< MLVector< WeightType > > grad_weights, const Eigen::Ref< const MLVector< WeightType > > temp_grad_weights ) const{
						grad_weights.segment(offset_output_weights,n_output_weights) += temp_grad_weights.segment(offset_output_weights,n_output_weights);
					}
					const Eigen::Ref< const MLVector< WeightType > > getGradientInputWeights( const Eigen::Ref< const MLVector< WeightType > > grad_weights ) const{
						return grad_weights.segment(offset_input_weights,n_input_weights);
					}
					const Eigen::Ref< const MLVector< WeightType > > getGradientHiddenWeights( const Eigen::Ref< const MLVector< WeightType > > grad_weights ) const{
						return grad_weights.segment(offset_hidden_weights,n_hidden_weights);
					}
					const Eigen::Ref< const MLVector< WeightType > > getGradientOutputWeights( const Eigen::Ref< const MLVector< WeightType > > grad_weights ) const{
						return grad_weights.segment(offset_output_weights,n_output_weights);
					}
				protected:
					size_t offset_input_weights = 0;
					size_t n_input_weights = 0;
					size_t offset_hidden_weights = 0;
					size_t n_hidden_weights = 0;
					size_t offset_output_weights = 0;
					size_t n_output_weights = 0;
					RNNCellBase() = default;
				};


				template < 	typename WeightType,
							RNNType TYPE = RNNType::VANILLA, 
							ActivationType HIDDEN_ACTIVATION = ActivationType::LOGISTIC, 
							ActivationType OUTPUT_ACTIVATION = ActivationType::LOGISTIC >
				class RNNCell: public RNNCellBase< WeightType >{};

				template < 	typename WeightType, 
							ActivationType HIDDEN_ACTIVATION, 
							ActivationType OUTPUT_ACTIVATION >
				class RNNCell<WeightType,RNNType::VANILLA,HIDDEN_ACTIVATION,OUTPUT_ACTIVATION>: public RNNCellBase< WeightType >{
					static_assert(std::is_floating_point< WeightType >::value, "The weights type has to be floating point!"); 
				public:
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
					void attachWeights( const Eigen::Ref< const MLVector<WeightType> > weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( weights.size() == computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" );

						UINT curr_index = 0;

						// input to hidden 
						this->offset_input_weights = curr_index;
						new (&W_in_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						this->n_input_weights = curr_index - this->offset_input_weights;

						// hidden to hidden 
						this->offset_hidden_weights = curr_index;
						new (&W_hid_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						new (&b_hid) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz;
						this->n_hidden_weights = curr_index - this->offset_hidden_weights;

						// hidden to output 
						this->offset_output_weights = curr_index;
						new (&W_hid_out) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&b_out) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, output_sz );
						curr_index += output_sz;
						this->n_output_weights = curr_index - this->offset_output_weights;

					}
					template < typename UINT >
					void attachGradWeights( Eigen::Ref< MLVector<WeightType> > grad_weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( grad_weights.size() == computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" );

						UINT curr_index = 0;

						// input to hidden 
						this->offset_input_weights = curr_index;
						new (&grad_W_in_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						this->n_input_weights = curr_index - this->offset_input_weights;

						// hidden to hidden 
						this->offset_hidden_weights = curr_index;
						new (&grad_W_hid_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						new (&grad_b_hid) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz;
						this->n_hidden_weights = curr_index - this->offset_hidden_weights;

						// hidden to output 
						this->offset_output_weights = curr_index;
						new (&grad_W_hid_out) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&grad_b_out) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, output_sz );
						curr_index += output_sz;
						this->n_output_weights = curr_index - this->offset_output_weights;

					}
					// compute - forward
					void step_input_output( const Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.rows() == output.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += W_in_hid*input + b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						output.noalias() = W_hid_out*hidden + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));
					}

					void step_input( const Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += W_in_hid*input + b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden( Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_output( Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_hid_out.rows() == output.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden = W_hid_hid*hidden;
						hidden.noalias() += b_hid;
						hidden = hidden.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						output.noalias() = W_hid_out*hidden + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}

					void step_input_to_hidden_unroll( 	const Eigen::Ref< const MLVector<WeightType> > input, 
														const Eigen::Ref< const MLMatrix<WeightType> > old_hidden, 
														Eigen::Ref< MLMatrix<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_in_hid.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden_pre_activation.size(), "Hidden (pre activation) size not consistent with current cell setting!" );

						hidden_pre_activation.noalias() = W_hid_hid*old_hidden + W_in_hid*input + b_hid;
						hidden = hidden_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden_to_hidden_unroll( 	const Eigen::Ref< const MLMatrix<WeightType> > old_hidden, 
														Eigen::Ref< MLMatrix<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_hid_out.cols() == hidden_pre_activation.size(), "Hidden (pre activation) size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_hid_out.cols() == hidden.size(), "Hidden size not consistent with current cell setting!" );

						hidden_pre_activation.noalias() = W_hid_hid*old_hidden + b_hid;
						hidden.noalias() = hidden_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden_to_output_unroll( 	const Eigen::Ref< const MLMatrix<WeightType> > hidden, 
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
					void compute_grad_hidden_output( 	const Eigen::Ref< const MLVector<WeightType> > grad_output, 
														const Eigen::Ref< const MLVector<WeightType> > output_pre_activation, 
														const Eigen::Ref< const MLVector<WeightType> > output, 
														const Eigen::Ref< const MLMatrix<WeightType> >  hidden ){

						InternalImpl::DerivativeWrapper<OUTPUT_ACTIVATION>::template derive<WeightType>(output_pre_activation,output,grad_b_out);
						grad_b_out = grad_output.cwiseProduct( grad_b_out );
						grad_W_hid_out = grad_b_out*hidden.transpose(); 

					}

					void compute_grad_hidden_hidden( 	const Eigen::Ref< const MLMatrix<WeightType> > grad_hidden, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden_pre_activation, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden_previous ){

						InternalImpl::DerivativeWrapper<HIDDEN_ACTIVATION>::template derive<WeightType>(hidden_pre_activation,hidden,grad_b_hid);
						grad_b_hid = grad_hidden.cwiseProduct( grad_b_hid );
						grad_W_hid_hid = grad_b_hid*hidden_previous.transpose(); 

					}

					void compute_grad_input_hidden( const Eigen::Ref< const MLVector<WeightType> > input ){

						grad_W_in_hid = grad_b_hid*input.transpose(); 

					}

					void compute_hidden_gradient_from_hidden(Eigen::Ref< MLMatrix<WeightType> > grad_hidden, const Eigen::Ref< const MLMatrix<WeightType> > hidde){
						
						grad_hidden = W_hid_hid.transpose()*grad_b_hid;

					}

					void compute_hidden_gradient_from_output(Eigen::Ref< MLMatrix<WeightType> > grad_hidden, const Eigen::Ref< const MLMatrix<WeightType> > hidden ){
						
						grad_hidden = W_hid_out.transpose()*grad_b_out;

					}

				private:

					// mapped weights
					Eigen::Map< const MLMatrix<WeightType> > W_in_hid;
					Eigen::Map< const MLMatrix<WeightType> > W_hid_hid;
					Eigen::Map< const MLMatrix<WeightType> > W_hid_out;
					Eigen::Map< const MLVector<WeightType> > b_hid;
					Eigen::Map< const MLVector<WeightType> > b_out;
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
							ActivationType HIDDEN_ACTIVATION, 
							ActivationType OUTPUT_ACTIVATION >
				class RNNCell< WeightType,RNNType::LSTM,HIDDEN_ACTIVATION,OUTPUT_ACTIVATION >: public RNNCellBase< WeightType >{
					static_assert(std::is_floating_point< WeightType >::value, "The weights type has to be floating point!"); 
				public:
					// Constructor
					RNNCell():
						W_input_in(NULL,0,0),
						W_input_hid(NULL,0,0),
						b_input(NULL,0),
						W_forget_in(NULL,0,0),
						W_forget_hid(NULL,0,0),
						b_forget(NULL,0),
						W_candidate_in(NULL,0,0),
						W_candidate_hid(NULL,0,0),
						b_candidate(NULL,0),
						W_output_in(NULL,0,0),
						W_output_hid(NULL,0,0),
						b_output(NULL,0),
						W_out(NULL,0,0),
						b_out(NULL,0),
						grad_W_input_in(NULL,0,0),
						grad_W_input_hid(NULL,0,0),
						grad_b_input(NULL,0),
						grad_W_forget_in(NULL,0,0),
						grad_W_forget_hid(NULL,0,0),
						grad_b_forget(NULL,0),
						grad_W_candidate_in(NULL,0,0),
						grad_W_candidate_hid(NULL,0,0),
						grad_b_candidate(NULL,0),
						grad_W_output_in(NULL,0,0),
						grad_W_output_hid(NULL,0,0),
						grad_b_output(NULL,0),
						grad_W_out(NULL,0,0),
						grad_b_out(NULL,0)
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
					void attachWeights( const Eigen::Ref< const MLVector<WeightType> > weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( weights.size() == computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" );

						UINT curr_index = 0;

						// INPUT TO HIDDEN
						this->offset_input_weights = curr_index;
						// input gate - input
						new (&W_input_in) Eigen::Map< const MLMatrix< WeightType > >( weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// forget gate - input
						new (&W_forget_in) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// candidate gate - input
						new (&W_candidate_in) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// output gate - input
						new (&W_output_in) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						this->n_input_weights = curr_index - this->offset_input_weights;

						// HIDDEN TO HIDDEN
						this->offset_hidden_weights = curr_index;
						// input gate - hidden
						new (&W_input_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// input gate - bias
						new (&b_input) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// forget gate - hidden
						new (&W_forget_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// forget gate - bias
						new (&b_forget) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// candidate gate - hidden
						new (&W_candidate_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// candidate gate - bias
						new (&b_candidate) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// output gate - hidden
						new (&W_output_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// output gate - bias
						new (&b_output) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						this->n_hidden_weights = curr_index - this->offset_hidden_weights;

						// HIDDEN TO OUTPUT
						this->offset_output_weights = curr_index;
						new (&W_out) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&b_out) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, output_sz );
						curr_index += output_sz;
						this->n_output_weights = curr_index - this->offset_output_weights ;

					}
					template < typename UINT >
					void attachGradWeights( Eigen::Ref< MLVector<WeightType> > grad_weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( grad_weights.size() == computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" );

						UINT curr_index = 0;

						// INPUT TO HIDDEN
						this->offset_input_weights = curr_index;
						// input gate - input
						new (&grad_W_input_in) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// forget gate - input
						new (&grad_W_forget_in) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// candidate gate - input
						new (&grad_W_candidate_in) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// output gate - input
						new (&grad_W_output_in) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						this->n_input_weights = curr_index - this->offset_input_weights;

						// HIDDEN TO HIDDEN
						this->offset_hidden_weights = curr_index;
						// input gate - hidden
						new (&grad_W_input_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// input gate - bias
						new (&grad_b_input) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// forget gate - hidden
						new (&grad_W_forget_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// forget gate - bias
						new (&grad_b_forget) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// candidate gate - hidden
						new (&grad_W_candidate_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// candidate gate - bias
						new (&grad_b_candidate) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// output gate - hidden
						new (&grad_W_output_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// output gate - bias
						new (&grad_b_output) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						this->n_hidden_weights = curr_index - this->offset_hidden_weights;

						// HIDDEN TO OUTPUT
						this->offset_output_weights = curr_index;
						new (&grad_W_out) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&grad_b_out) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, output_sz );
						curr_index += output_sz;
						this->n_output_weights = curr_index - this->offset_output_weights ;

						cell_state_gradient.resize(hidden_sz);
					}
					// compute - forward
					void step_input_output( const Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > i(hidden.col(input_idx));
						Eigen::Ref< MLVector<WeightType> > f(hidden.col(forget_idx));
						Eigen::Ref< MLVector<WeightType> > C_tilde(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > C(hidden.col(cell_idx));
						Eigen::Ref< MLVector<WeightType> > o(hidden.col(output_idx));

						f.noalias() = W_forget_in*input + W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i = W_input_in*input + W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde = W_candidate_in*input + W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.array() = i.array()*C_tilde.array() + f.array()*C.array();

						o = W_output_in*input + W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));;

						output = W_out*h + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));
						
					}

					void step_input( const Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > i(hidden.col(input_idx));
						Eigen::Ref< MLVector<WeightType> > f(hidden.col(forget_idx));
						Eigen::Ref< MLVector<WeightType> > C_tilde(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > C(hidden.col(cell_idx));
						Eigen::Ref< MLVector<WeightType> > o(hidden.col(output_idx));

						f.noalias() = W_forget_in*input + W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i.noalias() = W_input_in*input + W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde.noalias() = W_candidate_in*input + W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.array() = i.array()*C_tilde.array() + f.array()*C.array();

						o.noalias() = W_output_in*input + W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_hidden( Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > i(hidden.col(input_idx));
						Eigen::Ref< MLVector<WeightType> > f(hidden.col(forget_idx));
						Eigen::Ref< MLVector<WeightType> > C_tilde(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > C(hidden.col(cell_idx));
						Eigen::Ref< MLVector<WeightType> > o(hidden.col(output_idx));

						f.noalias() = W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i.noalias() = W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde.noalias() = W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.array() = i.array()*C_tilde.array() + f.array()*C.array();

						o.noalias() = W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

					}

					void step_output( Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > i(hidden.col(input_idx));
						Eigen::Ref< MLVector<WeightType> > f(hidden.col(forget_idx));
						Eigen::Ref< MLVector<WeightType> > C_tilde(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > C(hidden.col(cell_idx));
						Eigen::Ref< MLVector<WeightType> > o(hidden.col(output_idx));

						f.noalias() = W_forget_hid*h + b_forget;
						f = f.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i.noalias() = W_input_hid*h + b_input;
						i = i.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde.noalias() = W_candidate_hid*h + b_candidate;
						C_tilde = C_tilde.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.array() = i.array()*C_tilde.array() + f.array()*C.array();

						o.noalias() = W_output_hid*h + b_output;
						o = o.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						output.noalias() = W_out*h + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}

					void step_input_to_hidden_unroll( 	const Eigen::Ref< const MLVector<WeightType> > input, 
														const Eigen::Ref< const MLMatrix<WeightType> > old_hidden, 
														Eigen::Ref< MLMatrix<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_input_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(old_hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden_pre_activation.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
					
						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > h_pre(hidden_pre_activation.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_h(old_hidden.col(hidden_idx));

						Eigen::Ref< MLVector<WeightType> > i(hidden.col(input_idx));
						Eigen::Ref< MLVector<WeightType> > i_pre(hidden_pre_activation.col(input_idx));
						
						Eigen::Ref< MLVector<WeightType> > f(hidden.col(forget_idx));
						Eigen::Ref< MLVector<WeightType> > f_pre(hidden_pre_activation.col(forget_idx));

						Eigen::Ref< MLVector<WeightType> > C_tilde(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > C_tilde_pre(hidden_pre_activation.col(candidate_idx));

						Eigen::Ref< MLVector<WeightType> > C(hidden.col(cell_idx));
						Eigen::Ref< MLVector<WeightType> > C_post(hidden_pre_activation.col(cell_idx)); // the pre-activation cell memory has to be saved, hence we use the preactivation internal state to actually store the post-activation
						const Eigen::Ref< const MLVector<WeightType> > old_C(old_hidden.col(cell_idx));

						Eigen::Ref< MLVector<WeightType> > o(hidden.col(output_idx));
						Eigen::Ref< MLVector<WeightType> > o_pre(hidden_pre_activation.col(output_idx));

						f_pre.noalias() = W_forget_in*input + W_forget_hid*old_h + b_forget;
						f.noalias() = f_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i_pre.noalias() = W_input_in*input + W_input_hid*old_h + b_input;
						i.noalias() = i_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde_pre.noalias() = W_candidate_in*input + W_candidate_hid*old_h + b_candidate;
						C_tilde.noalias() = C_tilde_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.array() = i.array()*C_tilde.array() + f.array()*old_C.array();
						C_post.array() = C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						o_pre.noalias() = W_output_in*input + W_output_hid*old_h + b_output;
						o.noalias() = o_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C_post.array();

					}

					void step_hidden_to_hidden_unroll( 	const Eigen::Ref< const MLMatrix<WeightType> > old_hidden, 
														Eigen::Ref< MLMatrix<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(old_hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden_pre_activation.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
					
						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > h_pre(hidden_pre_activation.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_h(old_hidden.col(hidden_idx));

						Eigen::Ref< MLVector<WeightType> > i(hidden.col(input_idx));
						Eigen::Ref< MLVector<WeightType> > i_pre(hidden_pre_activation.col(input_idx));
						
						Eigen::Ref< MLVector<WeightType> > f(hidden.col(forget_idx));
						Eigen::Ref< MLVector<WeightType> > f_pre(hidden_pre_activation.col(forget_idx));

						Eigen::Ref< MLVector<WeightType> > C_tilde(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > C_tilde_pre(hidden_pre_activation.col(candidate_idx));

						Eigen::Ref< MLVector<WeightType> > C(hidden.col(cell_idx));
						Eigen::Ref< MLVector<WeightType> > C_post(hidden_pre_activation.col(cell_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_C(old_hidden.col(cell_idx));

						Eigen::Ref< MLVector<WeightType> > o(hidden.col(output_idx));
						Eigen::Ref< MLVector<WeightType> > o_pre(hidden_pre_activation.col(output_idx));

						f_pre.noalias() = W_forget_hid*old_h + b_forget;
						f.noalias() = f_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						i_pre.noalias() = W_input_hid*old_h + b_input;
						i.noalias() = i_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						C_tilde_pre.noalias() = W_candidate_hid*old_h + b_candidate;
						C_tilde.noalias() = C_tilde_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						C.array() = i.array()*C_tilde.array() + f.array()*old_C.array();
						C_post.array() = C.array().unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						o_pre.noalias() = W_output_hid*old_h + b_output;
						o.noalias() = o_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						h.array() = o.array()*C_post.array();
					}

					void step_hidden_to_output_unroll( 	const Eigen::Ref< const MLMatrix<WeightType> > hidden, 
														Eigen::Ref< MLVector<WeightType> > output_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > output ) const{

						MLEARN_ASSERT( W_out.rows() == output_pre_activation.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output (pre activation) size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );

						const Eigen::Ref< const MLVector<WeightType> > h(hidden.col(hidden_idx));
						output_pre_activation.noalias() = W_out*h + b_out;
						output.noalias() = output_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}


					// compute - gradient backprop
					// NOTE: for the correct use of this function, attachWeights and attachGradWeights have to be called first
					void compute_grad_hidden_output( 	const Eigen::Ref< const MLVector<WeightType> > grad_output, 
														const Eigen::Ref< const MLVector<WeightType> > output_pre_activation, 
														const Eigen::Ref< const MLVector<WeightType> > output, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden ){

						InternalImpl::DerivativeWrapper< OUTPUT_ACTIVATION >::template derive<WeightType>(output_pre_activation,output,grad_b_out);
						grad_b_out = grad_output.cwiseProduct( grad_b_out );

						const Eigen::Ref< const MLVector<WeightType> > h(hidden.col(hidden_idx));

						grad_W_out = grad_b_out*h.transpose(); 

					}

					void compute_grad_hidden_hidden( 	Eigen::Ref< MLMatrix<WeightType> > grad_hidden, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden_pre_activation, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden, 
														const Eigen::Ref< const MLMatrix<WeightType> > old_hidden ){

						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(old_hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden_pre_activation.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_input_in.rows())*RNNCellTraits< RNNType::LSTM >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
					
						const Eigen::Ref< const MLVector<WeightType> > h(hidden.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > h_pre(hidden_pre_activation.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_h(old_hidden.col(hidden_idx));

						const Eigen::Ref< const MLVector<WeightType> > i(hidden.col(input_idx));
						const Eigen::Ref< const MLVector<WeightType> > i_pre(hidden_pre_activation.col(input_idx));
						
						const Eigen::Ref< const MLVector<WeightType> > f(hidden.col(forget_idx));
						const Eigen::Ref< const MLVector<WeightType> > f_pre(hidden_pre_activation.col(forget_idx));

						const Eigen::Ref< const MLVector<WeightType> > C_tilde(hidden.col(candidate_idx));
						const Eigen::Ref< const MLVector<WeightType> > C_tilde_pre(hidden_pre_activation.col(candidate_idx));

						const Eigen::Ref< const MLVector<WeightType> > C(hidden.col(cell_idx));
						const Eigen::Ref< const MLVector<WeightType> > C_post(hidden_pre_activation.col(cell_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_C(old_hidden.col(cell_idx));

						const Eigen::Ref< const MLVector<WeightType> > o(hidden.col(output_idx));
						const Eigen::Ref< const MLVector<WeightType> > o_pre(hidden_pre_activation.col(output_idx));

						InternalImpl::DerivativeWrapper<HIDDEN_ACTIVATION>::template derive<WeightType>(C,C_post,cell_state_gradient);
						cell_state_gradient = cell_state_gradient.cwiseProduct(o);
						cell_state_gradient = cell_state_gradient.cwiseProduct(grad_hidden.col(hidden_idx));
						grad_hidden.col(cell_idx) += cell_state_gradient;

						InternalImpl::DerivativeWrapper<ActivationType::LOGISTIC>::template derive<WeightType>(o_pre,o,grad_b_output);
						grad_b_output = grad_b_output.cwiseProduct( C_post );
						grad_b_output = grad_b_output.cwiseProduct( grad_hidden.col(hidden_idx) );
						grad_W_output_hid = grad_b_output*old_h.transpose(); 

						InternalImpl::DerivativeWrapper<HIDDEN_ACTIVATION>::template derive<WeightType>(C_tilde_pre,C_tilde,grad_b_candidate);
						InternalImpl::DerivativeWrapper<ActivationType::LOGISTIC>::template derive<WeightType>(i_pre,i,grad_b_input);
						InternalImpl::DerivativeWrapper<ActivationType::LOGISTIC>::template derive<WeightType>(f_pre,f,grad_b_forget);

						grad_b_candidate = grad_b_candidate.cwiseProduct( i );
						grad_b_candidate = grad_b_candidate.cwiseProduct( grad_hidden.col(cell_idx) );
						grad_W_candidate_hid = grad_b_candidate*old_h.transpose();

						grad_b_input = grad_b_input.cwiseProduct( C_tilde );
						grad_b_input = grad_b_input.cwiseProduct( grad_hidden.col(cell_idx) );
						grad_W_input_hid = grad_b_input*old_h.transpose();

						grad_b_forget = grad_b_forget.cwiseProduct( old_C );
						grad_b_forget = grad_b_forget.cwiseProduct( grad_hidden.col(cell_idx) );
						grad_W_forget_hid = grad_b_forget*old_h.transpose();

					}

					void compute_grad_input_hidden( const Eigen::Ref< const MLVector<WeightType> > input ){

						grad_W_output_in = grad_b_output*input.transpose(); 
						grad_W_candidate_in = grad_b_candidate*input.transpose();
						grad_W_input_in = grad_b_input*input.transpose();
						grad_W_forget_in = grad_b_forget*input.transpose();

					}

					void compute_hidden_gradient_from_hidden(Eigen::Ref< MLMatrix<WeightType> > grad_hidden, const Eigen::Ref< const MLMatrix<WeightType> > hidden ){
						
						const Eigen::Ref< const MLVector<WeightType> > f(hidden.col(forget_idx));
						
						grad_hidden.col(hidden_idx) = 	W_output_hid.transpose()*grad_b_output +
														W_candidate_hid.transpose()*grad_b_candidate +
														W_input_hid.transpose()*grad_b_input + 
														W_forget_hid.transpose()*grad_b_forget;
						grad_hidden.col(cell_idx).array() *= f.array();

					}

					void compute_hidden_gradient_from_output(Eigen::Ref< MLMatrix<WeightType> > grad_hidden, const Eigen::Ref< const MLMatrix<WeightType> > hidden ){
						
						grad_hidden.setZero();
						grad_hidden.col(hidden_idx) = W_out.transpose()*grad_b_out;


					}
				private:
					// state/gates indexing
					static const size_t hidden_idx = 0,	
										cell_idx = 1,
										candidate_idx = 2,
										forget_idx = 3,
										input_idx = 4,
										output_idx = 5;


					// mapped weights
					Eigen::Map< const MLMatrix<WeightType> > W_input_in;
					Eigen::Map< const MLMatrix<WeightType> > W_input_hid;
					Eigen::Map< const MLVector<WeightType> > b_input;
					Eigen::Map< const MLMatrix<WeightType> > W_forget_in;
					Eigen::Map< const MLMatrix<WeightType> > W_forget_hid;
					Eigen::Map< const MLVector<WeightType> > b_forget;
					Eigen::Map< const MLMatrix<WeightType> > W_candidate_in;
					Eigen::Map< const MLMatrix<WeightType> > W_candidate_hid;
					Eigen::Map< const MLVector<WeightType> > b_candidate;
					Eigen::Map< const MLMatrix<WeightType> > W_output_in;
					Eigen::Map< const MLMatrix<WeightType> > W_output_hid;
					Eigen::Map< const MLVector<WeightType> > b_output;
					Eigen::Map< const MLMatrix<WeightType> > W_out;
					Eigen::Map< const MLVector<WeightType> > b_out;
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
					MLVector<WeightType> cell_state_gradient;
				};

				// GRU specialization
				// order of layers: hidden, preprocessed cell state, processed cell state, input gate result, forget gate result
				template < 	typename WeightType, 
							ActivationType HIDDEN_ACTIVATION, 
							ActivationType OUTPUT_ACTIVATION >
				class RNNCell< WeightType,RNNType::GRU,HIDDEN_ACTIVATION,OUTPUT_ACTIVATION >: public RNNCellBase< WeightType >{
					static_assert(std::is_floating_point< WeightType >::value, "The weights type has to be floating point!"); 
				public:
					// Constructor
					RNNCell():
						W_update_in(NULL,0,0),
						W_update_hid(NULL,0,0),
						b_update(NULL,0),
						W_reset_in(NULL,0,0),
						W_reset_hid(NULL,0,0),
						b_reset(NULL,0),
						W_candidate_in(NULL,0,0),
						W_candidate_hid(NULL,0,0),
						b_candidate(NULL,0),
						W_out(NULL,0,0),
						b_out(NULL,0),
						grad_W_update_in(NULL,0,0),
						grad_W_update_hid(NULL,0,0),
						grad_b_update(NULL,0),
						grad_W_reset_in(NULL,0,0),
						grad_W_reset_hid(NULL,0,0),
						grad_b_reset(NULL,0),
						grad_W_candidate_in(NULL,0,0),
						grad_W_candidate_hid(NULL,0,0),
						grad_b_candidate(NULL,0),
						grad_W_out(NULL,0,0),
						grad_b_out(NULL,0)
					{
						MLEARN_WARNING_MESSAGE( "Cell constructed but no weights attached yet!" );
					}
					// utils
					template < typename UINT >
					static UINT computeNWeights( UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						return (input_sz + hidden_sz + 1)*hidden_sz*3 + (hidden_sz + 1)*output_sz;
					}

					template < typename UINT >
					void attachWeights( const Eigen::Ref< const MLVector<WeightType> > weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( weights.size() == computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" );

						UINT curr_index = 0;

						// INPUT TO HIDDEN
						this->offset_input_weights = curr_index;
						// update gate - input
						new (&W_update_in) Eigen::Map< const MLMatrix< WeightType > >( weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// reset gate - input
						new (&W_reset_in) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// candidate gate - input
						new (&W_candidate_in) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						this->n_input_weights = curr_index - this->offset_input_weights;

						// HIDDEN TO HIDDEN
						this->offset_hidden_weights = curr_index;
						// update gate - hidden
						new (&W_update_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// update gate - bias
						new (&b_update) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// reset gate - hidden
						new (&W_reset_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// reset gate - bias
						new (&b_reset) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// candidate gate - hidden
						new (&W_candidate_hid) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// candidate gate - bias
						new (&b_candidate) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						this->n_hidden_weights = curr_index - this->offset_hidden_weights;

						// HIDDEN TO OUTPUT
						this->offset_output_weights = curr_index;
						new (&W_out) Eigen::Map< const MLMatrix< WeightType > >( weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&b_out) Eigen::Map< const MLVector< WeightType > >( weights.data() + curr_index, output_sz );
						curr_index += output_sz;
						this->n_output_weights = curr_index - this->offset_output_weights ;

					}
					template < typename UINT >
					void attachGradWeights( Eigen::Ref< MLVector<WeightType> > grad_weights, UINT input_sz, UINT hidden_sz, UINT output_sz ){
						static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Size values must be unsigned integers!" );
						// input weights might be part of a bigger array, hence we only require it to have enough allocated memory
						MLEARN_ASSERT( grad_weights.size() == computeNWeights(input_sz,hidden_sz,output_sz), "Dimensions not consistent!" );

						UINT curr_index = 0;

						// INPUT TO HIDDEN
						this->offset_input_weights = curr_index;
						// update gate - input
						new (&grad_W_update_in) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data(), hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// reset gate - input
						new (&grad_W_reset_in) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						// candidate gate - input
						new (&grad_W_candidate_in) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, input_sz );
						curr_index += input_sz*hidden_sz; 
						this->n_input_weights = curr_index - this->offset_input_weights;

						// HIDDEN TO HIDDEN
						this->offset_hidden_weights = curr_index;
						// update gate - hidden
						new (&grad_W_update_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// update gate - bias
						new (&grad_b_update) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// reset gate - hidden
						new (&grad_W_reset_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// reset gate - bias
						new (&grad_b_reset) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz; 
						// candidate gate - hidden
						new (&grad_W_candidate_hid) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, hidden_sz, hidden_sz );
						curr_index += hidden_sz*hidden_sz; 
						// candidate gate - bias
						new (&grad_b_candidate) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, hidden_sz );
						curr_index += hidden_sz;  
						this->n_hidden_weights = curr_index - this->offset_hidden_weights;

						// HIDDEN TO OUTPUT
						this->offset_output_weights = curr_index;
						new (&grad_W_out) Eigen::Map< MLMatrix< WeightType > >( grad_weights.data() + curr_index, output_sz, hidden_sz );
						curr_index += hidden_sz*output_sz; 
						new (&grad_b_out) Eigen::Map< MLVector< WeightType > >( grad_weights.data() + curr_index, output_sz );
						curr_index += output_sz;
						this->n_output_weights = curr_index - this->offset_output_weights ;

					}
					// compute - forward
					void step_input_output( const Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_update_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > u(hidden.col(update_idx));
						Eigen::Ref< MLVector<WeightType> > r(hidden.col(reset_idx));
						Eigen::Ref< MLVector<WeightType> > candidate(hidden.col(candidate_idx));

						u.noalias() = W_update_in*input + W_update_hid*h + b_update;
						u = u.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						r = W_reset_in*input + W_reset_hid*h + b_reset;
						r = r.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						candidate = W_candidate_in*input + W_candidate_hid*(h.cwiseProduct(r)) + b_candidate;
						candidate = candidate.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						h.array() = ( 1 - u.array() )*h.array() + candidate.array()*u.array();

						output = W_out*h + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));
						
					}

					void step_input( const Eigen::Ref< const MLVector<WeightType> > input, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_update_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > u(hidden.col(update_idx));
						Eigen::Ref< MLVector<WeightType> > r(hidden.col(reset_idx));
						Eigen::Ref< MLVector<WeightType> > candidate(hidden.col(candidate_idx));

						u.noalias() = W_update_in*input + W_update_hid*h + b_update;
						u = u.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						r = W_reset_in*input + W_reset_hid*h + b_reset;
						r = r.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						candidate = W_candidate_in*input + W_candidate_hid*(h.cwiseProduct(r)) + b_candidate;
						candidate = candidate.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						h.array() = ( 1 - u.array() )*h.array() + candidate.array()*u.array();

					}

					void step_hidden( Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > u(hidden.col(update_idx));
						Eigen::Ref< MLVector<WeightType> > r(hidden.col(reset_idx));
						Eigen::Ref< MLVector<WeightType> > candidate(hidden.col(candidate_idx));

						u.noalias() = W_update_hid*h + b_update;
						u = u.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						r = W_reset_hid*h + b_reset;
						r = r.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						candidate = W_candidate_hid*(h.cwiseProduct(r)) + b_candidate;
						candidate = candidate.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						h.array() = ( 1 - u.array() )*h.array() + candidate.array()*u.array();

					}

					void step_output( Eigen::Ref< MLVector<WeightType> > output, Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output size not consistent with current cell setting!" );

						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > u(hidden.col(update_idx));
						Eigen::Ref< MLVector<WeightType> > r(hidden.col(reset_idx));
						Eigen::Ref< MLVector<WeightType> > candidate(hidden.col(candidate_idx));

						u.noalias() = W_update_hid*h + b_update;
						u = u.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						r =W_reset_hid*h + b_reset;
						r = r.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						candidate = W_candidate_hid*(h.cwiseProduct(r)) + b_candidate;
						candidate = candidate.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						h.array() = ( 1 - u.array() )*h.array() + candidate.array()*u.array();

						output = W_out*h + b_out;
						output = output.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}

					void step_input_to_hidden_unroll( 	const Eigen::Ref< const MLVector<WeightType> > input, 
														const Eigen::Ref< const MLMatrix<WeightType> > old_hidden, 
														Eigen::Ref< MLMatrix<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( W_update_in.cols() == input.size(), "Input size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(old_hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden_pre_activation.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
					
						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > h_pre(hidden_pre_activation.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_h(old_hidden.col(hidden_idx));

						Eigen::Ref< MLVector<WeightType> > u(hidden.col(update_idx));
						Eigen::Ref< MLVector<WeightType> > u_pre(hidden_pre_activation.col(update_idx));
						
						Eigen::Ref< MLVector<WeightType> > r(hidden.col(reset_idx));
						Eigen::Ref< MLVector<WeightType> > r_pre(hidden_pre_activation.col(reset_idx));

						Eigen::Ref< MLVector<WeightType> > candidate(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > candidate_pre(hidden_pre_activation.col(candidate_idx));

						u_pre.noalias() = W_update_in*input + W_update_hid*old_h + b_update;
						u.noalias() = u_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						r_pre.noalias() = W_reset_in*input + W_reset_hid*old_h + b_reset;
						r.noalias() = r_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						candidate_pre.noalias() = W_candidate_in*input + W_candidate_hid*(old_h.cwiseProduct(r)) + b_candidate;
						candidate.noalias() = candidate_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						h.array() = ( 1 - u.array() )*old_h.array() + candidate.array()*u.array();

					}

					void step_hidden_to_hidden_unroll( 	const Eigen::Ref< const MLMatrix<WeightType> > old_hidden, 
														Eigen::Ref< MLMatrix<WeightType> > hidden_pre_activation, 
														Eigen::Ref< MLMatrix<WeightType> > hidden ) const{

						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(old_hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden_pre_activation.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
					
						Eigen::Ref< MLVector<WeightType> > h(hidden.col(hidden_idx));
						Eigen::Ref< MLVector<WeightType> > h_pre(hidden_pre_activation.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_h(old_hidden.col(hidden_idx));

						Eigen::Ref< MLVector<WeightType> > u(hidden.col(update_idx));
						Eigen::Ref< MLVector<WeightType> > u_pre(hidden_pre_activation.col(update_idx));
						
						Eigen::Ref< MLVector<WeightType> > r(hidden.col(reset_idx));
						Eigen::Ref< MLVector<WeightType> > r_pre(hidden_pre_activation.col(reset_idx));

						Eigen::Ref< MLVector<WeightType> > candidate(hidden.col(candidate_idx));
						Eigen::Ref< MLVector<WeightType> > candidate_pre(hidden_pre_activation.col(candidate_idx));

						u_pre.noalias() = W_update_hid*old_h + b_update;
						u.noalias() = u_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						r_pre.noalias() = W_reset_hid*old_h + b_reset;
						r.noalias() = r_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<ActivationType::LOGISTIC>::evaluate));

						candidate_pre.noalias() = W_candidate_hid*(old_h.cwiseProduct(r)) + b_candidate;
						candidate.noalias() = candidate_pre.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<HIDDEN_ACTIVATION>::evaluate));

						h.array() = ( 1 - u.array() )*old_h.array() + candidate.array()*u.array();
					}

					void step_hidden_to_output_unroll( 	const Eigen::Ref< const MLMatrix<WeightType> > hidden, 
														Eigen::Ref< MLVector<WeightType> > output_pre_activation, 
														Eigen::Ref< MLVector<WeightType> > output ) const{

						MLEARN_ASSERT( W_out.rows() == output_pre_activation.size(), "Output size not consistent with current cell setting!" );
						MLEARN_ASSERT( W_out.rows() == output.size(), "Output (pre activation) size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );

						const Eigen::Ref< const MLVector<WeightType> > h(hidden.col(hidden_idx));
						output_pre_activation.noalias() = W_out*h + b_out;
						output.noalias() = output_pre_activation.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<OUTPUT_ACTIVATION>::evaluate));

					}


					// compute - gradient backprop
					// NOTE: for the correct use of this function, attachWeights and attachGradWeights have to be called first
					void compute_grad_hidden_output( 	const Eigen::Ref< const MLVector<WeightType> > grad_output, 
														const Eigen::Ref< const MLVector<WeightType> > output_pre_activation, 
														const Eigen::Ref< const MLVector<WeightType> > output, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden ){

						InternalImpl::DerivativeWrapper< OUTPUT_ACTIVATION >::template derive<WeightType>(output_pre_activation,output,grad_b_out);
						grad_b_out = grad_output.cwiseProduct( grad_b_out );

						const Eigen::Ref< const MLVector<WeightType> > h(hidden.col(hidden_idx));

						grad_W_out = grad_b_out*h.transpose(); 

					}

					void compute_grad_hidden_hidden( 	const Eigen::Ref< const MLMatrix<WeightType> > grad_hidden, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden_pre_activation, 
														const Eigen::Ref< const MLMatrix<WeightType> > hidden, 
														const Eigen::Ref< const MLMatrix<WeightType> > old_hidden ){

						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(old_hidden.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden_pre_activation.size()), "Hidden size not consistent with current cell setting!" );
						MLEARN_ASSERT( size_t(W_update_in.rows())*RNNCellTraits< RNNType::GRU >::N_internal_states == size_t(hidden.size()), "Hidden size not consistent with current cell setting!" );
					
						const Eigen::Ref< const MLVector<WeightType> > h(hidden.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > h_pre(hidden_pre_activation.col(hidden_idx));
						const Eigen::Ref< const MLVector<WeightType> > old_h(old_hidden.col(hidden_idx));

						const Eigen::Ref< const MLVector<WeightType> > u(hidden.col(update_idx));
						const Eigen::Ref< const MLVector<WeightType> > u_pre(hidden_pre_activation.col(update_idx));
						
						const Eigen::Ref< const MLVector<WeightType> > r(hidden.col(reset_idx));
						const Eigen::Ref< const MLVector<WeightType> > r_pre(hidden_pre_activation.col(reset_idx));

						const Eigen::Ref< const MLVector<WeightType> > candidate(hidden.col(candidate_idx));
						const Eigen::Ref< const MLVector<WeightType> > candidate_pre(hidden_pre_activation.col(candidate_idx));

						InternalImpl::DerivativeWrapper<ActivationType::LOGISTIC>::template derive<WeightType>(u_pre,u,grad_b_update);
						grad_b_update = grad_b_update.cwiseProduct( candidate - old_h );
						grad_b_update = grad_b_update.cwiseProduct( grad_hidden );
						grad_W_update_hid = grad_b_update*old_h.transpose(); 

						InternalImpl::DerivativeWrapper<HIDDEN_ACTIVATION>::template derive<WeightType>(candidate_pre,candidate,grad_b_candidate);
						grad_b_candidate.array() = grad_b_candidate.array()*u.array();
						grad_b_candidate = grad_b_candidate.cwiseProduct( grad_hidden );
						grad_W_candidate_hid = grad_b_candidate*((old_h.cwiseProduct(r)).transpose());

						InternalImpl::DerivativeWrapper<ActivationType::LOGISTIC>::template derive<WeightType>(r_pre,r,grad_b_reset);
						grad_b_reset = grad_b_reset.cwiseProduct( old_h.cwiseProduct( W_candidate_hid.transpose()*grad_b_candidate ) );
						grad_W_reset_hid = grad_b_reset*old_h.transpose();

					}

					void compute_grad_input_hidden( const Eigen::Ref< const MLVector<WeightType> > input ){

						grad_W_update_in = grad_b_update*input.transpose(); 
						grad_W_reset_in = grad_b_reset*input.transpose();
						grad_W_candidate_in = grad_b_candidate*input.transpose();

					}

					void compute_hidden_gradient_from_hidden(Eigen::Ref< MLMatrix<WeightType> > grad_hidden, const Eigen::Ref< const MLMatrix<WeightType> > hidden){
						
						const Eigen::Ref< const MLVector<WeightType> > u(hidden.col(update_idx));
						const Eigen::Ref< const MLVector<WeightType> > candidate(hidden.col(reset_idx));
						const Eigen::Ref< const MLVector<WeightType> > r(hidden.col(reset_idx));

						grad_hidden.array() *= 1 - u.array();
						grad_hidden += W_update_hid.transpose()*grad_b_update;
						grad_hidden += (W_candidate_hid.transpose()*grad_b_candidate).cwiseProduct( r );
						grad_hidden += W_reset_hid.transpose()*grad_b_reset;

					}

					void compute_hidden_gradient_from_output(Eigen::Ref< MLMatrix<WeightType> > grad_hidden, const Eigen::Ref< const MLMatrix<WeightType> > hidden ){
						
						grad_hidden.col(hidden_idx) = W_out.transpose()*grad_b_out;

					}
				private:
					// state/gates indexing
					static const size_t hidden_idx = 0,
										update_idx = 1,
										reset_idx = 2,	
										candidate_idx = 3;


					// mapped weights
					Eigen::Map< const MLMatrix<WeightType> > W_update_in;
					Eigen::Map< const MLMatrix<WeightType> > W_update_hid;
					Eigen::Map< const MLVector<WeightType> > b_update;
					Eigen::Map< const MLMatrix<WeightType> > W_reset_in;
					Eigen::Map< const MLMatrix<WeightType> > W_reset_hid;
					Eigen::Map< const MLVector<WeightType> > b_reset;
					Eigen::Map< const MLMatrix<WeightType> > W_candidate_in;
					Eigen::Map< const MLMatrix<WeightType> > W_candidate_hid;
					Eigen::Map< const MLVector<WeightType> > b_candidate;
					Eigen::Map< const MLMatrix<WeightType> > W_out;
					Eigen::Map< const MLVector<WeightType> > b_out;
					// mapped gradient weights
					Eigen::Map< MLMatrix<WeightType> > grad_W_update_in;
					Eigen::Map< MLMatrix<WeightType> > grad_W_update_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_update;
					Eigen::Map< MLMatrix<WeightType> > grad_W_reset_in;
					Eigen::Map< MLMatrix<WeightType> > grad_W_reset_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_reset;
					Eigen::Map< MLMatrix<WeightType> > grad_W_candidate_in;
					Eigen::Map< MLMatrix<WeightType> > grad_W_candidate_hid;
					Eigen::Map< MLVector<WeightType> > grad_b_candidate;
					Eigen::Map< MLMatrix<WeightType> > grad_W_out;
					Eigen::Map< MLVector<WeightType> > grad_b_out;
					MLVector<WeightType> cell_state_gradient;
				};


			}

		}


	}

}

#endif