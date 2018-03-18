#ifndef SIMPLE_RNN_COST_FUNCTION_H
#define SIMPLE_RNN_COST_FUNCTION_H

// MLearn includes
#include <MLearn/Core>
#include <MLearn/Optimization/CostFunction.h>
#include "RecurrentLayer.h"
#include "../ActivationFunction.h"
#include <MLearn/Utility/DataInterface/SequentialDataInterface.h> 

namespace MLearn{

	namespace NeuralNets{

		namespace RecurrentNets{

			using namespace Utility::DataInterface;

			template <	LossType L,
						typename RNNLayer,
						typename DataInterface
						>
			class SimpleRNNCost: public Optimization::CostFunction< SimpleRNNCost<L,RNNLayer,DataInterface> >
			{
				typedef typename DataInterface::Scalar ScalarType;
				typedef typename DataInterface::Index IndexType;
			public:
				static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,"IndexType has to be an unsigned integer!");
				static_assert(std::is_floating_point<ScalarType>::value, "Scalar type has to be floating point!");
				static_assert(std::is_same<ScalarType,typename DataInterface::Scalar>::value, "Scalar types have to be the same!");
				// Constructors
				SimpleRNNCost( const SequentialDataInterface<ScalarType,IndexType,DataInterface>& _interface, RNNLayer& _rnn, MLVector<ScalarType>& _grad_tmp, MLMatrix<ScalarType>& _grad_out):
					data_interface_(_interface),
					rnn_ref_(_rnn),
					gradient_tmp(_grad_tmp),
					grad_output(_grad_out)
				{}
				// evaluation
				template < 	typename DERIVED >
				typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,ScalarType>::value, "Scalar types have to be the same!");
					static_assert(DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
					typename DERIVED::Scalar loss = typename DERIVED::Scalar(0);
					rnn_ref_.attachWeightsToCell(x);
					rnn_ref_.resetHiddenState();
					for ( size_t idx = 0; idx < data_interface_.getNSamples(); ++idx ){
						
						rnn_ref_.forwardpass_unroll( 	data_interface_.getInput(idx), 
														data_interface_.getDelay(idx), 
														data_interface_.getNOutputSteps(idx),
														data_interface_.getReset(idx));

						const Eigen::Ref< const MLMatrix<ScalarType>> outputs = rnn_ref_.getAllOutputs();
						const Eigen::Ref< const MLMatrix<ScalarType>> expected_outputs = data_interface_.getOutput(idx);

						for ( size_t out_idx = 0; out_idx < data_interface_.getNOutputSteps(idx); ++out_idx ){

							loss += LossFunction<L>::evaluate( outputs.col(out_idx), expected_outputs.col(out_idx) );

						}

					}
					loss /= ScalarType(data_interface_.getNSamples());
					// L1 or L2 regularization
					// TODO(phineasng)

					return loss;

				}
				// analytical gradient
				template < 	typename DERIVED,
							typename DERIVED_2 >
				void compute_analytical_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase< DERIVED_2 >& gradient ) const{
					static_assert(std::is_same<ScalarType,typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value , "Scalar types have to be the same!");
					static_assert((DERIVED::ColsAtCompileTime == 1)&&(DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
					gradient = Eigen::MatrixBase< DERIVED_2 >::Zero(gradient.size());
					gradient_tmp = gradient;
					rnn_ref_.attachWeightsToCell(x);
					rnn_ref_.resetHiddenState();
					for ( size_t idx = 0; idx < data_interface_.getNSamples(); ++idx ){
						compute_gradient_on_sample(idx);
						gradient += gradient_tmp;
					}
					gradient /= ScalarType(data_interface_.getNSamples());
					// L1 or L2 regularization
					// TODO(phineasng)
				}
				// stochastic gradient
				template < 	typename DERIVED,
							typename DERIVED_2 >
				void compute_stochastic_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase< DERIVED_2 >& gradient, const MLVector<IndexType>& indeces ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value , "Scalar types have to be the same!");
					static_assert((DERIVED::ColsAtCompileTime == 1)&&(DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
					gradient = Eigen::MatrixBase< DERIVED_2 >::Zero(gradient.size());
					gradient_tmp = gradient;
					rnn_ref_.attachWeightsToCell(x);
					rnn_ref_.resetHiddenState();

					for ( decltype(indeces.size()) idx = 0; idx < indeces.size(); ++idx ){
						compute_gradient_on_sample(indeces[idx]);
						gradient += gradient_tmp;
					}
					gradient /= ScalarType(indeces.size());
					// L1 or L2 regularization
					// TODO(phineasng)
				}
			private:
				const SequentialDataInterface<ScalarType,IndexType,DataInterface>& data_interface_;
				RNNLayer& rnn_ref_;
				MLVector< ScalarType >& gradient_tmp;		
				MLMatrix< ScalarType >& grad_output;
				// single sample gradient 
				inline void compute_gradient_on_sample( IndexType idx ) const{
					
					rnn_ref_.forwardpass_unroll( 	data_interface_.getInput(idx), 
													data_interface_.getDelay(idx), 
													data_interface_.getNOutputSteps(idx),
													data_interface_.getReset(idx));

					const Eigen::Ref< const MLMatrix<ScalarType>> outputs = rnn_ref_.getAllOutputs();
					const Eigen::Ref< const MLMatrix<ScalarType>> expected_outputs = data_interface_.getOutput(idx);

					if (data_interface_.getNOutputSteps(idx) > grad_output.cols()){
						grad_output.resize(grad_output.rows(),data_interface_.getNOutputSteps(idx));
					}


					for ( size_t out_idx = 0; out_idx < data_interface_.getNOutputSteps(idx); ++out_idx ){

						Eigen::Ref< MLVector<ScalarType> > curr_grad_out(grad_output.col(out_idx)); 
						LossFunction<L>::gradient( outputs.col(out_idx), expected_outputs.col(out_idx), curr_grad_out);

					}

					rnn_ref_.bptt( gradient_tmp, grad_output.leftCols(data_interface_.getNOutputSteps(idx)) );

				}
			};

		}

	}

}

#endif