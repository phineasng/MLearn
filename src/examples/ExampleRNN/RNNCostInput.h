#ifndef RNN_GRAD_CHECK_H
#define RNN_GRAD_CHECK_H

// MLearn includes
#include <MLearn/Core>
#include <MLearn/Optimization/CostFunction.h>
#include <MLearn/NeuralNets/RecurrentNets/RecurrentLayer.h>
#include <MLearn/NeuralNets/ActivationFunction.h>

namespace MLearn{

	namespace NeuralNets{

		namespace RecurrentNets{

			template <	LossType L,
						typename RNNLayer
						>
			class RNNInputGradCheck: public Optimization::CostFunction< RNNInputGradCheck<L,RNNLayer> >
			{
				typedef typename RNNLayer::scalar_t WeightType;
				typedef typename RNNLayer::index_t IndexType;
			public:
				static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,"IndexType has to be an unsigned integer!");
				static_assert(std::is_floating_point<WeightType>::value, "Scalar type has to be floating point!");
				// Constructors
				RNNInputGradCheck( RNNLayer& _rnn ):
					rnn_ref_(_rnn)
				{}
				void setOutput(const MLMatrix<WeightType>& _refTarget){
					target.resize(_refTarget.rows(),_refTarget.cols());
					target = _refTarget;
				}
				void setDelay(IndexType _d){
					delay = _d;
				}
				void setInputSize(IndexType _i_sz){
					input_sz = _i_sz;
				}
				void setNInputSamples(IndexType _n_i){
					input_samples = _n_i;
				}
				// evaluation
				template < 	typename DERIVED >
				typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,WeightType>::value, "Scalar types have to be the same!");
					static_assert(DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
					typename DERIVED::Scalar loss = typename DERIVED::Scalar(0);

					MLMatrix<WeightType> input(input_sz,input_samples);
					input.array() = x.array();
					rnn_ref_.forwardpass_unroll( 	input, 
													delay,	
													target.cols(),
													true);
					const Eigen::Ref< const MLMatrix<WeightType>> outputs = rnn_ref_.getAllOutputs();

					for ( size_t out_idx = 0; out_idx < target.cols(); ++out_idx ){

						loss += LossFunction<L>::evaluate( outputs.col(out_idx), target.col(out_idx) );

					}

					return loss;

				}
				// analytical gradient
				template < 	typename DERIVED,
							typename DERIVED_2 >
				void compute_analytical_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase< DERIVED_2 >& gradient ) const{
					static_assert(std::is_same<WeightType,typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value , "Scalar types have to be the same!");
					static_assert((DERIVED::ColsAtCompileTime == 1)&&(DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
					gradient = Eigen::MatrixBase< DERIVED_2 >::Zero(gradient.size());
					MLMatrix<WeightType> input(input_sz,input_samples);
					input.array() = x.array();
					rnn_ref_.forwardpass_unroll( 	input, 
													delay,	
													target.cols(),
													true);
					const Eigen::Ref< const MLMatrix<WeightType>> outputs = rnn_ref_.getAllOutputs();
					MLMatrix<WeightType> grad_output(rnn_ref_.getOutputSize(),outputs.cols());

					for ( size_t out_idx = 0; out_idx < size_t(outputs.cols()); ++out_idx ){

						Eigen::Ref< MLVector<WeightType> > curr_grad_out(grad_output.col(out_idx)); 
						LossFunction<L>::gradient( outputs.col(out_idx), target.col(out_idx), curr_grad_out);

					}
					MLMatrix<WeightType> input_gradient(input_sz,input_samples);
					MLVector<WeightType> grad_tmp(rnn_ref_.getNWeights());
					rnn_ref_.bptt( grad_tmp, grad_output,true );
					input_gradient = rnn_ref_.getInputGradient();
					gradient.array() = input_gradient.array();
				}
			private:
				RNNLayer& rnn_ref_;	
				MLMatrix< WeightType > target;
				IndexType delay;
				IndexType input_sz;
				IndexType input_samples;
				mutable MLMatrix< WeightType > hidden_state;

			};

		}

	}

}

#endif