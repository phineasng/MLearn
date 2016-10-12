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
			class RNNGradCheck: public Optimization::CostFunction< RNNGradCheck<L,RNNLayer> >
			{
				typedef typename RNNLayer::scalar_t WeightType;
				typedef typename RNNLayer::index_t IndexType;
			public:
				static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,"IndexType has to be an unsigned integer!");
				static_assert(std::is_floating_point<WeightType>::value, "Scalar type has to be floating point!");
				// Constructors
				RNNGradCheck( RNNLayer& _rnn ):
					rnn_ref_(_rnn)
				{}
				void setOutput(const MLVector<WeightType>& _refTarget){
					target.resize(_refTarget.size());
					target = _refTarget;
				}
				void setDelay(IndexType _d){
					delay = _d;
				}
				// evaluation
				template < 	typename DERIVED >
				typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,WeightType>::value, "Scalar types have to be the same!");
					static_assert(DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
					typename DERIVED::Scalar loss = typename DERIVED::Scalar(0);

					rnn_ref_.resetHiddenState();
					hidden_state.resize( rnn_ref_.getHiddenSize(), RNNCellTraits<RNNLayer::CellType>::N_internal_states );
					hidden_state.col(0) = x;
					rnn_ref_.setHiddenState(hidden_state);
					MLVector<WeightType> output(rnn_ref_.getOutputSize());
					for (IndexType i = 0; i < delay;++i){
						rnn_ref_.forwardpass_step();
					}
					output = rnn_ref_.forwardpass_step_output();
					loss = LossFunction<L>::evaluate( output, target );

					return loss;

				}
				// analytical gradient
				template < 	typename DERIVED,
							typename DERIVED_2 >
				void compute_analytical_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase< DERIVED_2 >& gradient ) const{
					static_assert(std::is_same<WeightType,typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value , "Scalar types have to be the same!");
					static_assert((DERIVED::ColsAtCompileTime == 1)&&(DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
					gradient = Eigen::MatrixBase< DERIVED_2 >::Zero(gradient.size());
					rnn_ref_.resetHiddenState();
					hidden_state.resize( rnn_ref_.getHiddenSize(), RNNCellTraits<RNNLayer::CellType>::N_internal_states );
					hidden_state.col(0) = x;
					rnn_ref_.setHiddenState(hidden_state);
					MLMatrix< WeightType > input( rnn_ref_.getInputSize(),0 );
					rnn_ref_.forwardpass_unroll( input, delay, 1u, false );
					const Eigen::Ref< const MLMatrix<WeightType>> outputs = rnn_ref_.getAllOutputs();
					MLVector<WeightType> grad_output(rnn_ref_.getOutputSize());
					LossFunction<L>::gradient( outputs.col(0), target, grad_output);
					MLVector< WeightType > gradient_tmp(rnn_ref_.getNWeights());
					rnn_ref_.bptt( gradient_tmp, grad_output );
					gradient = rnn_ref_.getLastHiddenGradient().col(0);
					//std::cout << outputs.col(0).transpose() << std::endl;
				}
			private:
				RNNLayer& rnn_ref_;	
				MLVector< WeightType > target;
				IndexType delay;
				mutable MLMatrix< WeightType > hidden_state;

			};

		}

	}

}

#endif