#ifndef MLEARN_FULLY_CONNECTED_COST_FUNCTION_INCLUDE
#define MLEARN_FULLY_CONNECTED_COST_FUNCTION_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include <MLearn/Optimization/CostFunction.h>
#include "../../Regularizer.h"
#include "../../ActivationFunction.h"
#include "FCNetsExplorer.h"

// STL includes
#include <type_traits>

// Useful macro for (a lil bit more user friendly) cost construction
#define TEMPLATED_FC_NEURAL_NET_COST_CONSTRUCTION( loss,reg,arg0,arg1,arg2,arg3,arg4,arg5,arg6,cost_variable_name )\
		MLearn::NeuralNets::FeedForwardNets::FCCostFunction<	loss,\
																reg,\
																decltype(arg3)::HiddenActivationType,\
																decltype(arg3)::OutputActivationType,\
																typename std::remove_reference<decltype(arg0[0])>::type,\
																decltype(arg1),\
																decltype(arg2) >\
			cost_variable_name(arg0,arg1,arg2,arg3,arg4,arg5,arg6)


#define TEMPLATED_FC_NEURAL_NET_COST_CONSTRUCTION_WITH_SHARED_WEIGHTS( loss,reg,arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,cost_variable_name )\
		MLearn::NeuralNets::FeedForwardNets::FCCostFunction<	loss,\
																reg,\
																decltype(arg3)::HiddenActivationType,\
																decltype(arg3)::OutputActivationType,\
																typename std::remove_reference<decltype(arg0[0])>::type,\
																decltype(arg1),\
																decltype(arg2) >\
			cost_variable_name(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8)

namespace MLearn{

	namespace NeuralNets{

		namespace FeedForwardNets{


			/*!
			*	\brief		Dummy class for additional fields
			*	\author		phineasng
			*
			*/
			template< 	Regularizer R,
						typename IndexType >
			class AdditionalFields{
			public:
				AdditionalFields(){}
				AdditionalFields( const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& shared_weights, const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& transposed_shared_weights ){
					static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,"IndexType has to be an unsigned integer!");
				}
			};

			template< typename IndexType >
			class AdditionalFields< Regularizer::SHARED_WEIGHTS, IndexType >{
			public:
				AdditionalFields( const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& _shared_weights, const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& _transposed_shared_weights ):
					shared_weights(_shared_weights),
					transposed_shared_weights(_transposed_shared_weights)
				{
					static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,"IndexType has to be an unsigned integer!");
				}
				const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& shared_weights; 
				const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& transposed_shared_weights;
			};

			/*!
			*	\brief		L2 Regulator - helper (of the helper) class
			*	\author 	phineasng
			*/
			template < bool R >
			struct L2Regulator{
				template < 	typename DERIVED >
				static inline void apply_to_loss_segment( const Eigen::MatrixBase< DERIVED >& x_segment, typename DERIVED::Scalar& loss ){
					static_assert( DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar type has to be floating point!" );
				}
				template < 	typename DERIVED,
							typename DERIVED_2 >
				static inline void apply_to_gradient_segment( const Eigen::MatrixBase< DERIVED >& x_segment, Eigen::VectorBlock< DERIVED_2 > gradient_segment, const typename DERIVED::Scalar& factor ){
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
				}
			};
			template <>
			struct L2Regulator<true>{
				template < 	typename DERIVED >
				static inline void apply_to_loss_segment( const Eigen::MatrixBase< DERIVED >& x_segment, typename DERIVED::Scalar& loss ){
					static_assert( DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar type has to be floating point!" );
					loss += x_segment.dot(x_segment);
				}
				template < 	typename DERIVED,
							typename DERIVED_2 >
				static inline void apply_to_gradient_segment( const Eigen::MatrixBase< DERIVED >& x_segment, Eigen::VectorBlock< DERIVED_2 > gradient_segment, const typename DERIVED::Scalar& factor ){
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
					gradient_segment += typename DERIVED::Scalar(2)*factor*x_segment;
				}
			};

			/*!
			*	\brief		L1 Regulator - helper (of the helper) class
			*	\author 	phineasng
			*/
			template < bool R >
			struct L1Regulator{
				template < 	typename DERIVED >
				static inline void apply_to_loss_segment( const Eigen::MatrixBase< DERIVED >& x_segment, typename DERIVED::Scalar& loss ){
					static_assert( DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar type has to be floating point!" );
				}
				template < 	typename DERIVED,
							typename DERIVED_2 >
				static inline void apply_to_gradient_segment( const Eigen::MatrixBase< DERIVED >& x_segment, Eigen::VectorBlock< DERIVED_2 > gradient_segment, const typename DERIVED::Scalar& factor ){
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
				}
			};
			template <>
			struct L1Regulator<true>{
				template < 	typename DERIVED >
				static inline void apply_to_loss_segment( const Eigen::MatrixBase< DERIVED >& x_segment, typename DERIVED::Scalar& loss ){
					static_assert( DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar type has to be floating point!" );
					loss += x_segment.array().abs().sum();
				}
				template < 	typename DERIVED,
							typename DERIVED_2 >
				static inline void apply_to_gradient_segment( const Eigen::MatrixBase< DERIVED >& x_segment, Eigen::VectorBlock< DERIVED_2 > gradient_segment, const typename DERIVED::Scalar& factor ){
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
					gradient_segment += factor*x_segment.unaryExpr( std::pointer_to_unary_function< typename DERIVED::Scalar, typename DERIVED::Scalar >(ml_signum) );
				}
			};

			/*!
			*	\brief		Shared weights regulator - helper (of the helper) class
			*	\details	Shared weights information are considered correct and not repeated. Therefore no check is performed here.
			*	\author 	phineasng
			*/
			template < bool R >
			struct SharedRegulator{
				template < 	typename DERIVED,
							Regularizer REG,
							typename IndexType >
				static inline void process_gradient( Eigen::MatrixBase< DERIVED >& gradient, const MLVector<IndexType>& layers, const AdditionalFields<REG,IndexType>& additional_fields ){
					static_assert( (DERIVED::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar types have to be the same and floating point!" );
					static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value, "IndexType has to be unsigned integer!");
				}
			};

			
			template <>
			struct SharedRegulator<true>{
				template < 	typename DERIVED,
							Regularizer REG,
							typename IndexType >
				static inline void process_gradient( Eigen::MatrixBase< DERIVED >& gradient, const MLVector<IndexType>& layers, const AdditionalFields<REG,IndexType>& additional_fields ){
					static_assert( (DERIVED::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar types have to be the same and floating point!" );
					static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value, "IndexType has to be unsigned integer!");

					IndexType offset_1 = 0, offset_2 = 0;
					const auto& shared = additional_fields.shared_weights;
					const auto& tr_shared = additional_fields.transposed_shared_weights;

					auto compute_offset = [&layers]( IndexType idx ){
						return layers.head( idx ).dot( layers.segment( 1, idx ) ) + layers.segment( 1, idx ).array().sum(); 
					};

					Eigen::Map< MLMatrix<typename DERIVED::Scalar> > weights_1(gradient.segment(offset_1,0).data(),0,0);
					Eigen::Map< MLMatrix<typename DERIVED::Scalar> > weights_2(gradient.segment(offset_2,0).data(),0,0);

					if (shared.cols() > 0){
						for ( decltype(shared.cols()) idx = 0; idx < shared.cols(); ++idx){
					
							offset_1 = compute_offset( shared(0,idx) );
							offset_2 = compute_offset( shared(1,idx) );
					
							new (&weights_1) Eigen::Map< MLMatrix<typename DERIVED::Scalar> >(gradient.segment(offset_1,0).data(),layers[ shared(0,idx) + 1 ],layers[ shared(0,idx) ]);
							new (&weights_2) Eigen::Map< MLMatrix<typename DERIVED::Scalar> >(gradient.segment(offset_2,0).data(),layers[ shared(1,idx) + 1 ],layers[ shared(1,idx) ]);
					
							weights_1 += weights_2;
							weights_2 = weights_1;
						}
					}

					if (tr_shared.cols() > 0){
						for ( decltype(tr_shared.cols()) idx = 0; idx < tr_shared.cols(); ++idx){

							offset_1 = compute_offset( tr_shared(0,idx) );
							offset_2 = compute_offset( tr_shared(1,idx) );

							new (&weights_1) Eigen::Map< MLMatrix<typename DERIVED::Scalar> >(gradient.segment(offset_1,0).data(),layers[ tr_shared(0,idx) + 1 ],layers[ tr_shared(0,idx) ]);
							new (&weights_2) Eigen::Map< MLMatrix<typename DERIVED::Scalar> >(gradient.segment(offset_2,0).data(),layers[ tr_shared(1,idx) + 1 ],layers[ tr_shared(1,idx) ]);

							weights_1 += weights_2.transpose();
							weights_2 = weights_1.transpose();
						}
					}

					return;
				}
			};

			/*!
			*	\brief 		Regulator - helper class
			*	\author 	phineasng
			*
			*/
			template < bool V >
			struct Regulator{
				template < 	typename IndexType,
							Regularizer R,
							typename DERIVED >
				static inline void apply_to_loss( const MLVector< IndexType >& layers, const Eigen::MatrixBase< DERIVED >& x, typename DERIVED::Scalar& loss, const RegularizerOptions<typename DERIVED::Scalar>& options ){
					static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value, "IndexType has to be unsigned integer!");
					static_assert( DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar type has to be floating point!" );
				}
				template < 	typename IndexType,
							Regularizer R,
							typename DERIVED,
							typename DERIVED_2 >
				static inline void apply_to_gradient( const MLVector< IndexType >& layers, const Eigen::MatrixBase< DERIVED >& x, Eigen::MatrixBase< DERIVED_2 >& gradient, const RegularizerOptions<typename DERIVED::Scalar>& options ){
					static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value, "IndexType has to be unsigned integer!");
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
				}
			};
			template <>
			struct Regulator< true >{
				template < 	typename IndexType,
							Regularizer R,
							typename DERIVED >
				static inline void apply_to_loss( const MLVector< IndexType >& layers, const Eigen::MatrixBase< DERIVED >& x, typename DERIVED::Scalar& loss, const RegularizerOptions<typename DERIVED::Scalar>& options ){
					static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value, "IndexType has to be unsigned integer!");
					static_assert( DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar type has to be floating point!" );
					typename DERIVED::Scalar lossL1 = typename DERIVED::Scalar(0);
					typename DERIVED::Scalar lossL2 = typename DERIVED::Scalar(0);
					decltype(layers.size()) offset = 0;
					decltype(layers.size()) tmp = 0;
					decltype(layers.size()) idx_p1;

					for (decltype(layers.size()) idx = 0; idx < layers.size() - 1; ++idx ){

						idx_p1 = idx+1;
						tmp = layers[idx]*layers[idx_p1];

						L1Regulator< (R & Regularizer::L1) == Regularizer::L1 >::apply_to_loss_segment( x.segment(offset,tmp) , lossL1 );
						L2Regulator< (R & Regularizer::L2) == Regularizer::L2 >::apply_to_loss_segment( x.segment(offset,tmp) , lossL2 );

						offset += tmp + layers[idx_p1];

					}

					loss += options._l1_param*lossL1 + options._l2_param*lossL2;
					
				}
				template < 	typename IndexType,
							Regularizer R,
							typename DERIVED,
							typename DERIVED_2 >
				static inline void apply_to_gradient( const MLVector< IndexType >& layers, const Eigen::MatrixBase< DERIVED >& x, Eigen::MatrixBase< DERIVED_2 >& gradient, const RegularizerOptions<typename DERIVED::Scalar>& options ){
					static_assert( std::is_unsigned<IndexType>::value && std::is_integral<IndexType>::value, "IndexType has to be unsigned integer!");
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
					static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
					decltype(layers.size()) offset = 0;
					decltype(layers.size()) tmp = 0;
					decltype(layers.size()) idx_p1;

					for (decltype(layers.size()) idx = 0; idx < layers.size() - 1; ++idx ){

						idx_p1 = idx+1;
						tmp = layers[idx]*layers[idx_p1];

						L1Regulator< (R & Regularizer::L1) == Regularizer::L1 >::apply_to_gradient_segment( x.segment(offset,tmp) , gradient.segment(offset,tmp), options._l1_param );
						L2Regulator< (R & Regularizer::L2) == Regularizer::L2 >::apply_to_gradient_segment( x.segment(offset,tmp) , gradient.segment(offset,tmp), options._l2_param );

						offset += tmp + layers[idx_p1];

					}
				}
			};


			/*!
			*	\brief		Cost functions for fully connected feed forward nets optimization (training)
			*	\author		phineasng
			*
			*/
			template< 	LossType L,
						Regularizer R,
						ActivationType HiddenLayerActivation,
						ActivationType OutputLayerActivation,
						typename IndexType,
						typename DERIVED,
						typename DERIVED_2 >
			class FCCostFunction: public Optimization::CostFunction< FCCostFunction< L,R,HiddenLayerActivation,OutputLayerActivation,IndexType,DERIVED,DERIVED_2 > >{
			public:
				static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value,"IndexType has to be an unsigned integer!");
				static_assert(std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value && std::is_floating_point<typename DERIVED::Scalar>::value, "Scalar types have to be the same and floating point!");
				// function constructor
				FCCostFunction( const MLVector< IndexType >& refLayers, 
								const Eigen::MatrixBase<DERIVED>& refInputs, 
								const Eigen::MatrixBase<DERIVED_2>& refOutputs, 
								FCNetsExplorer< typename DERIVED::Scalar,IndexType,HiddenLayerActivation,OutputLayerActivation >& refExplorer, 
								const RegularizerOptions<typename DERIVED::Scalar>& refOptions,
								MLVector< typename DERIVED::Scalar >& ref_allocated_grad_output,
								MLVector< typename DERIVED::Scalar >& ref_allocated_grad ):
					layers(refLayers),
					inputs(refInputs),
					outputs(refOutputs),
					options(refOptions),
					net_explorer(refExplorer),
					grad_output(ref_allocated_grad_output),
					gradient_tmp(ref_allocated_grad)
				{
					static_assert( (R & Regularizer::SHARED_WEIGHTS) != Regularizer::SHARED_WEIGHTS,"This constructor is available only if the shared weights are not used!");
					MLEARN_ASSERT( refInputs.cols() == refOutputs.cols(), "Inputs and corresponding outputs have to be the same number!" );
				}

				FCCostFunction( const MLVector< IndexType >& refLayers, 
								const Eigen::MatrixBase<DERIVED>& refInputs, 
								const Eigen::MatrixBase<DERIVED_2>& refOutputs, 
								FCNetsExplorer< typename DERIVED::Scalar,IndexType,HiddenLayerActivation,OutputLayerActivation >& refExplorer, 
								const RegularizerOptions<typename DERIVED::Scalar>& refOptions,
								MLVector< typename DERIVED::Scalar >& ref_allocated_grad_output,
								MLVector< typename DERIVED::Scalar >& ref_allocated_grad,
								const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& shared_weights,
								const Eigen::Matrix< IndexType, 2, -1, Eigen::ColMajor | Eigen::AutoAlign >& transposed_shared_weights ):
					layers(refLayers),
					inputs(refInputs),
					outputs(refOutputs),
					options(refOptions),
					net_explorer(refExplorer),
					grad_output(ref_allocated_grad_output),
					gradient_tmp(ref_allocated_grad),
					additional_fields(shared_weights,transposed_shared_weights)
				{
					MLEARN_ASSERT( refInputs.cols() == refOutputs.cols(), "Inputs and corresponding outputs have to be the same number!" );
				}
				// evaluation
				template < 	typename INNER_DERIVED >
				typename INNER_DERIVED::Scalar eval( const Eigen::MatrixBase<INNER_DERIVED>& x ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,typename INNER_DERIVED::Scalar>::value, "Scalar types have to be the same!");
					static_assert(INNER_DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
					typename INNER_DERIVED::Scalar loss = typename INNER_DERIVED::Scalar(0);
					
					for ( decltype(inputs.cols()) idx = 0; idx < inputs.cols(); ++idx ){
						net_explorer.FCNetsExplorer<typename DERIVED::Scalar,IndexType,HiddenLayerActivation,OutputLayerActivation>::template forwardpass< ( R & Regularizer::DROPOUT) == Regularizer::DROPOUT >( x, inputs.col(idx) );
						loss += LossFunction<L>::evaluate( net_explorer.getActivations().tail( layers[layers.size() - 1] ), outputs.col(idx));
					}
					loss /= inputs.cols();
					// L1 or L2 regularization
					Regulator< ((R & Regularizer::L1) == Regularizer::L1) || ((R & Regularizer::L2) == Regularizer::L2) >::template apply_to_loss<IndexType,R>( layers, x, loss, options );

					return loss;

				}
				// analytical gradient
				template < 	typename INNER_DERIVED,
							typename INNER_DERIVED_2 >
				void compute_analytical_gradient( const Eigen::MatrixBase<INNER_DERIVED>& x, Eigen::MatrixBase< INNER_DERIVED_2 >& gradient ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,typename INNER_DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename INNER_DERIVED_2::Scalar>::value , "Scalar types have to be the same!");
					static_assert((INNER_DERIVED::ColsAtCompileTime == 1)&&(INNER_DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
					gradient = Eigen::MatrixBase< INNER_DERIVED_2 >::Zero(gradient.size());
					gradient_tmp = gradient;

					for ( decltype(inputs.cols()) idx = 0; idx < inputs.cols(); ++idx ){
						compute_gradient_on_sample(x,gradient_tmp,idx);
						gradient += gradient_tmp;
					}
					gradient /= inputs.cols();
					// Shared weights regularization
					SharedRegulator< (R & Regularizer::SHARED_WEIGHTS) == Regularizer::SHARED_WEIGHTS >::process_gradient(gradient,layers,additional_fields);
					// L1 or L2 regularization
					Regulator< ((R & Regularizer::L1) == Regularizer::L1) || ((R & Regularizer::L2) == Regularizer::L2) >::template apply_to_gradient<IndexType,R>( layers, x, gradient, options );
				}
				// stochastic gradient
				template < 	typename INNER_DERIVED,
							typename INNER_DERIVED_2 >
				void compute_stochastic_gradient( const Eigen::MatrixBase<INNER_DERIVED>& x, Eigen::MatrixBase< INNER_DERIVED_2 >& gradient, const MLVector<IndexType>& indeces ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,typename INNER_DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename INNER_DERIVED_2::Scalar>::value , "Scalar types have to be the same!");
					static_assert((INNER_DERIVED::ColsAtCompileTime == 1)&&(INNER_DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
					gradient = Eigen::MatrixBase< INNER_DERIVED_2 >::Zero(gradient.size());
					gradient_tmp = gradient;

					for ( decltype(indeces.size()) idx = 0; idx < indeces.size(); ++idx ){
						compute_gradient_on_sample(x,gradient_tmp,indeces[idx]);
						gradient += gradient_tmp;
					}
					gradient /= indeces.size();
					// Shared weights regularization
					SharedRegulator< (R & Regularizer::SHARED_WEIGHTS) == Regularizer::SHARED_WEIGHTS >::process_gradient(gradient,layers,additional_fields);
					// L1 or L2 regularization
					Regulator< ((R & Regularizer::L1) == Regularizer::L1) || ((R & Regularizer::L2) == Regularizer::L2) >::template apply_to_gradient<IndexType,R>( layers, x, gradient, options );

				}
			private:
				const MLVector< IndexType >& layers;
				const Eigen::MatrixBase<DERIVED>& inputs;
				const Eigen::MatrixBase<DERIVED_2>& outputs;
				const RegularizerOptions<typename DERIVED::Scalar>& options;
				FCNetsExplorer< typename DERIVED::Scalar,IndexType,HiddenLayerActivation,OutputLayerActivation >& net_explorer;
				/// \details For safety (in referencing the needed objects) and more flexibility (using the eigen types),
				/// we plan to reconstruct this cost function class everytime a training algorithm is called.
				/// This means that, in an ONLINE or CROSSVALIDATION setting, the construction may incur in
				/// a not negligible overhead if this class allocates memory or has to build objects which allocates
				/// memory (e.g. Eigen objects).
				/// Therefore, we require the necessary temporaries to be built outside the class
				/// and a reference to them at construction.
				MLVector< typename DERIVED::Scalar >& grad_output;
				MLVector< typename DERIVED::Scalar >& gradient_tmp;		
				AdditionalFields< (R & Regularizer::SHARED_WEIGHTS), IndexType > additional_fields;
				// single sample gradient 
				template < 	typename INNER_DERIVED,
							typename INNER_DERIVED_2 >
				inline void compute_gradient_on_sample( const Eigen::MatrixBase< INNER_DERIVED >& x, Eigen::MatrixBase< INNER_DERIVED_2 >& gradient, IndexType idx ) const{
					static_assert(std::is_same<typename DERIVED::Scalar,typename INNER_DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar,typename INNER_DERIVED_2::Scalar>::value , "Scalar types have to be the same!");
					static_assert((INNER_DERIVED::ColsAtCompileTime == 1)&&(INNER_DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
					net_explorer.FCNetsExplorer<typename DERIVED::Scalar,IndexType,HiddenLayerActivation,OutputLayerActivation>::template forwardpass< ( R & Regularizer::DROPOUT) == Regularizer::DROPOUT >( x, inputs.col(idx) );
					LossFunction<L>::gradient( net_explorer.getActivations().tail( layers[layers.size() - 1] ), outputs.col(idx),grad_output );
					net_explorer.FCNetsExplorer<typename DERIVED::Scalar,IndexType,HiddenLayerActivation,OutputLayerActivation>::template backpropagate< ( R & Regularizer::DROPOUT) == Regularizer::DROPOUT >( x, grad_output, gradient );
				}
			};


		}
	
	}

}

#endif