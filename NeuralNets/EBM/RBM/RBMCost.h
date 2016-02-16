#ifndef MLEARN_RBM_COST_INCLUDE
#define MLEARN_RBM_COST_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include <MLearn/NeuralNets/Regularizer.h>
#include <MLearn/Optimization/CostFunction.h>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include "RBMUtils.h"
#include "RBMProcessing.h"

// STL includes
#include <cmath>

namespace MLearn{

	namespace NeuralNets{

		namespace RBMSupport{

			/*
			*	\brief Hidden distribution integral computer
			*/
			template < RBMUnitType HID_TYPE >
			class HiddenIntegral{
			public:
				template < typename RBM, typename DERIVED >
				static inline typename RBM::SCALAR compute_integral( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBM& rbm ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBM::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					typename RBM::SCALAR integral = 0;
					auto& bias_hidden = rbm.bias_hidden;
					auto binary_op = [](const typename RBM::SCALAR& s1, const typename RBM::SCALAR& s2){
						return static_cast<typename RBM::SCALAR>( std::log(1+std::exp(s1+s2)) );
					};

					integral = -(interaction_terms.binaryExpr(bias_hidden,binary_op).sum());

					return integral;
				}
				template < typename RBMCOST >
				using DerReturnType = decltype(RBMCOST::bias_hidden);
				template < typename RBMCOST, typename DERIVED >
				static inline DerReturnType<RBMCOST> compute_hidden_derivative( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBMCOST& cost ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBMCOST::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					auto& rbm = cost.rbm;
					auto& bias_hidden = rbm.bias_hidden;
					cost.bias_hidden = -((interaction_terms+bias_hidden).unaryExpr(std::pointer_to_unary_function<typename RBMCOST::SCALAR,typename RBMCOST::SCALAR>(ActivationFunction<ActivationType::LOGISTIC>::evaluate)));
					return cost.bias_hidden;
				}

			};

			/*
			*	\brief	RBM free energy. 
			*/
			template < RBMUnitType VIS_TYPE,RBMUnitType HID_TYPE >
			class FreeEnergy{
			public:
				template< typename RBMCOST, typename DERIVED >
				static inline typename RBMCOST::SCALAR compute_energy( const Eigen::MatrixBase<DERIVED>& input, const RBMCOST& cost){
					typename RBMCOST::SCALAR energy = 0;
					auto& rbm = cost.rbm;
					auto& visible_bias = rbm.bias_visible;
					auto& weights = rbm.weights;

					energy -= input.dot(visible_bias);
					energy += HiddenIntegral< HID_TYPE >::compute_integral( weights*input, rbm );

					return energy;
				}
				template< typename RBMCOST, typename DERIVED >
				static inline void compute_gradient( const Eigen::MatrixBase<DERIVED>& input, RBMCOST& cost){
					auto& rbm = cost.rbm;
					cost.bias_visible = -input;
					auto& weights = rbm.weights;
					cost.weights = HiddenIntegral< HID_TYPE >::compute_hidden_derivative( weights*input, cost )*input.transpose();

					return;
				}
			};

			/*
			*	\brief	RBM cost function to be minimized
			*	\author phineasng
			*/
			template< Regularizer R,
					  typename RBMSampler,
					  RBMTrainingMode MODE,
					  int N = 1 >
			class RBMCost: public Optimization::CostFunction< RBMCost<R,RBMSampler,MODE,N> >{
			public:	// useful typedefs 
				typedef typename RBMSampler::SCALAR SCALAR;
				static const RBMUnitType HID_TYPE = RBMSampler::HID_UNIT_TYPE;
				static const RBMUnitType VIS_TYPE = RBMSampler::VIS_UNIT_TYPE;
			public:
				// CONSTRUCTOR
				RBMCost(RBMSampler& refRbm, 
						const Eigen::Ref< MLMatrix<SCALAR> > refInputs,
						const RegularizerOptions<SCALAR>& refOptions,
						MLVector< SCALAR >& ref_allocated_grad ):
					rbm(refRbm),
					inputs(refInputs),
					options(refOptions),
					gradient_tmp(ref_allocated_grad),
					weights( NULL, 0, 0 ),
					bias_visible( NULL, 0 ),
					bias_hidden( NULL, 0 ),
					additional_parameters_visible( NULL, 0 ),
					additional_parameters_hidden( NULL, 0 )
				{
					mapVariables();
				}
				// MODIFIERS
				void remap(){
					mapVariables();
				}
				// EVALUATE COST FUNCTION
				/*
				*	\brief This is actually the free energy and not a cost related to the "gradient"
				*/
				template < 	typename DERIVED >
				typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
					static_assert(std::is_same<SCALAR,typename DERIVED::Scalar>::value, "Scalar types have to be the same!");
					static_assert(DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
					MLEARN_ASSERT( x.size() == gradient_tmp.size(), "Dimension not valid!" );
					SCALAR loss = SCALAR(0);
					rbm.attachParameters(x);

					for ( decltype(inputs.cols()) idx = 0; idx < inputs.cols(); ++idx ){
						loss += FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_energy(inputs.col(idx),*this);
					}

					return loss;
				}
				template< 	typename DERIVED,
					   		typename DERIVED_2 >
				void compute_analytical_gradient(  const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient ) const{
					static_assert( std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types must be the same" );
					static_assert( std::is_same<typename DERIVED::Scalar,SCALAR>::value, "Scalar types must be the same" );
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Expected column vectors!" );
					MLEARN_ASSERT( x.size() == gradient_tmp.size(), "Dimension not valid!" );
					rbm.attachParameters(x);
					gradient = MLVector<SCALAR>::Zero(x.size());

					for ( decltype(inputs.cols()) idx = 0; idx < inputs.cols(); ++idx ){
						rbm.sampleHFromV(inputs.col(idx));
						rbm.sampleVFromH();
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(inputs.col(idx),*this);
						gradient += gradient_tmp;
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(rbm.visible_units,*this);
						gradient -= gradient_tmp;
					}

					return;
				}
				template< 	typename IndexType,
							typename DERIVED,
				   			typename DERIVED_2 >
				void compute_stochastic_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const MLVector< IndexType >& idx ) const{
					static_assert( std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types must be the same" );
					static_assert( std::is_same<typename DERIVED::Scalar,SCALAR>::value, "Scalar types must be the same" );
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Expected column vectors!" );
					MLEARN_ASSERT( x.size() == gradient_tmp.size(), "Dimension not valid!" );
					rbm.attachParameters(x);
					gradient = MLVector<SCALAR>::Zero(x.size());

					for ( decltype(idx.size()) i = 0; i < idx.size(); ++i ){
						rbm.sampleHFromV(inputs.col(idx[i]));
						rbm.sampleVFromH();
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(inputs.col(idx[i]),*this);
						gradient += gradient_tmp;
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(rbm.visible_units,*this);
						gradient -= gradient_tmp;
					}

					return;
				}
			private:
				RBMSampler& rbm;
				const Eigen::Ref< MLMatrix<SCALAR> > inputs;
				const RegularizerOptions<SCALAR>& options;
				/// \details For safety (in referencing the needed objects) and more flexibility (using the eigen types),
				/// we plan to reconstruct this cost function class everytime a training algorithm is called.
				/// This means that, in an ONLINE or CROSSVALIDATION setting, the construction may incur in
				/// a not negligible overhead if this class allocates memory or has to build objects which allocates
				/// memory (e.g. Eigen objects).
				/// Therefore, we require the necessary temporaries to be built outside the class
				/// and a reference to them at construction.
				MLVector< SCALAR >& gradient_tmp;	
				mutable Eigen::Map< MLMatrix< SCALAR > > weights;
				mutable Eigen::Map< MLVector< SCALAR > > bias_visible;
				mutable Eigen::Map< MLVector< SCALAR > > bias_hidden;
				mutable Eigen::Map< MLVector< SCALAR > > additional_parameters_visible;
				mutable Eigen::Map< MLVector< SCALAR > > additional_parameters_hidden;
			private: // friends declarations
				template <RBMUnitType V, RBMUnitType H >
				friend class FreeEnergy;
				template < RBMUnitType H >
				friend class HiddenIntegral;
			private: // helper functions
				void mapVariables(){
					auto N_vis = rbm.visible_units.size();
					auto N_hid = rbm.hidden_units.size();
					auto N_weights = rbm.weights.size();
					auto N_bias_hid = rbm.bias_hidden.size();
					auto N_bias_vis = rbm.bias_visible.size();
					auto N_add_hid = rbm.additional_parameters_hidden.size();
					auto N_add_vis = rbm.additional_parameters_visible.size();
					MLEARN_ASSERT(N_vis==inputs.rows(),"Visible units must be the same size as the training samples!");
					gradient_tmp.resize( N_weights + N_bias_hid + N_bias_vis + N_add_vis + N_add_hid );
					new (&weights) Eigen::Map< MLMatrix<SCALAR> >( gradient_tmp.data(), N_hid, N_vis );
					auto offset = N_weights;
					new (&bias_hidden) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_hid );
					offset += N_hid;
					new (&bias_visible) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_vis );
					offset += N_vis;
					new (&additional_parameters_hidden) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_add_hid );
					offset += N_add_hid;
					new (&additional_parameters_visible) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_add_vis );
				}
			};


		}
	}
}

#endif