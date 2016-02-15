#ifndef MLEARN_RBM_COST_INCLUDE
#define MLEARN_RBM_COST_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include <MLearn/NeuralNets/Regularizer.h>
#include <MLearn/Optimization/CostFunction.h>
#include "RBMUtils.h"

namespace MLearn{

	namespace NeuralNets{

		namespace RBMSupport{

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
						const Eigen::Ref< const MLMatrix<SCALAR> >& refInputs,
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
				template < 	typename DERIVED >
				typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
					static_assert(std::is_same<SCALAR,typename DERIVED::Scalar>::value, "Scalar types have to be the same!");
					static_assert(DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
					MLEARN_ASSERT( x.size() == gradient_tmp.size(), "Dimension not valid!" );
					SCALAR loss = SCALAR(0);

					return loss;
				}
			private:
				RBMSampler& rbm;
				const Eigen::Ref< const MLMatrix<SCALAR> >& inputs;
				const RegularizerOptions<SCALAR>& options;
				/// \details For safety (in referencing the needed objects) and more flexibility (using the eigen types),
				/// we plan to reconstruct this cost function class everytime a training algorithm is called.
				/// This means that, in an ONLINE or CROSSVALIDATION setting, the construction may incur in
				/// a not negligible overhead if this class allocates memory or has to build objects which allocates
				/// memory (e.g. Eigen objects).
				/// Therefore, we require the necessary temporaries to be built outside the class
				/// and a reference to them at construction.
				MLVector< SCALAR >& gradient_tmp;	
				Eigen::Map< MLMatrix< SCALAR > > weights;
				Eigen::Map< MLVector< SCALAR > > bias_visible;
				Eigen::Map< MLVector< SCALAR > > bias_hidden;
				Eigen::Map< MLVector< SCALAR > > additional_parameters_visible;
				Eigen::Map< MLVector< SCALAR > > additional_parameters_hidden;
			private: 
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