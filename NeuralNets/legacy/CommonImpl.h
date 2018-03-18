#ifndef MLEARN_COMMON_NEURAL_NETS_FUNCS_H_FILE
#define MLEARN_COMMON_NEURAL_NETS_FUNCS_H_FILE

// MLearn includes
#include <MLearn/Core>
#include "ActivationFunction.h"

namespace MLearn{

	namespace NeuralNets{

		namespace InternalImpl{

			/*!
			*	\brief 		Helper class for efficient computation of derivatives
			*	\details	If the logistic activation is used, the specialization is called, allowing for efficient computations
			*/
			template < ActivationType TYPE >
			struct DerivativeWrapper{
				template < typename WeightType >
				static inline void derive( Eigen::Ref< const MLVector< WeightType > > pre_activations, Eigen::Ref< const MLVector< WeightType > > activations, Eigen::Ref< MLVector< WeightType > > derivatives ){
					derivatives = pre_activations.unaryExpr(std::pointer_to_unary_function<WeightType,WeightType>(ActivationFunction<TYPE>::first_derivative));
				}	
			};

			template <>
			struct DerivativeWrapper< ActivationType::LOGISTIC >{
				template < typename WeightType >
				static inline void derive( Eigen::Ref< const MLVector< WeightType > > pre_activations, Eigen::Ref< const MLVector< WeightType > > activations, Eigen::Ref< MLVector< WeightType > > derivatives ){
					derivatives = activations - activations.cwiseProduct(activations);
				}	
			};

		}

	}

}

#endif