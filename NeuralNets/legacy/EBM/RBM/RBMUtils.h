#ifndef MLEARN_RBM_UTILS_INCLUDE
#define MLEARN_RBM_UTILS_INCLUDE

// MLearn includes
#include "RBMUnitType.h"

// STL includes
#include <cmath>

namespace MLearn{

	namespace NeuralNets{

		namespace RBMSupport{

			typedef uint RBMIntegerType;

			/*
			*	\brief 	This operations are mainly used to make it easier
			*			to define expression types without digging into Eigen's internal:: code
			*/

			template < typename SCALAR >
			struct SUM{
				SCALAR operator()(const SCALAR& s1, const SCALAR& s2) const{
					return s1+s2;
				}
			};

			template < typename SCALAR >
			struct NEGATIVE_SUM{
				SCALAR operator()(const SCALAR& s1, const SCALAR& s2) const{
					return -(s1+s2);
				}
			};

			template < typename SCALAR >
			struct DIVISION{
				SCALAR operator()(const SCALAR& s1, const SCALAR& s2) const{
					return s1/s2;
				}
			};

			template < typename SCALAR >
			struct MULTIPLICATION{
				SCALAR operator()(const SCALAR& s1, const SCALAR& s2) const{
					return s1*s2;
				}
			};

			template < typename SCALAR >
			struct EXPONENTIAL{
				SCALAR operator()(const SCALAR& s1) const{
					return std::exp(s1);
				}
			};

			template < typename SCALAR >
			struct INV_EXPONENTIAL{
				SCALAR operator()(const SCALAR& s1) const{
					return std::exp(-s1);
				}
			};


			/*
			*	\brief Typedef for distributions
			*/
			template< RBMUnitType TYPE, typename ScalarType >
			struct Distribution{
				typedef std::normal_distribution<ScalarType> type;
			};

			// BERNOULLI
			template<typename ScalarType>
			struct Distribution<RBMUnitType::BERNOULLI,ScalarType>{
				typedef std::bernoulli_distribution type;
			};

			// BINOMIAL
			template<typename ScalarType>
			struct Distribution<RBMUnitType::BINOMIAL,ScalarType>{
				typedef std::binomial_distribution<RBMIntegerType> type;
			};

			/*
			*	\brief Routines to set distribution parameters
			*/
			template < RBMUnitType TYPE, typename ScalarType >
			struct DistributionSetter{
				template<typename... ARGS>
				static inline void set(ARGS... args){}
			};

			template< typename ScalarType >
			struct DistributionSetter< RBMUnitType::BINOMIAL, ScalarType >{
				static inline void set( typename Distribution<RBMUnitType::BINOMIAL,ScalarType>::type& dist, typename Distribution<RBMUnitType::BINOMIAL,ScalarType>::type::result_type N ){
					typedef typename Distribution<RBMUnitType::BINOMIAL,ScalarType>::type::param_type param_type;
					dist.param( param_type(N,dist.p()) );
				}
			};

			/*
			*	\brief Training mode
			*/
			enum class RBMTrainingMode{
				CONTRASTIVE_DIVERGENCE,
				PERSISTENT_CONTRASTIVE_DIVERGENCE
			};


		}
	}
}

#endif