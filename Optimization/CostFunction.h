#ifndef MLEARN_COST_FUNCTION_INCLUDED
#define MLEARN_COST_FUNCTION_INCLUDED

// MLearn Core include
#include <MLearn/Core>

// Optimization includes
#include "Differentiation/Differentiator.h"

namespace MLearn{

	namespace Optimization{

		/*!
		*
		*	\brief		Cost Function interface class
		*	\details	Implementation of the cost function using CRTP
		*	\author		phineasng
		*
		*/
		template  < typename Derived >
		class CostFunction{
		public:
			// evaluation function
			template < 	typename DERIVED,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type >
			typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& x ) const{
				return static_cast<const Derived*>(this)->eval(x);
			}
			// gradient function
			template< 	DifferentiationMode MODE, 
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
			void compute_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< MODE, typename DERIVED::Scalar >& options = GradientOption< MODE, typename DERIVED::Scalar >() ) const{
				Differentiator<MODE>::compute_gradient(*(static_cast<const Derived*>(this)),x,gradient,options);
			}
			// default functions
			// -- eval
			template < 	typename DERIVED,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type >
			typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				return DERIVED::Scalar(0);
			}
			// -- analytical gradient
			template< 	typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
			void compute_analytical_gradient(  const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				gradient.resize(x.size());
				return;
			}
			// -- stochastic gradient
			template< 	typename IndexType,
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, void >::type >
			void compute_stochastic_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const MLVector< IndexType >& idx ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				gradient.resize(x.size());
				return;
			}

		protected:
			// Protect Constructors
			CostFunction() = default;
			CostFunction( const CostFunction<Derived>& ) = default;
			CostFunction( CostFunction<Derived>&& ) = default;
		private:
		};

	}

} /*End MLearn namespace*/

#endif