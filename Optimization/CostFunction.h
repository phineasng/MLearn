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
			template < typename ScalarType >
			ScalarType evaluate( const MLVector< ScalarType >& x ) const{
				return static_cast<const Derived*>(this)->eval(x);
			}
			// gradient function
			template< 	DifferentiationMode MODE, 
						typename ScalarType,
						typename = typename std::enable_if< std::is_floating_point<ScalarType>::value , ScalarType >::type >
			void compute_gradient( const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const GradientOption< MODE, ScalarType >& options = GradientOption< MODE, ScalarType >() ) const{
				Differentiator<MODE>::compute_gradient(*(static_cast<const Derived*>(this)),x,gradient,options);
			}
			// default functions
			// -- eval
			template < typename ScalarType >
			ScalarType eval( const MLVector< ScalarType >& x ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				return ScalarType(0);
			}
			// -- analytical gradient
			template < typename ScalarType >
			void compute_analytical_gradient(  const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				gradient.resize(x.size());
				return;
			}
			// -- stochastic gradient
			template < typename ScalarType, typename IndexType >
			void compute_stochastic_gradient(  const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const MLVector< IndexType >& idx ) const{
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