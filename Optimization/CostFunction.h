#ifndef MLEARN_COST_FUNCTION_INCLUDED
#define MLEARN_COST_FUNCTION_INCLUDED

// MLearn Core include
#include <MLearn/Core>

// Optimization includes
#include "Differentiation/Differentiator.h"

// USEFUL MACRO FOR DEFINING NECESSARY MEMBER FUNCTIONS
#define TEMPLATED_SIGNATURE_EVAL_FUNCTION(x_variable)\
	template < 	typename MATRIX_DERIVED_TYPE >\
	typename MATRIX_DERIVED_TYPE::Scalar eval( const Eigen::MatrixBase<MATRIX_DERIVED_TYPE>& x_variable ) const


#define TEMPLATED_SIGNATURE_ANALYTICAL_GRADIENT_FUNCTION(x_variable,gradient_to_compute)\
	template< 	typename MATRIX_DERIVED_TYPE,\
		   		typename MATRIX_DERIVED_TYPE_2 >\
	void compute_analytical_gradient(  const Eigen::MatrixBase<MATRIX_DERIVED_TYPE>& x_variable, Eigen::MatrixBase<MATRIX_DERIVED_TYPE_2>& gradient_to_compute ) const


#define TEMPLATED_SIGNATURE_STOCHASTIC_GRADIENT_FUNCTION(x_variable,gradient_to_compute,idx_to_sample)\
	template< 	typename INDEX_TYPE_IN_MACRO_111,\
				typename MATRIX_DERIVED_TYPE,\
				typename MATRIX_DERIVED_TYPE_2 >\
	void compute_stochastic_gradient( const Eigen::MatrixBase<MATRIX_DERIVED_TYPE>& x_variable, Eigen::MatrixBase<MATRIX_DERIVED_TYPE_2>& gradient_to_compute, const MLVector< INDEX_TYPE_IN_MACRO_111 >& idx_to_sample ) const


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
				static_assert(DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
				return static_cast<const Derived*>(this)->eval(x);
			}
			// gradient function
			template< 	DifferentiationMode MODE, 
						typename DERIVED,
				   		typename DERIVED_2 >
			void compute_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< MODE, typename DERIVED::Scalar >& options = GradientOption< MODE, typename DERIVED::Scalar >() ) const{
				static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!");
				static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!");
				Differentiator<MODE>::compute_gradient(*(static_cast<const Derived*>(this)),x,gradient,options);
			}
			// default functions
			// -- eval
			template < 	typename DERIVED >
			typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				return typename DERIVED::Scalar(0);
			}
			// -- analytical gradient
			template< 	typename DERIVED,
				   		typename DERIVED_2 >
			void compute_analytical_gradient(  const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				static_cast<DERIVED_2*>(&gradient)->resize(x.size());
				return;
			}
			// -- stochastic gradient
			template< 	typename IndexType,
						typename DERIVED,
				   		typename DERIVED_2 >
			void compute_stochastic_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const MLVector< IndexType >& idx ) const{
				MLEARN_FORCED_WARNING_MESSAGE("USING DEFAULT IMPLEMENTATION!");
				static_cast<DERIVED_2*>(&gradient)->resize(x.size());
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