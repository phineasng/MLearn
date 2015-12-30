#ifndef MLEARN_LINE_SEARCH_INCLUDED
#define MLEARN_LINE_SEARCH_INCLUDED

// MLearn Core
#include <MLearn/Core>	

// Utilities
#include <MLearn/Utility/VerbosityLogger.h>

// STL
#include <limits>

// Optimization includes
#include "Differentiation/Differentiator.h"

namespace MLearn{

	namespace Optimization{

		enum class LineSearchStrategy{
			FIXED,
			BACKTRACKING,			// Reference: "https://www.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf"
			BISECTION_WEAK_WOLFE 	// Reference: "https://www.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf"
		};

		template< 	LineSearchStrategy STRATEGY,
					typename ScalarType = double, 
					typename IndexType = uint, 
					typename = typename std::enable_if< std::is_floating_point<ScalarType>::value, void >::type,
					typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, void >::type >
		class LineSearch{};

		template < typename ScalarType, typename IndexType >
		class LineSearch< LineSearchStrategy::FIXED, ScalarType, IndexType >{
		public:
			LineSearch( const ScalarType& step = 0.05 ): step_size(step) {}
			ScalarType step_size;
			// algorithm
			template < 	typename CostFunc, 
						DifferentiationMode MODE, 
						uint VERB_LEVEL = 0u, 
						uint VERB_REF = 0,
						typename DERIVED,
						typename DERIVED_2,
						typename DERIVED_3,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   		typename = typename std::enable_if< DERIVED_3::ColsAtCompileTime == 1 , DERIVED_3 >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED_2::Scalar, typename DERIVED_3::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, ScalarType >::value,typename DERIVED::Scalar >::type > 
			ScalarType getStep( const CostFunc& cost, const Eigen::MatrixBase< DERIVED >& x, const Eigen::MatrixBase< DERIVED_2 >& gradient, const Eigen::MatrixBase< DERIVED_3 >& direction, const GradientOption<MODE,ScalarType,IndexType>& gradient_options ) const{
				return step_size;
			}
		};

		template < typename ScalarType, typename IndexType >
		class LineSearch< LineSearchStrategy::BACKTRACKING, ScalarType, IndexType >{
		public:
			LineSearch( ScalarType refGamma = ScalarType(0.99), ScalarType refC = ScalarType(0.1), IndexType ref_iter = IndexType(100) ): gamma(refGamma), c(refC), max_iter(ref_iter) {}
			ScalarType gamma;
			ScalarType c;
			IndexType max_iter;
			// algorithm
			template < 	typename CostFunc, 
						DifferentiationMode MODE, 
						uint VERB_LEVEL = 0u, 
						uint VERB_REF = 0,
						typename DERIVED,
						typename DERIVED_2,
						typename DERIVED_3,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   		typename = typename std::enable_if< DERIVED_3::ColsAtCompileTime == 1 , DERIVED_3 >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED_2::Scalar, typename DERIVED_3::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, ScalarType >::value,typename DERIVED::Scalar >::type > 
			ScalarType getStep( const CostFunc& cost, const Eigen::MatrixBase< DERIVED >& x_curr, const Eigen::MatrixBase< DERIVED_2 >& gradient_curr, const Eigen::MatrixBase< DERIVED_3 >& direction, const GradientOption<MODE,ScalarType,IndexType>& gradient_options ) const{
				
				Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "< Started backtracking line search >\n" );
				
				ScalarType f_curr = cost.evaluate(x_curr);
				ScalarType delta_f = c*gradient_curr.dot(direction);
				ScalarType new_f = cost.evaluate(x_curr + direction);
				ScalarType t = ScalarType(1);
				IndexType iter = IndexType(0);
				while ( (new_f > f_curr + t*delta_f) && ( iter < max_iter) ){
					
					t *= gamma;
					new_f = cost.evaluate(x_curr + t*direction);

					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( iter );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( ") " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "For step = " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( t );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( ": cost = " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( new_f );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( " vs " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( f_curr + t*delta_f );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "\n" );

					++iter;
				}
				Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "< Backtracking line search terminated >\n" );
				return t;
			}
		};

		template < typename ScalarType, typename IndexType >
		class LineSearch< LineSearchStrategy::BISECTION_WEAK_WOLFE, ScalarType, IndexType >{
		public:
			LineSearch( ScalarType refC1 = ScalarType(0.001), ScalarType refC2 = ScalarType(0.9), IndexType ref_iter = IndexType(100) ): c1(refC1), c2(refC2), max_iter(ref_iter) {}
			ScalarType c1;
			ScalarType c2;
			IndexType max_iter;
			// algorithm
			template < 	typename CostFunc, 
						DifferentiationMode MODE, 
						uint VERB_LEVEL = 0u, 
						uint VERB_REF = 0,
						typename DERIVED,
						typename DERIVED_2,
						typename DERIVED_3,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   		typename = typename std::enable_if< DERIVED_3::ColsAtCompileTime == 1 , DERIVED_3 >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED_2::Scalar, typename DERIVED_3::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, ScalarType >::value,typename DERIVED::Scalar >::type > 
			ScalarType getStep( const CostFunc& cost, const Eigen::MatrixBase< DERIVED >& x_curr, const Eigen::MatrixBase< DERIVED_2 >& gradient_curr, const Eigen::MatrixBase< DERIVED_3 >& direction, const GradientOption<MODE,ScalarType,IndexType>& gradient_options ) const{
				
				Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "< Started bisection line search >\n" );
				
				ScalarType f_curr = cost.evaluate(x_curr);
				ScalarType delta_f = c1*gradient_curr.dot(direction);
				
				ScalarType new_f;
				MLVector<ScalarType> new_gradient(gradient_curr.size());

				ScalarType alpha = ScalarType(0);
				ScalarType beta = std::numeric_limits<ScalarType>::infinity();
				ScalarType t = ScalarType(1);
				
				IndexType iter = IndexType(0);
				MLVector<ScalarType> temporary;

				while ( iter < max_iter ){
					
					temporary = x_curr + t*direction;
					new_f = cost.evaluate( temporary );
					cost.compute_gradient( temporary, new_gradient, gradient_options );

					if ( new_f > (f_curr+c1*t*delta_f) ){
						beta = t;
						t = ScalarType(0.5)*(alpha+beta);
					}else if( new_gradient.dot(direction) < c2*delta_f ){
						alpha = t;
						if ( beta ==  std::numeric_limits<ScalarType>::infinity()){
							t = ScalarType(2)*alpha;
						}else{
							t = ScalarType(0.5)*(alpha+beta);
						}
					}else{
						break;
					}

					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( iter );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( ") " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "For step = " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( t );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( ": cost = " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( new_f );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( " vs " );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( f_curr + t*delta_f );
					Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "\n" );

					++iter;
				}
				Utility::VerbosityLogger<VERB_LEVEL,VERB_REF>::log( "< Bisection line search terminated >\n" );
				return t;
			}
		};

	}

}
	
#endif