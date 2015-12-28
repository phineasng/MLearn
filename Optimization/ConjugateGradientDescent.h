#ifndef MLEARN_CONJUGATE_GRADIENT_DESCENT_ROUTINE_INCLUDED
#define MLEARN_CONJUGATE_GRADIENT_DESCENT_ROUTINE_INCLUDED

// MLearn Core 
#include <MLearn/Core>

// Optimization includes
#include "CostFunction.h"
#include "Differentiation/Differentiator.h"
#include "LineSearch.h"

// MLearn utilities
#include <MLearn/Utility/VerbosityLogger.h>

// STL includes
#include <utility>
#include <algorithm>
#include <type_traits>

namespace MLearn{

	namespace Optimization{

		enum class ConjugateFormula{
			FLETCHER_REEVES,
			POLAK_RIBIERE,
			HESTENES_STIEFEL,
			PR_WITH_RESET
		};

		template < ConjugateFormula FORMULA = ConjugateFormula::PR_WITH_RESET, typename ScalarType = double >
		struct BetaComputer{
			static ScalarType getBeta( const MLVector<ScalarType>& delta_x_n, const MLVector<ScalarType>& delta_x_n_1, const MLVector<ScalarType>& s_n_1){}
		};
		
		template < typename ScalarType >
		struct BetaComputer<ConjugateFormula::FLETCHER_REEVES,ScalarType>{
			static ScalarType getBeta( const MLVector<ScalarType>& delta_x_n, const MLVector<ScalarType>& delta_x_n_1, const MLVector<ScalarType>& s_n_1){
				return (delta_x_n.dot(delta_x_n)/(delta_x_n_1.dot(delta_x_n_1)));
			}
		};

		template < typename ScalarType >
		struct BetaComputer<ConjugateFormula::POLAK_RIBIERE,ScalarType>{
			static ScalarType getBeta( const MLVector<ScalarType>& delta_x_n, const MLVector<ScalarType>& delta_x_n_1, const MLVector<ScalarType>& s_n_1){
				return (delta_x_n.dot(delta_x_n-delta_x_n_1)/(delta_x_n_1.dot(delta_x_n_1)));
			}
		};

		template < typename ScalarType >
		struct BetaComputer<ConjugateFormula::HESTENES_STIEFEL,ScalarType>{
			static ScalarType getBeta( const MLVector<ScalarType>& delta_x_n, const MLVector<ScalarType>& delta_x_n_1, const MLVector<ScalarType>& s_n_1){
				return -(delta_x_n.dot(delta_x_n-delta_x_n_1)/(s_n_1.dot(delta_x_n-delta_x_n_1)));
			}
		};

		template < typename ScalarType >
		struct BetaComputer<ConjugateFormula::PR_WITH_RESET,ScalarType>{
			static ScalarType getBeta( const MLVector<ScalarType>& delta_x_n, const MLVector<ScalarType>& delta_x_n_1, const MLVector<ScalarType>& s_n_1){
				return std::max<ScalarType>((delta_x_n.dot(delta_x_n-delta_x_n_1)/(delta_x_n_1.dot(delta_x_n_1))),ScalarType(0));
			}
		};

		template < 	DifferentiationMode MODE, 
					LineSearchStrategy STRATEGY = LineSearchStrategy::FIXED,
					ConjugateFormula FORMULA = ConjugateFormula::PR_WITH_RESET,
					typename ScalarType = double,
					typename UnsignedIntegerType = uint,
					ushort VERBOSITY_REF = 0,
					typename = typename std::enable_if< std::is_floating_point<ScalarType>::value, void >::type,
					typename = typename std::enable_if< std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value, void >::type,
					typename = typename std::enable_if< MODE != DifferentiationMode::STOCHASTIC, void >::type >
		class ConjugateGradientDescent{
		public:
			// Constructors
			ConjugateGradientDescent() = default;
			ConjugateGradientDescent( const ConjugateGradientDescent<MODE,STRATEGY,FORMULA,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient ): gradient_options(refGradient.gradient_options), tolerance(refGradient.tolerance), max_iter(refGradient.max_iter), line_search(refGradient.line_search) {}
			ConjugateGradientDescent( ConjugateGradientDescent<MODE,STRATEGY,FORMULA,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient ): gradient_options(std::move(refGradient.gradient_options)), tolerance(std::move(refGradient.tolerance)), max_iter(std::move(refGradient.max_iter)), line_search(std::move(refGradient.line_search)) {}
			// Assignment
			ConjugateGradientDescent<MODE,STRATEGY,FORMULA,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const ConjugateGradientDescent<MODE,STRATEGY,FORMULA,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient) { gradient_options = refGradient.gradient_options; tolerance = refGradient.tolerance; max_iter = refGradient.max_iter; line_search = refGradient.line_search; }
			ConjugateGradientDescent<MODE,STRATEGY,FORMULA,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( ConjugateGradientDescent<MODE,STRATEGY,FORMULA,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient) { gradient_options = std::move(refGradient.gradient_options); tolerance = std::move(refGradient.tolerance); max_iter = std::move(refGradient.max_iter); line_search = std::move(refGradient.line_search); }
			// Modifiers
			void setGradientOptions( const GradientOption<MODE,ScalarType>& options ){ gradient_options = options; }
			void setGradientOptions( GradientOption<MODE,ScalarType>&& options ){ gradient_options = std::move(options); }
			void setLineSearchMethod( const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& refLineSearch ){ line_search = refLineSearch; }
			void setLineSearchMethod( LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>&& refLineSearch ){ line_search = std::move(refLineSearch); }
			void setTolerance( ScalarType refTolerance ){ tolerance = refTolerance; }
			void setMaxIter( UnsignedIntegerType refMaxIter ){ max_iter = refMaxIter; }
			// Observers
			const GradientOption<MODE,ScalarType>& getGradientOptions() const { return gradient_options; }
			const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& getLineSearchMethod() const { return line_search; };
			ScalarType getTolerance() const { return tolerance; }
			UnsignedIntegerType getMaxIter() const { return max_iter; }
			// Minimize
			template < 	typename Cost,
						typename DERIVED,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, ScalarType >::value,typename DERIVED::Scalar >::type >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				MLVector< ScalarType > delta_x_n(x.size());
				MLVector< ScalarType > delta_x_n_1(x.size());
				MLVector< ScalarType > s_n_1(x.size());
				ScalarType sqTolerance = tolerance*tolerance;
				UnsignedIntegerType iter = UnsignedIntegerType(0);
				cost.compute_gradient(x,delta_x_n,gradient_options);
				s_n_1 = -delta_x_n;
				ScalarType step = line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,MODE,3,VERBOSITY_REF>(cost,x,delta_x_n,s_n_1,gradient_options);
				delta_x_n = s_n_1;
				ScalarType beta;

				x -= step*delta_x_n; 
				
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== STARTING: Conjugate Gradient Descent Optimization ======\n" );
				
				do{
					// 1. 
					std::swap(delta_x_n_1,delta_x_n);
					cost.compute_gradient(x,delta_x_n,gradient_options);
					delta_x_n *= ScalarType(-1);
					// 2.
					beta = BetaComputer<FORMULA,ScalarType>::getBeta(delta_x_n,delta_x_n_1,s_n_1);
					// 3.
					s_n_1 = delta_x_n + beta*s_n_1;
					// 4. 
					step = line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,MODE,3,VERBOSITY_REF>(cost,x,-delta_x_n,s_n_1,gradient_options);
					x += step*s_n_1;
					
					++iter;

					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( iter );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Squared gradient norm = " );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( delta_x_n.squaredNorm() );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );
				}while( (delta_x_n_1.squaredNorm() > sqTolerance) && (iter < max_iter) );
				
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "INFO: Terminated in " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " out of " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( max_iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " iterations!\n" );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== DONE:	 Conjugate Gradient Descent Optimization ======\n" );
			}
		private:
			GradientOption<MODE,ScalarType> gradient_options;
			LineSearch<STRATEGY,ScalarType,UnsignedIntegerType> line_search;
			UnsignedIntegerType max_iter = UnsignedIntegerType(1000);
			ScalarType tolerance = ScalarType(1e-5);
		};

	}

}

#endif