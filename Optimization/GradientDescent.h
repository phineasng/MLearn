#ifndef MLEARN_GRADIENT_DESCENT_ROUTINE_INCLUDED
#define MLEARN_GRADIENT_DESCENT_ROUTINE_INCLUDED

// MLearn Core 
#include <MLearn/Core>

// Optimization includes
#include "CostFunction.h"
#include "Differentiation/Differentiator.h"
#include "LineSearch.h"

// MLearn utilities
#include <MLearn/Utility/VerbosityLogger.h>

// STL includes
#include <type_traits>

namespace MLearn{

	namespace Optimization{

		template < 	DifferentiationMode MODE, 
					LineSearchStrategy STRATEGY = LineSearchStrategy::FIXED,
					typename ScalarType = double,
					typename UnsignedIntegerType = uint,
					ushort VERBOSITY_REF = 0 >
		class GradientDescent{
		public:
			static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point!");
			static_assert(std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value,"An unsigned integer type is required!");
			static_assert(MODE != DifferentiationMode::STOCHASTIC,"Method not compatible with stochastic gradient!");
			// Constructors
			GradientDescent() = default;
			GradientDescent( const GradientDescent<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient ): gradient_options(refGradient.gradient_options), tolerance(refGradient.tolerance), max_iter(refGradient.max_iter), line_search(refGradient.line_search) {}
			GradientDescent( GradientDescent<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient ): gradient_options(std::move(refGradient.gradient_options)), tolerance(std::move(refGradient.tolerance)), max_iter(std::move(refGradient.max_iter)), line_search(std::move(refGradient.line_search)) {}
			// Assignment
			GradientDescent<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const GradientDescent<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient) { gradient_options = refGradient.gradient_options; tolerance = refGradient.tolerance; max_iter = refGradient.max_iter; line_search = refGradient.line_search; }
			GradientDescent<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( GradientDescent<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient) { gradient_options = std::move(refGradient.gradient_options); tolerance = std::move(refGradient.tolerance); max_iter = std::move(refGradient.max_iter); line_search = std::move(refGradient.line_search); }
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
						typename DERIVED >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				static_assert(std::is_same<typename DERIVED::Scalar, ScalarType >::value, "The scalar type of the vector has to be the same as the one declared for the minimizer!");
				MLVector< ScalarType > gradient(x.size());
				cost.compute_gradient(x,gradient,gradient_options);
				ScalarType sqTolerance = tolerance*tolerance;
				UnsignedIntegerType iter = UnsignedIntegerType(0);
				
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== STARTING: Gradient Descent Optimization ======\n" );
				
				while( (gradient.squaredNorm() > sqTolerance) && (iter < max_iter) ){
					
					x -= line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,MODE,3,VERBOSITY_REF>(cost,x,gradient,-gradient,gradient_options)*gradient;
					cost.compute_gradient(x,gradient,gradient_options);
					++iter;

					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( iter );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Squared gradient norm = " );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( gradient.squaredNorm() );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );
				}
				
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "INFO: Terminated in " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " out of " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( max_iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " iterations!\n" );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== DONE:	 Gradient Descent Optimization ======\n" );
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