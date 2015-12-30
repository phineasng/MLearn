#ifndef MLEARN_STOCHASTIC_GRADIENT_DESCENT_ROUTINE_INCLUDED
#define MLEARN_STOCHASTIC_GRADIENT_DESCENT_ROUTINE_INCLUDED

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
#include <random>

namespace MLearn{

	namespace Optimization{

		template < 	LineSearchStrategy STRATEGY = LineSearchStrategy::FIXED,
					typename ScalarType = double,
					typename UnsignedIntegerType = uint,
					ushort VERBOSITY_REF = 0,
					typename = typename std::enable_if< std::is_floating_point<ScalarType>::value, void >::type,
					typename = typename std::enable_if< std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value, void >::type >
		class StochasticGradientDescent{
		private:
			typedef std::mt19937 RNG_TYPE;
			typedef std::uniform_int_distribution<UnsignedIntegerType> DIST_TYPE;
		public:
			// Constructors
			StochasticGradientDescent(): to_sample(10), gradient_options(to_sample){}
			StochasticGradientDescent( const StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient ):to_sample(10), gradient_options(to_sample), tolerance(refGradient.tolerance), max_iter(refGradient.max_iter), line_search(refGradient.line_search),size_batch(refGradient.size_batch) {}
			StochasticGradientDescent( StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient ):to_sample(10), gradient_options(to_sample), tolerance(std::move(refGradient.tolerance)), max_iter(std::move(refGradient.max_iter)), line_search(std::move(refGradient.line_search)),size_batch(refGradient.size_batch) {}
			// Assignment
			StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient) { tolerance = refGradient.tolerance; max_iter = refGradient.max_iter; line_search = refGradient.line_search; size_batch = refGradient.size_batch; }
			StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient) { tolerance = std::move(refGradient.tolerance); max_iter = std::move(refGradient.max_iter); line_search = std::move(refGradient.line_search);size_batch = refGradient.size_batch; }
			// Modifiers
			void setLineSearchMethod( const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& refLineSearch ){ line_search = refLineSearch; }
			void setLineSearchMethod( LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>&& refLineSearch ){ line_search = std::move(refLineSearch); }
			void setTolerance( ScalarType refTolerance ){ tolerance = refTolerance; }
			void setMaxIter( UnsignedIntegerType refMaxIter ){ max_iter = refMaxIter; }
			void setSeed( RNG_TYPE::result_type s){ rng.seed(s); }
			void setDistributionParameters( UnsignedIntegerType a, UnsignedIntegerType b ){ dist = DIST_TYPE(a,b); }
			void setSizeBatch( UnsignedIntegerType n ){ size_batch = n; }
			// Observers
			const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& getLineSearchMethod() const { return line_search; };
			ScalarType getTolerance() const { return tolerance; }
			UnsignedIntegerType getMaxIter() const { return max_iter; }
			UnsignedIntegerType getMinDist() const { return dist.a(); }
			UnsignedIntegerType getMaxDist() const { return dist.b(); }
			UnsignedIntegerType getSizeBatch() const { return size_batch; }
			// Minimize
			template < 	typename Cost,
						typename DERIVED,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, ScalarType >::value,typename DERIVED::Scalar >::type >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				MLVector< ScalarType > gradient(x.size());
				to_sample.resize( size_batch );

				sample_indeces();
				cost.compute_gradient(x,gradient,gradient_options);
				ScalarType sqTolerance = tolerance*tolerance;
				UnsignedIntegerType iter = UnsignedIntegerType(0);
				
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== STARTING: Gradient Descent Optimization ======\n" );
				
				while( (gradient.squaredNorm() > sqTolerance) && (iter < max_iter) ){
					
					x -= line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,DifferentiationMode::STOCHASTIC,3,VERBOSITY_REF>(cost,x,gradient,-gradient,gradient_options)*gradient;
					sample_indeces();
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
			MLVector< UnsignedIntegerType > to_sample;
			GradientOption<DifferentiationMode::STOCHASTIC,ScalarType,UnsignedIntegerType> gradient_options;
			LineSearch<STRATEGY,ScalarType,UnsignedIntegerType> line_search;
			UnsignedIntegerType max_iter = UnsignedIntegerType(1000);
			UnsignedIntegerType size_batch = UnsignedIntegerType(10);
			ScalarType tolerance = ScalarType(1e-5);
			RNG_TYPE rng;
			DIST_TYPE dist;
			void sample_indeces() {
				for (UnsignedIntegerType i = 0; i < size_batch; ++i){
					to_sample[i] = dist(rng);
				}
			}
		};

	}

}

#endif