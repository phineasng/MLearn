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
#include <algorithm>

namespace MLearn{

	namespace Optimization{

		template < 	LineSearchStrategy STRATEGY = LineSearchStrategy::FIXED,
					typename ScalarType = double,
					typename UnsignedIntegerType = uint,
					ushort VERBOSITY_REF = 0 >
		class StochasticGradientDescent{
		private:
			static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point!");
			static_assert(std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value,"An unsigned integer type is required!");
			typedef std::mt19937 RNG_TYPE;
		public:
			// Constructors
			StochasticGradientDescent(): gradient_options(to_sample){}
			StochasticGradientDescent( const StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient ):n_samples(refGradient.n_samples), to_sample(refGradient.size_batch), gradient_options(to_sample), tolerance(refGradient.tolerance), max_iter(refGradient.max_iter), line_search(refGradient.line_search),size_batch(refGradient.size_batch) {}
			StochasticGradientDescent( StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient ):n_samples(refGradient.n_samples), to_sample(refGradient.size_batch), gradient_options(to_sample), tolerance(std::move(refGradient.tolerance)), max_iter(std::move(refGradient.max_iter)), line_search(std::move(refGradient.line_search)),size_batch(refGradient.size_batch) {}
			// Assignment
			StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient) { n_samples = refGradient.n_samples; tolerance = refGradient.tolerance; max_iter = refGradient.max_iter; line_search = refGradient.line_search; size_batch = refGradient.size_batch; }
			StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient) { n_samples = refGradient.n_samples; tolerance = std::move(refGradient.tolerance); max_iter = std::move(refGradient.max_iter); line_search = std::move(refGradient.line_search);size_batch = refGradient.size_batch; }
			// Modifiers
			void setLineSearchMethod( const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& refLineSearch ){ line_search = refLineSearch; }
			void setLineSearchMethod( LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>&& refLineSearch ){ line_search = std::move(refLineSearch); }
			void setTolerance( ScalarType refTolerance ){ tolerance = refTolerance; }
			void setMaxIter( UnsignedIntegerType refMaxIter ){ max_iter = refMaxIter; }
			void setMaxEpoch( UnsignedIntegerType refMaxEpoch ){ max_epoch = refMaxEpoch; }
			void setSeed( RNG_TYPE::result_type s){ rng.seed(s); }
			void setNSamples( UnsignedIntegerType N ){ n_samples = N; }
			void setSizeBatch( UnsignedIntegerType n ){ size_batch = n; }
			// Observers
			const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& getLineSearchMethod() const { return line_search; };
			ScalarType getTolerance() const { return tolerance; }
			UnsignedIntegerType getMaxIter() const { return max_iter; }
			UnsignedIntegerType getNSamples() const { return n_samples; }
			UnsignedIntegerType getSizeBatch() const { return size_batch; }
			UnsignedIntegerType getMaxEpoch() const { return max_epoch; }
			// Minimize
			template < 	typename Cost,
						typename DERIVED >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				static_assert(std::is_same<typename DERIVED::Scalar, ScalarType >::value, "Input vector has to be the same type declared in the minimizer!");
				
				// reallocate gradient if necessary
				gradient.resize(x.size());
				

				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== STARTING: Stochastic Gradient Descent Optimization ======\n" );

				ScalarType sqTolerance = tolerance*tolerance;
				UnsignedIntegerType iter = UnsignedIntegerType(0);
				UnsignedIntegerType epoch = UnsignedIntegerType(0);
				UnsignedIntegerType n_batches = ScalarType(n_samples)/ScalarType(size_batch);
				UnsignedIntegerType r = n_samples - n_batches*size_batch;
				UnsignedIntegerType offset;
				bool stopped = false;
				
				// shuffle indeces
				shuffle_indeces();

				to_sample = shuffled_indeces.segment(0,size_batch);
				cost.compute_gradient(x,gradient,gradient_options);
				ScalarType sq_norm = gradient.squaredNorm();
						
				stopped = (sq_norm < sqTolerance);

				do{
					++epoch;
					stopped = ( iter > max_iter ) || ( sq_norm < sqTolerance ) || (epoch > max_epoch);

					offset = 0;
					// shuffle indeces
					shuffle_indeces();

					for ( UnsignedIntegerType i = 0; (i < n_batches) && !stopped; ++i ){


						x -= line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,DifferentiationMode::STOCHASTIC,3,VERBOSITY_REF>(cost,x,gradient,-gradient,gradient_options)*gradient;
						
						to_sample = shuffled_indeces.segment(offset,size_batch);
						cost.compute_gradient(x,gradient,gradient_options);
						
						sq_norm = gradient.squaredNorm();
						
						++iter;
						offset += size_batch;

						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( iter );
						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Squared gradient norm = " );
						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( sq_norm );
						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );

						stopped = ( iter > max_iter ) || ( sq_norm < sqTolerance );

					}

					if ( stopped ){
						return;
					}
					
					if ( offset >= n_samples ) continue;
					
					x -= line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,DifferentiationMode::STOCHASTIC,3,VERBOSITY_REF>(cost,x,gradient,-gradient,gradient_options)*gradient;

					to_sample = shuffled_indeces.segment(offset,r);
					cost.compute_gradient(x,gradient,gradient_options);
					
					sq_norm = gradient.squaredNorm();
					
					++iter;

					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( iter );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Squared gradient norm = " );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( sq_norm );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );

					stopped = ( iter > max_iter ) || (sq_norm < sqTolerance);


				}while( !stopped );
				
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "INFO: Terminated in " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " out of " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( max_iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " iterations!\n" );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== DONE:	 Stochastic Gradient Descent Optimization ======\n" );
			}
		private:
			MLVector< UnsignedIntegerType > to_sample;
			MLVector< UnsignedIntegerType > shuffled_indeces;
			GradientOption<DifferentiationMode::STOCHASTIC,ScalarType,UnsignedIntegerType> gradient_options;
			LineSearch<STRATEGY,ScalarType,UnsignedIntegerType> line_search;
			UnsignedIntegerType max_iter = UnsignedIntegerType(1000);
			UnsignedIntegerType max_epoch = UnsignedIntegerType(100);
			UnsignedIntegerType n_samples = UnsignedIntegerType(0);
			UnsignedIntegerType size_batch = UnsignedIntegerType(10);
			MLVector< ScalarType > gradient;
			ScalarType tolerance = ScalarType(1e-5);
			RNG_TYPE rng;
			void shuffle_indeces() {
				shuffled_indeces.resize(n_samples);
				// fill the indeces
				for ( decltype(n_samples) i = 0; i < n_samples; ++i ){
					shuffled_indeces[i] = i;
				}
				auto generator = [&](UnsignedIntegerType i){
					return rng() % n_samples;
				};
				// shuffle
				std::random_shuffle(shuffled_indeces.data(),shuffled_indeces.data() + n_samples,generator);
			}
		};

	}

}

#endif