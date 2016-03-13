#ifndef MLEARN_STOCHASTIC_BASE_ROUTINE_INCLUDED
#define MLEARN_STOCHASTIC_BASE_ROUTINE_INCLUDED

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

		template < 	typename DERIVED,
					LineSearchStrategy STRATEGY = LineSearchStrategy::FIXED,
					typename ScalarType = double,
					typename UnsignedIntegerType = uint >
		class StochasticBase{
		protected:
			typedef std::mt19937 RNG_TYPE;
		protected:
			// Constructors
			StochasticBase(): gradient_options(to_sample){}
			StochasticBase( const StochasticBase<DERIVED,STRATEGY,ScalarType,UnsignedIntegerType>& refBase ):n_samples(refBase.n_samples), to_sample(refBase.size_batch), gradient_options(to_sample), tolerance(refBase.tolerance), max_iter(refBase.max_iter), line_search(refBase.line_search),size_batch(refBase.size_batch) {}
			StochasticBase( StochasticBase<DERIVED,STRATEGY,ScalarType,UnsignedIntegerType>&& refBase ):n_samples(refBase.n_samples), to_sample(refBase.size_batch), gradient_options(to_sample), tolerance(std::move(refBase.tolerance)), max_iter(std::move(refBase.max_iter)), line_search(std::move(refBase.line_search)),size_batch(refBase.size_batch) {}
		public:
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
						typename EIGEN_DERIVED >
			void minimize( const Cost& cost, Eigen::MatrixBase<EIGEN_DERIVED>& x ){
				static_cast<DERIVED*>(this)->minimize(cost,x);
			}
		protected:
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