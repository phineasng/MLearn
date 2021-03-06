#ifndef MLEARN_STOCHASTIC_GRADIENT_DESCENT_ROUTINE_INCLUDED
#define MLEARN_STOCHASTIC_GRADIENT_DESCENT_ROUTINE_INCLUDED

// MLearn Core 
#include <MLearn/Core>

// Optimization includes
#include "CostFunction.h"
#include "StochasticBase.h"
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
		class StochasticGradientDescent: public StochasticBase< StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>,STRATEGY,ScalarType,UnsignedIntegerType >{
		private:
			static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point!");
			static_assert(std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value,"An unsigned integer type is required!");
			typedef std::mt19937 RNG_TYPE;
			typedef StochasticBase< StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>,STRATEGY,ScalarType,UnsignedIntegerType > BASE_TYPE;
		public:
			// Constructors
			StochasticGradientDescent(): BASE_TYPE() {}
			StochasticGradientDescent( const StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient ):BASE_TYPE(refGradient) {}
			StochasticGradientDescent( StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient ):BASE_TYPE(std::move(refGradient)) {}
			// Assignment
			StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient) { 
				this->n_samples = refGradient.n_samples; 
				this->tolerance = refGradient.tolerance; 
				this->max_iter = refGradient.max_iter; 
				this->line_search = refGradient.line_search; 
				this->size_batch = refGradient.size_batch; 
				return *this; 
			}
			StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( StochasticGradientDescent<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient) { 
				this->n_samples = refGradient.n_samples; 
				this->tolerance = std::move(refGradient.tolerance); 
				this->max_iter = std::move(refGradient.max_iter); 
				this->line_search = std::move(refGradient.line_search);
				this->size_batch = refGradient.size_batch; 
				return *this; 
			}
			// Minimize
			template < 	typename Cost,
						typename DERIVED >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				static_assert(std::is_same<typename DERIVED::Scalar, ScalarType >::value, "Input vector has to be the same type declared in the minimizer!");
				
				// reallocate gradient if necessary
				this->gradient.resize(x.size());
				

				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== STARTING: Stochastic Gradient Descent Optimization ======\n" );

				ScalarType sqTolerance = (this->tolerance)*(this->tolerance);
				UnsignedIntegerType iter = UnsignedIntegerType(0);
				UnsignedIntegerType epoch = UnsignedIntegerType(0);
				UnsignedIntegerType n_batches = ScalarType(this->n_samples)/ScalarType(this->size_batch);
				UnsignedIntegerType r = (this->n_samples) - n_batches*(this->size_batch);
				UnsignedIntegerType offset;
				bool stopped = false;
				
				// shuffle indeces
				this->shuffle_indeces();

				this->to_sample =this->shuffled_indeces.segment(0,this->size_batch);
				cost.compute_gradient(x,this->gradient,this->gradient_options);
				ScalarType sq_norm = this->gradient.squaredNorm();
						
				stopped = (sq_norm < sqTolerance);

				do{
					++epoch;
					stopped = ( iter > (this->max_iter) ) || ( sq_norm < sqTolerance ) || (epoch > (this->max_epoch));

					offset = 0;
					// shuffle indeces
					this->shuffle_indeces();

					for ( UnsignedIntegerType i = 0; (i < n_batches) && !stopped; ++i ){


						x -= this->line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,DifferentiationMode::STOCHASTIC,3,VERBOSITY_REF>(cost,x,this->gradient,-this->gradient,this->gradient_options)*(this->gradient);
						
						this->to_sample = this->shuffled_indeces.segment(offset,this->size_batch);
						cost.compute_gradient(x,this->gradient,this->gradient_options);
						
						sq_norm = this->gradient.squaredNorm();
						
						++iter;
						offset += this->size_batch;

						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( iter );
						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Squared gradient norm = " );
						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( sq_norm );
						Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );

						stopped = ( iter > this->max_iter ) || ( sq_norm < sqTolerance );

					}

					if ( stopped ){
						return;
					}
					
					if ( offset >= this->n_samples ) continue;
					
					x -= this->line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,DifferentiationMode::STOCHASTIC,3,VERBOSITY_REF>(cost,x,this->gradient,-this->gradient,this->gradient_options)*(this->gradient);

					this->to_sample = this->shuffled_indeces.segment(offset,r);
					cost.compute_gradient(x,this->gradient,this->gradient_options);
					
					sq_norm = this->gradient.squaredNorm();
					
					++iter;

					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( iter );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Squared gradient norm = " );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( sq_norm );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );

					stopped = ( iter > this->max_iter ) || (sq_norm < sqTolerance);


				}while( !stopped );
				
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "INFO: Terminated in " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " out of " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( this->max_iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " iterations!\n" );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== DONE:	 Stochastic Gradient Descent Optimization ======\n" );
			}
		};

	}

}

#endif