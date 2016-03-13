#ifndef MLEARN_ADAGRAD_ROUTINE_INCLUDED
#define MLEARN_ADAGRAD_ROUTINE_INCLUDED

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
		class AdaGrad: public StochasticBase< AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>,STRATEGY,ScalarType,UnsignedIntegerType >{
		private:
			static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point!");
			static_assert(std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value,"An unsigned integer type is required!");
			typedef std::mt19937 RNG_TYPE;
			typedef StochasticBase< AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>,STRATEGY,ScalarType,UnsignedIntegerType > BASE_TYPE;
		public:
			// Constructors
			AdaGrad(): BASE_TYPE() {}
			AdaGrad( const AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient ):BASE_TYPE(refGradient), delta(refGradient.delta) {}
			AdaGrad( AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient ):BASE_TYPE(std::move(refGradient)), delta(refGradient.delta) {}
			// Assignment
			AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient) { 
				this->n_samples = refGradient.n_samples; 
				this->tolerance = refGradient.tolerance; 
				this->max_iter = refGradient.max_iter; 
				this->line_search = refGradient.line_search; 
				this->size_batch = refGradient.size_batch; 
				delta = refGradient.delta;
				initializedFlag = false;
				return *this; 
			}
			AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( AdaGrad<STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient) { 
				this->n_samples = refGradient.n_samples; 
				this->tolerance = std::move(refGradient.tolerance); 
				this->max_iter = std::move(refGradient.max_iter); 
				this->line_search = std::move(refGradient.line_search);
				this->size_batch = refGradient.size_batch;  
				delta = refGradient.delta;
				initializedFlag = false;
				return *this; 
			}
			// Observers
			ScalarType getDelta() const{
				return delta;
			}
			bool isInitialized() const{
				return initializedFlag;
			}
			// Modifiers
			void setDelta(ScalarType refDelta) {
				delta = refDelta;
			}
			void setInitializedFlag(bool flag){
				initializedFlag = flag;
			}
			// Minimize
			template < 	typename Cost,
						typename DERIVED >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				static_assert(std::is_same<typename DERIVED::Scalar, ScalarType >::value, "Input vector has to be the same type declared in the minimizer!");
				
				// reallocate gradient if necessary
				this->gradient.resize(x.size());
				if (!initializedFlag){
					gradient_cumul = MLVector<ScalarType>::Zero(x.size());
					initializedFlag = true;
				}else{
					MLEARN_ASSERT( gradient_cumul.size() == x.size(), "Size of input vector and update vectors not compatible! Set initialization flag to false!" );
				}
				

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
				gradient_cumul.array() += (this->gradient.array())*(this->gradient.array());
				ScalarType sq_norm = this->gradient.squaredNorm();
						
				stopped = (sq_norm < sqTolerance);

				do{
					++epoch;
					stopped = ( iter > (this->max_iter) ) || ( sq_norm < sqTolerance ) || (epoch > (this->max_epoch));

					offset = 0;
					// shuffle indeces
					this->shuffle_indeces();

					for ( UnsignedIntegerType i = 0; (i < n_batches) && !stopped; ++i ){

						this->gradient.array() /= (gradient_cumul.array().sqrt() + delta); 
						x -= this->line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,DifferentiationMode::STOCHASTIC,3,VERBOSITY_REF>(cost,x,this->gradient,-this->gradient,this->gradient_options)*(this->gradient);
						
						this->to_sample = this->shuffled_indeces.segment(offset,this->size_batch);
						cost.compute_gradient(x,this->gradient,this->gradient_options);
						gradient_cumul.array() += (this->gradient.array())*(this->gradient.array());
				
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
					
					this->gradient.array() /= (gradient_cumul.array().sqrt() + delta); 
					x -= this->line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,DifferentiationMode::STOCHASTIC,3,VERBOSITY_REF>(cost,x,this->gradient,-this->gradient,this->gradient_options)*(this->gradient);

					this->to_sample = this->shuffled_indeces.segment(offset,r);
					cost.compute_gradient(x,this->gradient,this->gradient_options);
					gradient_cumul.array() += (this->gradient.array())*(this->gradient.array());
				
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
		private:
			MLVector< ScalarType > gradient_cumul;
			ScalarType delta = ScalarType(1e-8);
			bool initializedFlag = false;
		};

	}

}

#endif