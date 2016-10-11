#ifndef MLEARN_ADA_DELTA_ROUTINE_INCLUDED
#define MLEARN_ADA_DELTA_ROUTINE_INCLUDED

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

		template < 	typename ScalarType = double,
					typename UnsignedIntegerType = uint,
					ushort VERBOSITY_REF = 0 >
		class AdaDelta: public StochasticBase< AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>,LineSearchStrategy::FIXED,ScalarType,UnsignedIntegerType >{
		private:
			static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point!");
			static_assert(std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value,"An unsigned integer type is required!");
			typedef std::mt19937 RNG_TYPE;
			typedef StochasticBase< AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>,LineSearchStrategy::FIXED,ScalarType,UnsignedIntegerType > BASE_TYPE;
		public:
			// Constructors
			AdaDelta(): BASE_TYPE() {}
			AdaDelta( const AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient ):BASE_TYPE(refGradient), forget_factor(refGradient.forget_factor), epsilon(refGradient.epsilon) {}
			AdaDelta( AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient ):BASE_TYPE(std::move(refGradient)), forget_factor(refGradient.forget_factor), epsilon(refGradient.epsilon) {}
			// Assignment
			AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refGradient) { 
				this->n_samples = refGradient.n_samples; 
				this->tolerance = refGradient.tolerance; 
				this->max_iter = refGradient.max_iter; 
				this->line_search = refGradient.line_search; 
				this->size_batch = refGradient.size_batch; 
				forget_factor = refGradient.forget_factor;
				epsilon = refGradient.epsilon;
				initializedFlag = false;
				return *this; 
			}
			AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( AdaDelta<ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refGradient) { 
				this->n_samples = refGradient.n_samples; 
				this->tolerance = std::move(refGradient.tolerance); 
				this->max_iter = std::move(refGradient.max_iter); 
				this->line_search = std::move(refGradient.line_search);
				this->size_batch = refGradient.size_batch;  
				forget_factor = refGradient.forget_factor;
				epsilon = refGradient.epsilon;
				initializedFlag = false;
				return *this; 
			}
			// Observers
			ScalarType getForgetFactor() const{
				return forget_factor;
			}
			ScalarType getEpsilon() const{
				return epsilon;
			}
			bool isInitialized() const{
				return initializedFlag;
			}
			// Modifiers
			void setForgetFactor(ScalarType refFactor){
				forget_factor = refFactor;
			}
			void setEpsilon(ScalarType refEpsilon){
				epsilon = refEpsilon;
			}
			void setInitializedFlag(bool flag){
				initializedFlag = flag;
			}
			// Minimize
			template < 	typename Cost,
						typename DERIVED >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				static_assert(std::is_same<typename DERIVED::Scalar, ScalarType >::value, "Input vector has to be the same type declared in the minimizer!");
				
				bool gradient_is_zero = false;
				// reallocate gradient if necessary
				this->gradient.resize(x.size());
				update.resize(x.size());
				if (!initializedFlag){
					initializedFlag = true;
					gradient_mean = MLVector<ScalarType>::Zero(x.size());
					gradient_is_zero = true;
					update_mean = MLVector<ScalarType>::Zero(x.size());
				}else{
					MLEARN_ASSERT( gradient_mean.size() == x.size(), "Size of input vector and update vectors not compatible! Set initialization flag to false!" );
					MLEARN_ASSERT( update_mean.size() == x.size(), "Size of input vector and update vectors not compatible! Set initialization flag to false!" );
				}

				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== STARTING: Stochastic Gradient Descent Optimization ======\n" );

				ScalarType sqTolerance = (this->tolerance)*(this->tolerance);
				ScalarType _1m_forget_factor = ScalarType(1) - forget_factor;
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
				if (gradient_is_zero)
					gradient_mean = (this->gradient).cwiseAbs2();
				else
					gradient_mean = forget_factor*gradient_mean + _1m_forget_factor*(this->gradient).cwiseAbs2();

				ScalarType sq_norm = this->gradient.squaredNorm();
						
				stopped = (sq_norm < sqTolerance);

				do{
					++epoch;
					stopped = ( iter > (this->max_iter) ) || ( sq_norm < sqTolerance ) || (epoch > (this->max_epoch));

					offset = 0;
					// shuffle indeces
					this->shuffle_indeces();

					for ( UnsignedIntegerType i = 0; (i < n_batches) && !stopped; ++i ){

						update.array() = -(this->gradient.array())*( ( update_mean.array() + epsilon ).sqrt() )/( ( gradient_mean.array() + epsilon ).sqrt() ) ;
						update_mean = forget_factor*update_mean + _1m_forget_factor*update.cwiseAbs2();
						x += update;
						
						this->to_sample = this->shuffled_indeces.segment(offset,this->size_batch);
						cost.compute_gradient(x,this->gradient,this->gradient_options);
						gradient_mean = forget_factor*gradient_mean + _1m_forget_factor*(this->gradient).cwiseAbs2();
				
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
					
					update.array() = -(this->gradient.array())*( ( update_mean.array() + epsilon ).sqrt() )/( ( gradient_mean.array() + epsilon ).sqrt() ) ;
					update_mean = forget_factor*update_mean + _1m_forget_factor*update.cwiseAbs2();
					x += update;
						
					this->to_sample = this->shuffled_indeces.segment(offset,r);
					cost.compute_gradient(x,this->gradient,this->gradient_options);
					gradient_mean = forget_factor*gradient_mean + _1m_forget_factor*(this->gradient).cwiseAbs2();
				
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
			MLVector< ScalarType > gradient_mean;
			MLVector< ScalarType > update_mean;
			MLVector< ScalarType > update;
			ScalarType forget_factor = ScalarType(0.95);
			ScalarType epsilon = ScalarType(1e-3);
			bool initializedFlag = false;
		};

	}

}

#endif