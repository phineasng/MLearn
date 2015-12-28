#ifndef MLEARN_BFGS_OPTIMIZER_INCLUDED
#define MLEARN_BFGS_OPTIMIZER_INCLUDED

// MLearn Core 
#include <MLearn/Core>

// Optimization includes
#include "CostFunction.h"
#include "Differentiation/Differentiator.h"

// MLearn utilities
#include <MLearn/Utility/VerbosityLogger.h>

// STL includes
#include <type_traits>

// Eigen includes
#include <Eigen/QR>

namespace MLearn{

	namespace Optimization{

		template < 	DifferentiationMode MODE, 
					LineSearchStrategy STRATEGY = LineSearchStrategy::BACKTRACKING,
					typename ScalarType = double,
					typename UnsignedIntegerType = uint,
					ushort VERBOSITY_REF = 0,
					typename = typename std::enable_if< std::is_floating_point<ScalarType>::value, void >::type,
					typename = typename std::enable_if< std::is_integral<UnsignedIntegerType>::value && std::is_unsigned<UnsignedIntegerType>::value, void >::type,
					typename = typename std::enable_if< MODE != DifferentiationMode::STOCHASTIC, void >::type >
		class BFGS{
		public:
			// Constructors
			BFGS() = default;
			BFGS( const BFGS<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refBFGS ): gradient_options(refBFGS.gradient_options), tolerance(refBFGS.tolerance), max_iter(refBFGS.max_iter), line_search(refBFGS.line_search), hessian_approx(refBFGS.hessian_approx), initialize_hessian(refBFGS.initialize_hessian) {}
			BFGS( BFGS<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refBFGS ): gradient_options(std::move(refBFGS.gradient_options)), tolerance(std::move(refBFGS.tolerance)), max_iter(std::move(refBFGS.max_iter)), line_search(std::move(refBFGS.line_search)), hessian_approx(std::move(refBFGS.hessian_approx)), initialize_hessian(refBFGS.initialize_hessian) {}
			// Assignment
			BFGS<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( const BFGS<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& refBFGS) { gradient_options = refBFGS.gradient_options; tolerance = refBFGS.tolerance; max_iter = refBFGS.max_iter; line_search = refBFGS.line_search; hessian_approx = refBFGS.hessian_approx; initialize_hessian = refBFGS.initialize_hessian; }
			BFGS<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>& operator= ( BFGS<MODE,STRATEGY,ScalarType,UnsignedIntegerType,VERBOSITY_REF>&& refBFGS) { gradient_options = std::move(refBFGS.gradient_options); tolerance = std::move(refBFGS.tolerance); max_iter = std::move(refBFGS.max_iter); line_search = std::move(refBFGS.line_search); hessian_approx = std::move(refBFGS.hessian_approx); initialize_hessian = refBFGS.initialize_hessian; }
			// Modifiers
			void setGradientOptions( const GradientOption<MODE,ScalarType>& options ){ gradient_options = options; }
			void setGradientOptions( GradientOption<MODE,ScalarType>&& options ){ gradient_options = std::move(options); }
			void setLineSearchMethod( const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& refLineSearch ){ line_search = refLineSearch; }
			void setLineSearchMethod( LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>&& refLineSearch ){ line_search = std::move(refLineSearch); }
			void setHessian(const MLMatrix<ScalarType>& refHessian){ hessian_approx = refHessian; initialize_hessian = false; }
			void setHessian(MLMatrix<ScalarType>&& refHessian){ hessian_approx = std::move(refHessian); initialize_hessian = false; }
			void setTolerance( ScalarType refTolerance ){ tolerance = refTolerance; }
			void setMaxIter( UnsignedIntegerType refMaxIter ){ max_iter = refMaxIter; }
			void setHessianDummyInitialization( bool initialization ){ initialize_hessian = initialization; }
			// Observers
			const GradientOption<MODE,ScalarType>& getGradientOptions() const { return gradient_options; }
			const LineSearch<STRATEGY,ScalarType,UnsignedIntegerType>& getLineSearchMethod() const { return line_search; };
			ScalarType getTolerance() const { return tolerance; }
			const MLMatrix<ScalarType>& getHessian() const { return hessian_approx; }
			UnsignedIntegerType getMaxIter() const { return max_iter; }
			bool getHessianDummyInitialization() const { return initialize_hessian; }
			// Minimize
			template < 	typename Cost,
						typename DERIVED,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, ScalarType >::value,typename DERIVED::Scalar >::type >
			void minimize( const Cost& cost, Eigen::MatrixBase<DERIVED>& x ){
				
				// Initialization
				if (initialize_hessian){
					hessian_approx = MLMatrix<ScalarType>::Identity(x.size(),x.size());
				}
				MLVector<ScalarType> direction(x.size()),y_k(x.size());
				MLVector<ScalarType> gradient(x.size());
				ScalarType sqTolerance = tolerance*tolerance;
				UnsignedIntegerType iter = UnsignedIntegerType(0);
				ScalarType step;

				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== STARTING: BFGS Optimization ======\n" );

				cost.compute_gradient(x,gradient,gradient_options);

				while ( (gradient.squaredNorm()>sqTolerance) && (iter<max_iter) ){

					direction = hessian_approx.colPivHouseholderQr().solve(gradient);
					direction *= ScalarType(-1);
					step = line_search.LineSearch<STRATEGY,ScalarType>::template getStep<Cost,MODE,3,VERBOSITY_REF>(cost,x,gradient,direction,gradient_options);
					direction *= step;
					x += direction;
					y_k = -gradient;
					cost.compute_gradient(x,gradient,gradient_options);
					y_k += gradient;
					hessian_approx += (y_k*y_k.transpose())/(y_k.dot(direction)) - (((hessian_approx*direction)*direction.transpose())*hessian_approx)/(direction.dot(hessian_approx*direction));

					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( iter );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( ") Squared gradient norm = " );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( gradient.squaredNorm() );
					Utility::VerbosityLogger<2,VERBOSITY_REF>::log( "\n" );

					++iter;
				}

				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "INFO: Terminated in " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " out of " );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( max_iter );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( " iterations!\n" );
				Utility::VerbosityLogger<1,VERBOSITY_REF>::log( "====== DONE	: BFGS Optimization ======\n" );

			}
		private:
			GradientOption<MODE,ScalarType> gradient_options;
			LineSearch<STRATEGY,ScalarType,UnsignedIntegerType> line_search;
			UnsignedIntegerType max_iter = UnsignedIntegerType(1000);
			ScalarType tolerance = ScalarType(1e-5);
			bool initialize_hessian = true;
			MLMatrix<ScalarType> hessian_approx;
		};

	}

}

#endif