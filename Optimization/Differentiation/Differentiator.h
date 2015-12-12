#ifndef MLEARN_OPTIMIZATION_NUMERICAL_DIFFERENTIATOR
#define MLEARN_OPTIMIZATION_NUMERICAL_DIFFERENTIATOR

namespace MLearn{

	namespace Optimization{


		// Differentiation Mode
		enum class DifferentiationMode{
			ANALYTICAL,
			STOCHASTIC,
			NUMERICAL_FORWARD,
			NUMERICAL_CENTRAL,
			NUMERICAL_BACKWARD
		};

		// Differentiation options		
		template < DifferentiationMode MODE, typename ScalarType = double, typename IndexType = uint >
		struct GradientOption{
			// Constructor
			GradientOption( const ScalarType& ref_step = 1e-7 ): step_size(ref_step) {}
			// Fields
			ScalarType step_size;
		};

		template < typename ScalarType, typename IndexType >
		struct GradientOption<DifferentiationMode::ANALYTICAL, ScalarType, IndexType>{};

		template < typename ScalarType, typename IndexType >
		struct GradientOption<DifferentiationMode::STOCHASTIC, ScalarType, IndexType>{
			GradientOption() = delete;
			GradientOption( const MLVector< IndexType >& ref_to_sample ): to_sample(ref_to_sample) {}
			const MLVector< IndexType >& to_sample;
		};




		// Concrete  Differentiator
		template < DifferentiationMode MODE >
		class Differentiator{
		public:
			// --- gradient
			template < typename DifferentiableCost, typename ScalarType = double, typename IndexType = uint >
			static void compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const GradientOption< MODE, ScalarType, IndexType >& options = GradientOption< MODE, ScalarType, IndexType >());
		};

		// Specialization
		// --- analytical
		template<> template< typename DifferentiableCost, typename ScalarType, typename IndexType >
		void Differentiator<DifferentiationMode::ANALYTICAL>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const GradientOption< DifferentiationMode::ANALYTICAL, ScalarType, IndexType >& options ){
			cost.compute_analytical_gradient(x,gradient);
		}
		// --- numerical forward
		template<> template< typename DifferentiableCost, typename ScalarType, typename IndexType >
		void Differentiator<DifferentiationMode::NUMERICAL_FORWARD>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const GradientOption< DifferentiationMode::NUMERICAL_FORWARD, ScalarType, IndexType >& options ){
			gradient.resize(x.size());
			ScalarType inv_step_size = ScalarType(1)/options.step_size;
			MLVector< ScalarType > x1 = x;
			for ( decltype(x.size()) i = 0; i < x.size(); ++i ){
				x1[i] += options.step_size;
				gradient[i] = ( cost.evaluate(x1) - cost.evaluate(x) )*inv_step_size;
				x1[i] = x[i];
			}
		}
		// --- numerical central
		template<> template< typename DifferentiableCost, typename ScalarType, typename IndexType >
		void Differentiator<DifferentiationMode::NUMERICAL_CENTRAL>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const GradientOption< DifferentiationMode::NUMERICAL_CENTRAL, ScalarType, IndexType >& options ){
			gradient.resize(x.size());
			ScalarType inv_step_size = ScalarType(0.5)/options.step_size;
			MLVector< ScalarType > x1 = x;
			MLVector< ScalarType > x2 = x;
			for ( decltype(x.size()) i = 0; i < x.size(); ++i ){
				x1[i] += options.step_size;
				x2[i] -= options.step_size;
				gradient[i] = ( cost.evaluate(x1) - cost.evaluate(x2) )*inv_step_size;
				x1[i] = x[i];
				x2[i] = x[i];
			}
		}
		// --- numerical backward
		template<> template< typename DifferentiableCost, typename ScalarType, typename IndexType >
		void Differentiator<DifferentiationMode::NUMERICAL_BACKWARD>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const GradientOption< DifferentiationMode::NUMERICAL_BACKWARD, ScalarType, IndexType >& options ){
			gradient.resize(x.size());
			ScalarType inv_step_size = ScalarType(1)/options.step_size;
			MLVector< ScalarType > x1 = x;
			for ( decltype(x.size()) i = 0; i < x.size(); ++i ){
				x1[i] -= options.step_size;
				gradient[i] = ( cost.evaluate(x) - cost.evaluate(x1) )*inv_step_size;
				x1[i] = x[i];
			}
		}
		// --- stochastic
		template<> template< typename DifferentiableCost, typename ScalarType, typename IndexType >
		void Differentiator<DifferentiationMode::STOCHASTIC>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const GradientOption< DifferentiationMode::STOCHASTIC, ScalarType, IndexType >& options ){
			cost.compute_stochastic_gradient(x,gradient,options.to_sample);
		}

	}

}

#endif
