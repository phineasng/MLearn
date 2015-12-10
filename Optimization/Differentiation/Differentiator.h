#ifndef MLEARN_OPTIMIZATION_NUMERICAL_DIFFERENTIATOR
#define MLEARN_OPTIMIZATION_NUMERICAL_DIFFERENTIATOR

namespace MLearn{

	namespace Optimization{


		// Differentiation Mode
		enum class DifferentiationMode{
			ANALYTICAL,
			NUMERICAL_FORWARD,
			NUMERICAL_CENTRAL,
			NUMERICAL_BACKWARD
		};

		// Differentiation options
		template < DifferentiationMode MODE, typename ScalarType >
		struct NumericalGradientOption{
			// Constructor
			NumericalGradientOption(): step_size(1e-7) {}
			// Fields
			ScalarType step_size;
		};

		// Concrete Numerical Differentiator
		template < DifferentiationMode MODE >
		class Differentiator{
		public:
			// --- gradient
			template < typename DifferentiableCost, typename ScalarType >
			static void compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const NumericalGradientOption< MODE, ScalarType >& options = NumericalGradientOption< MODE, ScalarType >());
		};

		// Specialization
		// --- analytical
		template<> template< typename DifferentiableCost, typename ScalarType >
		void Differentiator<DifferentiationMode::ANALYTICAL>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const NumericalGradientOption< DifferentiationMode::ANALYTICAL, ScalarType >& options ){
			cost.compute_analytical_gradient(x,gradient);
		}
		// --- numerical forward
		template<> template< typename DifferentiableCost, typename ScalarType >
		void Differentiator<DifferentiationMode::NUMERICAL_FORWARD>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const NumericalGradientOption< DifferentiationMode::NUMERICAL_FORWARD, ScalarType >& options ){
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
		template<> template< typename DifferentiableCost, typename ScalarType >
		void Differentiator<DifferentiationMode::NUMERICAL_CENTRAL>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const NumericalGradientOption< DifferentiationMode::NUMERICAL_CENTRAL, ScalarType >& options ){
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
		template<> template< typename DifferentiableCost, typename ScalarType >
		void Differentiator<DifferentiationMode::NUMERICAL_BACKWARD>::compute_gradient(const DifferentiableCost& cost,const MLVector< ScalarType >& x, MLVector< ScalarType >& gradient, const NumericalGradientOption< DifferentiationMode::NUMERICAL_BACKWARD, ScalarType >& options ){
			gradient.resize(x.size());
			ScalarType inv_step_size = ScalarType(1)/options.step_size;
			MLVector< ScalarType > x1 = x;
			for ( decltype(x.size()) i = 0; i < x.size(); ++i ){
				x1[i] -= options.step_size;
				gradient[i] = ( cost.evaluate(x) - cost.evaluate(x1) )*inv_step_size;
				x1[i] = x[i];
			}
		}

	}

}

#endif
