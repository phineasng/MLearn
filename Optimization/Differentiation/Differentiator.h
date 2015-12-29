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
		template < 	DifferentiationMode MODE, typename ScalarType = double, typename IndexType = uint,
					typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, IndexType >::type >
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
			template < 	typename DifferentiableCost, 
						typename IndexType, 
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, IndexType >::type >
			static inline void compute_gradient(const DifferentiableCost& cost,const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< MODE, typename DERIVED::Scalar, IndexType >& options = GradientOption< MODE, typename DERIVED::Scalar, IndexType >() );
		};

		// Specialization
		// --- analytical
		template <>
		class Differentiator<DifferentiationMode::ANALYTICAL>{
		public:
			// --- gradient
			template < 	typename DifferentiableCost, 
						typename IndexType, 
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, IndexType >::type >
			static inline void compute_gradient(const DifferentiableCost& cost,const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< DifferentiationMode::ANALYTICAL, typename DERIVED::Scalar, IndexType >& options = GradientOption< DifferentiationMode::ANALYTICAL, typename DERIVED::Scalar, IndexType >() ){
				cost.compute_analytical_gradient(x,gradient);
			}	
		};

		// --- numerical forward
		template <>
		class Differentiator<DifferentiationMode::NUMERICAL_FORWARD>{
		public:
			// --- gradient
			template < 	typename DifferentiableCost, 
						typename IndexType, 
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, IndexType >::type >
			static inline void compute_gradient(const DifferentiableCost& cost,const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< DifferentiationMode::NUMERICAL_FORWARD, typename DERIVED::Scalar, IndexType >& options = GradientOption< DifferentiationMode::NUMERICAL_FORWARD, typename DERIVED::Scalar, IndexType >() ){
				gradient.resize(x.size());
				typename DERIVED::Scalar inv_step_size = typename DERIVED::Scalar(1)/options.step_size;
				MLVector< typename DERIVED::Scalar > x1 = x;
				for ( decltype(x.size()) i = 0; i < x.size(); ++i ){
					x1[i] += options.step_size;
					gradient[i] = ( cost.eval_for_numerical_gradient(x1) - cost.eval_for_numerical_gradient(x) )*inv_step_size;
					x1[i] = x[i];
				}
			}
		};
		// --- numerical central
		template <>
		class Differentiator<DifferentiationMode::NUMERICAL_CENTRAL>{
		public:
			// --- gradient
			template < 	typename DifferentiableCost, 
						typename IndexType, 
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, IndexType >::type >
			static inline void compute_gradient(const DifferentiableCost& cost,const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< DifferentiationMode::NUMERICAL_CENTRAL, typename DERIVED::Scalar, IndexType >& options = GradientOption< DifferentiationMode::NUMERICAL_CENTRAL, typename DERIVED::Scalar, IndexType >() ){
				gradient.resize(x.size());
				typename DERIVED::Scalar inv_step_size = typename DERIVED::Scalar(0.5)/options.step_size;
				MLVector< typename DERIVED::Scalar > x1 = x;
				MLVector< typename DERIVED::Scalar > x2 = x;
				for ( decltype(x.size()) i = 0; i < x.size(); ++i ){
					x1[i] += options.step_size;
					x2[i] -= options.step_size;
					gradient[i] = ( cost.eval_for_numerical_gradient(x1) - cost.eval_for_numerical_gradient(x2) )*inv_step_size;
					x1[i] = x[i];
					x2[i] = x[i];
				}
			}
		};
		// --- numerical backward
		template <>
		class Differentiator<DifferentiationMode::NUMERICAL_BACKWARD>{
		public:
			// --- gradient
			template < 	typename DifferentiableCost, 
						typename IndexType, 
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, IndexType >::type >
			static inline void compute_gradient(const DifferentiableCost& cost,const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< DifferentiationMode::NUMERICAL_BACKWARD, typename DERIVED::Scalar, IndexType >& options = GradientOption< DifferentiationMode::NUMERICAL_BACKWARD, typename DERIVED::Scalar, IndexType >() ){
				gradient.resize(x.size());
				typename DERIVED::Scalar inv_step_size = typename DERIVED::Scalar(1)/options.step_size;
				MLVector< typename DERIVED::Scalar > x1 = x;
				for ( decltype(x.size()) i = 0; i < x.size(); ++i ){
					x1[i] -= options.step_size;
					gradient[i] = ( cost.eval_for_numerical_gradient(x) - cost.eval_for_numerical_gradient(x1) )*inv_step_size;
					x1[i] = x[i];
				}
			}
		};
		// --- stochastic
		template <>
		class Differentiator<DifferentiationMode::STOCHASTIC>{
		public:
			// --- gradient
			template < 	typename DifferentiableCost, 
						typename IndexType, 
						typename DERIVED,
				   		typename DERIVED_2,
				   		typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   		typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
						typename = typename std::enable_if< std::is_floating_point<typename DERIVED::Scalar>::value , typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type,
						typename = typename std::enable_if< std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value, IndexType >::type >
			static inline void compute_gradient(const DifferentiableCost& cost,const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const GradientOption< DifferentiationMode::NUMERICAL_FORWARD, typename DERIVED::Scalar, IndexType >& options = GradientOption< DifferentiationMode::NUMERICAL_FORWARD, typename DERIVED::Scalar, IndexType >() ){
				cost.compute_stochastic_gradient(x,gradient,options.to_sample);
			}
		};

	}

}

#endif
