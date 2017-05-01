/*!	\file 		MLearnKernels.h
* 	\brief 		Positive Definite Kernels used in various ML algorithms.
*	\details 	Implementation of Positive Definite Kernels
*				(https://en.wikipedia.org/wiki/Positive-definite_kernel).
*	\author		phineasng
*/
#ifndef MLEARN_BASE_KERNELS_HFILE
#define MLEARN_BASE_KERNELS_HFILE


// Eigen includes
#include <Eigen/Core>

// STL includes
#include <cmath>
#include <type_traits>
#include <tuple>

// MLearn includes
#include "MLearnCommonFuncs.h"

// Boost includes
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/special_functions/gamma.hpp>


// Useful macros
#define KERNEL_COMPUTE_TEMPLATE_START(x,y)\
	template < typename DERIVED1, typename DERIVED2 >\
	typename DERIVED1::Scalar compute(const Eigen::MatrixBase<DERIVED1>& x,\
									  const Eigen::MatrixBase<DERIVED2>& y) const\
	{\
		static_assert(DERIVED1::ColsAtCompileTime == 1,\
			"First input has to be a column vector(or compatible structure)");\
		static_assert(DERIVED2::ColsAtCompileTime == 1,\
			"Second input has to be a column vector(or compatible structure)");\
		static_assert(std::is_floating_point<typename DERIVED1::Scalar>::value\
			&& std::is_same<typename DERIVED1::Scalar,\
			typename DERIVED2::Scalar>::value,\
			"Scalar types have to be the same and floating point!");

#define KERNEL_COMPUTE_TEMPLATE_END }

#define ENABLE_TYPE_CONST_GET(REF_INDEX, TYPE)\
	template< uint QUERY_INDEX >\
	inline const typename\
		std::enable_if< QUERY_INDEX == REF_INDEX, TYPE >::type&\
		get() const

#define ENABLE_TYPE_GET(REF_INDEX, TYPE)\
	template< uint QUERY_INDEX >\
	inline typename\
		std::enable_if< QUERY_INDEX == REF_INDEX, TYPE >::type& get()

namespace MLearn{

	enum class KernelType{
		// LINEAR is implemented separately from polynomial 
		// to avoid unnecessary computations/checks
		LINEAR,  
		POLYNOMIAL, 
		RBF,
		LAPLACIAN,
		ABEL,
		CONSTANT, 
		MIN,
		MATERN32,
		RATIONAL_QUADRATIC,
		PERIODIC,
		WHITE_NOISE,
		SUM,
		PRODUCT
	};

	template < KernelType KT, typename ... OTHER >
	class Kernel{};

	/**
	*	\brief	Linear Kernel
	*/
	template <>
	class Kernel< KernelType::LINEAR >{
	public:
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			return x.dot(y);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Polynomial kernel (default: square function with no intercept)
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::POLYNOMIAL, FLOAT_TYPE >{
	private:
		uint N = 2; // exponent
		FLOAT_TYPE r = FLOAT_TYPE(0.0); // bias/intercept
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint N_index = 0;
		static const uint r_index = 1; 
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(N_index, uint){
			return N;
		}
		ENABLE_TYPE_GET(N_index, uint){
			return N;
		}
		ENABLE_TYPE_CONST_GET(r_index, FLOAT_TYPE){
			return r;
		}
		ENABLE_TYPE_GET(r_index, FLOAT_TYPE){
			return r;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			FLOAT_TYPE base = x.dot(y) + r;
			return ml_pow(base, N);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief RBF kernel (default: sigma_squared = 1)
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::RBF, FLOAT_TYPE >{
	private:
		FLOAT_TYPE sigma_sq = FLOAT_TYPE(1.0); // "variance"
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint sigma_sq_index = 0;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(sigma_sq_index, FLOAT_TYPE){
			return sigma_sq;
		}
		ENABLE_TYPE_GET(sigma_sq_index, FLOAT_TYPE){
			return sigma_sq;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			MLEARN_ASSERT(sigma_sq > 0., "Sigma squared has to be positive!");
			FLOAT_TYPE l2_dist_squared = (x - y).squaredNorm()*0.5/sigma_sq;
			return std::exp(-l2_dist_squared);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Laplacian kernel (default: alpha = 1)
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::LAPLACIAN, FLOAT_TYPE >{
	private:
		FLOAT_TYPE alpha = FLOAT_TYPE(1.0); // "variance"
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint alpha_index = 0;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(alpha_index, FLOAT_TYPE){
			return alpha;
		}
		ENABLE_TYPE_GET(alpha_index, FLOAT_TYPE){
			return alpha;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			MLEARN_ASSERT(alpha > 0., "Sigma squared has to be positive!");
			FLOAT_TYPE l2_dist = (x - y).norm();
			return std::exp(-alpha*l2_dist);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Abel kernel (default: alpha = 1)
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::ABEL, FLOAT_TYPE >{
	private:
		FLOAT_TYPE alpha = FLOAT_TYPE(1.0); // "variance"
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint alpha_index = 0;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(alpha_index, FLOAT_TYPE){
			return alpha;
		}
		ENABLE_TYPE_GET(alpha_index, FLOAT_TYPE){
			return alpha;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			MLEARN_ASSERT(alpha > 0., "Sigma squared has to be positive!");
			FLOAT_TYPE l1_dist = (x - y).template lpNorm<1>();
			return std::exp(-alpha*l1_dist);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Constant kernel (default: C = 1)
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::CONSTANT, FLOAT_TYPE >{
	private:
		FLOAT_TYPE C = FLOAT_TYPE(1.0); // "variance"
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint C_index = 0;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(C_index, FLOAT_TYPE){
			return C;
		}
		ENABLE_TYPE_GET(C_index, FLOAT_TYPE){
			return C;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			return C;
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Min kernel
	*/
	template <>
	class Kernel< KernelType::MIN >{
	public:
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			return 0.5*(x.template lpNorm<1>() - 
				(x-y).template lpNorm<1>() + y.template lpNorm<1>());
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief White noise kernel
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::WHITE_NOISE, FLOAT_TYPE >{
	private:
		FLOAT_TYPE noise_level = FLOAT_TYPE(1.0);
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint noise_level_index = 0;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(noise_level_index, FLOAT_TYPE){
			return noise_level;
		}
		ENABLE_TYPE_GET(noise_level_index, FLOAT_TYPE){
			return noise_level;
		}
		// compute function
		// adding these curly brackets just because the sublime syntax checker
		// is complaining, but it shouldn't be necessary
		KERNEL_COMPUTE_TEMPLATE_START(x,y){ 
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			FLOAT_TYPE TOLERANCE = 1e-30;
			FLOAT_TYPE l2_dist = (x - y).squaredNorm();
			if (l2_dist < TOLERANCE){
				return noise_level;
			}else{
				return 0.;
			}
			return std::min(x.minCoeff(), y.minCoeff());
		}KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Matern kernel (default: smoothness = 2.5, length = 1.0)
	*/
	template < typename LENGTH_TYPE >
	class Kernel< KernelType::MATERN32, LENGTH_TYPE >{
	private:
		LENGTH_TYPE length_scale = LENGTH_TYPE(1.0);
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint length_scale_index = 0;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(length_scale_index, LENGTH_TYPE){
			return length_scale;
		}
		ENABLE_TYPE_GET(length_scale_index, LENGTH_TYPE){
			return length_scale;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				LENGTH_TYPE>::value, "Scalar types must be the same!");
			MLEARN_ASSERT(smoothness > 0, 
				"Smoothness parameter has to be positive!");
			MLEARN_ASSERT(length_scale > 0, 
				"Smoothness parameter has to be positive!");
			LENGTH_TYPE d = (x - y).norm()*SQRT_3/length_scale;
			return (1 + d)*std::exp(-d);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Rational quadratic kernel (default: alpha = 1.0, length = 1.0)
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::RATIONAL_QUADRATIC, FLOAT_TYPE >{
	private:
		FLOAT_TYPE alpha = FLOAT_TYPE(0.1); 
		FLOAT_TYPE length_squared = FLOAT_TYPE(1.0);
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint alpha_index = 0;
		static const uint length_squared_index = 1;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(alpha_index, FLOAT_TYPE){
			return alpha;
		}
		ENABLE_TYPE_GET(alpha_index, FLOAT_TYPE){
			return alpha;
		}
		ENABLE_TYPE_CONST_GET(length_squared_index, FLOAT_TYPE){
			return length_squared;
		}
		ENABLE_TYPE_GET(length_squared_index, FLOAT_TYPE){
			return length_squared;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			MLEARN_ASSERT(alpha > 0, 
				"Smoothness parameter has to be positive!");
			MLEARN_ASSERT(length_squared > 0, 
				"Smoothness parameter has to be positive!");
			FLOAT_TYPE d = (x - y).squaredNorm()*0.5/(alpha*length_squared);
			FLOAT_TYPE base = (1.0 + d);
			return std::pow(base, -alpha);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief PERIODIC kernel (default: period = 1.0, length_squared = 1.0)
	*/
	template < typename FLOAT_TYPE >
	class Kernel< KernelType::PERIODIC, FLOAT_TYPE >{
	private:
		FLOAT_TYPE period = FLOAT_TYPE(1.0); 
		FLOAT_TYPE length_squared = FLOAT_TYPE(1.0);
	public:
		// Indeces to retrieve and get/set hyperparameters
		static const uint period_index = 0;
		static const uint length_squared_index = 1;
		// hyperparams getter/setter
		ENABLE_TYPE_CONST_GET(period_index, FLOAT_TYPE){
			return period;
		}
		ENABLE_TYPE_GET(period_index, FLOAT_TYPE){
			return period;
		}
		ENABLE_TYPE_CONST_GET(length_squared_index, FLOAT_TYPE){
			return length_squared;
		}
		ENABLE_TYPE_GET(length_squared_index, FLOAT_TYPE){
			return length_squared;
		}
		// compute function
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			static_assert(std::is_same<typename DERIVED1::Scalar,
				FLOAT_TYPE>::value, "Scalar types must be the same!");
			MLEARN_ASSERT(period > 0, 
				"Smoothness parameter has to be positive!");
			MLEARN_ASSERT(length_squared > 0, 
				"Smoothness parameter has to be positive!");
			FLOAT_TYPE v = (x - y).norm()*M_PI/period;
			v = std::sin(v);
			return std::exp(-v*v*2.0/length_squared);
		KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief 		This struct is used as template parameter for the get() 
	*				function of composite kernels.
	*	\details 	K_IDX is the id of the kernel in the composite kernel.
	*				P_IDX is the index of the parameter in the K_IDth kernel.
	*/
	template <uint K_IDX, uint P_IDX >
	struct CompKerIndex{
		static const uint K_ID = K_IDX;
		static const uint PARAM_ID = P_IDX;
	};

	/**
	*	\brief 		Base class for composite kernel
	*	\details 	This is the base class for all composite kernels.
	*				E.g. Sum kernel and product kernel.
	*				The class contains common utilities like the get() function
	*				and instantiation of the sub-kernels.
	*/
	template < typename ...KERNEL_TYPES >
	class BaseCompositeKernel{
	protected:
		BaseCompositeKernel(){}
		std::tuple<KERNEL_TYPES...> kernels;
	private:
		template < uint K_ID, uint P_ID >
		using RETURN_TYPE = 
			decltype(std::get<K_ID>(kernels).template get<P_ID>());
	public:
		// const getter
		template < typename CK_IDX >
		RETURN_TYPE<CK_IDX::K_ID, CK_IDX::PARAM_ID> get() const{
			return std::get<CK_IDX::K_ID>(kernels).get<CK_IDX::PARAM_ID>();
		}
		// getter/setter
		template < typename CK_IDX >
		RETURN_TYPE<CK_IDX::K_ID, CK_IDX::PARAM_ID> get(){
			return std::get<CK_IDX::K_ID>(kernels).get<CK_IDX::PARAM_ID>();
		}
	};

	/**
	*	\brief Sum kernel
	*/
	template<typename ...KERNEL_TYPES>
	class Kernel<KernelType::SUM, KERNEL_TYPES...>: 
		public BaseCompositeKernel<KERNEL_TYPES...>{
	private:
		typedef BaseCompositeKernel<KERNEL_TYPES...> Base; 
		template <uint N, typename DERIVED1, typename DERIVED2>
		inline typename std::enable_if< N != 0, typename DERIVED1::Scalar>::type
				sum(const DERIVED1& x, const DERIVED2& y, 
				const typename DERIVED1::Scalar& cum_sum) const{
			return sum<N-1>
				(x, y, cum_sum + std::get<N-1>(Base::kernels).compute(x, y));
		} 
		template <uint N, typename DERIVED1, typename DERIVED2>
		inline typename std::enable_if< N == 0, typename DERIVED1::Scalar>::type
				sum(const DERIVED1& x, const DERIVED2& y, 
				const typename DERIVED1::Scalar& cum_sum) const{
			return cum_sum;
		}
	public:
		KERNEL_COMPUTE_TEMPLATE_START(x,y){
			return sum<sizeof...(KERNEL_TYPES)>(x, y, 0.);
		}KERNEL_COMPUTE_TEMPLATE_END
	};

	/**
	*	\brief Product kernel
	*/
	template<typename ...KERNEL_TYPES>
	class Kernel<KernelType::PRODUCT, KERNEL_TYPES...>: 
		public BaseCompositeKernel<KERNEL_TYPES...>{
	private:
		typedef BaseCompositeKernel<KERNEL_TYPES...> Base; 
		template <uint N, typename DERIVED1, typename DERIVED2>
		inline typename std::enable_if< N != 0, typename DERIVED1::Scalar>::type
				prod(const DERIVED1& x, const DERIVED2& y, 
				const typename DERIVED1::Scalar& cum_prod) const{
			return prod<N-1>
				(x, y, cum_prod*std::get<N-1>(Base::kernels).compute(x, y));
		} 
		template <uint N, typename DERIVED1, typename DERIVED2>
		inline typename std::enable_if< N == 0, typename DERIVED1::Scalar>::type
				prod(const DERIVED1& x, const DERIVED2& y, 
				const typename DERIVED1::Scalar& cum_prod) const{
			return cum_prod;
		}
	public:
		KERNEL_COMPUTE_TEMPLATE_START(x,y){
			return prod<sizeof...(KERNEL_TYPES)>(x, y, 1.);
		}KERNEL_COMPUTE_TEMPLATE_END
	};


}

#endif