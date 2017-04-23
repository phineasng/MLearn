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

// STD includes
#include <cmath>

// Useful macros
#define KERNEL_COMPUTE_TEMPLATE_START(x,y)\
	template < typename DERIVED1, typename DERIVED2 >\
	typename DERIVED1::Scalar compute(const Eigen::MatrixBase<DERIVED1>& x,\
									  const Eigen::MatrixBase<DERIVED2>& y){\
		static_assert(DERIVED1::ColsAtCompileTime == 1,\ 
			"First input has to be a column vector(or compatible structure)");\
		static_assert(DERIVED2::ColsAtCompileTime == 1,\
			"Second input has to be a column vector(or compatible structure)");\
		static_assert(std::is_floating_point<typename DERIVED::Scalar>::value\ 
			&& std::is_same<typename DERIVED1::Scalar,\ 
			typename DERIVED2::Scalar>::value,\ 
			"Scalar types have to be the same and floating point!");\

#define KERNEL_COMPUTE_TEMPLATE_END }

namespace MLearn{

	enum class KernelType{
		// LINEAR is implemented separately from polynomial 
		// to avoid unnecessary computations/checks
		LINEAR,  
		POLYNOMIAL, 
		GAUSSIAN,
		LAPLACIAN,
		ABEL,
		SOBOLEV_GEN,
		PALEY_WIENER
	};

	template < KernelType KT >
	class Kernel{};

	class Kernel< KernelType::LINEAR >{
	public:
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			return x.dot(y)
		KERNEL_COMPUTE_TEMPLATE_END
	};
}

#endif