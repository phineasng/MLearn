#ifndef TEST_COMMON_H_FILE_MLEARN
#define TEST_COMMON_H_FILE_MLEARN

#define TEST_FLOAT_TOLERANCE 1e-10

// MLearn includes
#include <MLearn/Core>

// Eigen includes
#include <Eigen/Core>

namespace TestUtils{
	/*!
		\brief compute inf norm of difference of input matrices
	*/
	template <typename DERIVED1, typename DERIVED2>
	typename DERIVED1::Scalar diff_norm(
			const Eigen::MatrixBase<DERIVED1>& m1,
			const Eigen::MatrixBase<DERIVED2>& m2){
		return (m1 - m2).template lpNorm<Eigen::Infinity>();
	}
}

#endif