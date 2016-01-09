#ifndef MLEARN_LOSS_FUNCTION_COMMON_INCLUDE
#define MLEARN_LOSS_FUNCTION_COMMON_INCLUDE

// MLearn includes
#include "MLearnTypes.h"

// STL includes
#include <type_traits>
#include <cmath>

namespace MLearn{

	enum class LossType{
		NONE,
		INDICATOR,
		L2_SQUARED,
		CROSS_ENTROPY,
		SOFTMAX_CROSS_ENTROPY
	};

	template < LossType TYPE >
	class LossFunction{
	public:
		template < typename DERIVED,
				   typename DERIVED_2 >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
			MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the LOSS FUNCTION" );
			return (typename DERIVED::Scalar(2));
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename DERIVED_3 >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, Eigen::MatrixBase<DERIVED_3>& gradient ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1) && (DERIVED_3::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_3::Scalar>::value, "Scalar types have to be the same and floating point!" );
			MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the LOSS FUNCTION" );
		}

	};

	template <>
	class LossFunction<LossType::INDICATOR>{
	public:
		template < typename DERIVED,
				   typename DERIVED_2 >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
			return ( output.array() != exp_output.array() ).sum();
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename DERIVED_3 >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, Eigen::MatrixBase<DERIVED_3>& gradient ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1) && (DERIVED_3::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_3::Scalar>::value, "Scalar types have to be the same and floating point!" );
			MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the LOSS FUNCTION" );
		}

	};

	template <>
	class LossFunction<LossType::L2_SQUARED>{
	public:
		template < typename DERIVED,
				   typename DERIVED_2 >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
			return (output-exp_output).squaredNorm();
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename DERIVED_3 >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, Eigen::MatrixBase<DERIVED_3>& gradient ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1) && (DERIVED_3::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_3::Scalar>::value, "Scalar types have to be the same and floating point!" );
			gradient = (typename DERIVED::Scalar(2))*(output-exp_output);
		}

	};

	template <>
	class LossFunction<LossType::CROSS_ENTROPY>{
	public:
		template < typename DERIVED,
				   typename DERIVED_2 >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
			return - ( exp_output.dot( output.unaryExpr( std::pointer_to_unary_function< typename DERIVED::Scalar, typename DERIVED::Scalar>(log) ) ) );
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename DERIVED_3 >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, Eigen::MatrixBase<DERIVED_3>& gradient ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1) && (DERIVED_3::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_3::Scalar>::value, "Scalar types have to be the same and floating point!" );
			gradient = - exp_output.cwiseQuotient( output );
		}

	};

	template <>
	class LossFunction<LossType::SOFTMAX_CROSS_ENTROPY>{
	public:
		template < typename DERIVED,
				   typename DERIVED_2 >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value, "Scalar types have to be the same and floating point!" );
			typename DERIVED::Scalar res = typename DERIVED::Scalar(0);
			typename DERIVED::Scalar max = output.maxCoeff();
			res = - output.dot(exp_output);
			res += exp_output.sum()*( max + std::log( ( output.array() - max ).unaryExpr( std::pointer_to_unary_function< typename DERIVED::Scalar, typename DERIVED::Scalar>(exp) ).sum() ) );
			return res;
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename DERIVED_3 >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, Eigen::MatrixBase<DERIVED_3>& gradient ){
			static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1) && (DERIVED_3::ColsAtCompileTime == 1), "Inputs have to be column vectors (or compatible structures)!" );
			static_assert( std::is_floating_point<typename DERIVED::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value && std::is_same<typename DERIVED::Scalar, typename DERIVED_3::Scalar>::value, "Scalar types have to be the same and floating point!" );
			typename DERIVED::Scalar max = output.maxCoeff();
			gradient = (output.array() - max).unaryExpr( std::pointer_to_unary_function< typename DERIVED::Scalar, typename DERIVED::Scalar>(exp) );
			gradient *= exp_output.sum()/gradient.sum();
			gradient -= exp_output; 
		}

	};
}

#endif