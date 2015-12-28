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
		CROSS_ENTROPY
	};

	template < LossType TYPE >
	class LossFunction{
	public:
		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the LOSS FUNCTION" );
			return (typename DERIVED::Scalar(2));
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, MLVector<typename DERIVED::Scalar>& gradient ){
			MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the LOSS FUNCTION" );
		}

	};

	template <>
	class LossFunction<LossType::INDICATOR>{
	public:
		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			return ( output.array() != exp_output.array() ).sum();
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, MLVector<typename DERIVED::Scalar>& gradient ){
			MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the LOSS FUNCTION" );
		}

	};

	template <>
	class LossFunction<LossType::L2_SQUARED>{
	public:
		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			return (output-exp_output).squaredNorm();
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, MLVector<typename DERIVED::Scalar>& gradient ){
			gradient = (typename DERIVED::Scalar(2))*(output-exp_output);
		}

	};

	template <>
	class LossFunction<LossType::CROSS_ENTROPY>{
	public:
		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline typename DERIVED::Scalar evaluate( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output ){
			return - ( exp_output.dot( output.unaryExpr( std::pointer_to_unary_function< typename DERIVED::Scalar, typename DERIVED::Scalar>(log) ) ) );
		}

		template < typename DERIVED,
				   typename DERIVED_2,
				   typename = typename std::enable_if< DERIVED::ColsAtCompileTime == 1 , DERIVED >::type,
				   typename = typename std::enable_if< DERIVED_2::ColsAtCompileTime == 1 , DERIVED_2 >::type,
				   typename = typename std::enable_if< std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,typename DERIVED::Scalar >::type >
		static inline void gradient( const Eigen::MatrixBase<DERIVED>& output, const Eigen::MatrixBase<DERIVED_2>& exp_output, MLVector<typename DERIVED::Scalar>& gradient ){
			gradient = - exp_output.cwiseQuotient( output );
		}

	};
}

#endif