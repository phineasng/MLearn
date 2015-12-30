#ifndef MLEARN_ACTIVATION_FUNCTION_INCLUDE
#define MLEARN_ACTIVATION_FUNCTION_INCLUDE

// STL includes
#include <type_traits>
#include <cmath>
#include <limits>

namespace MLearn{

	namespace NeuralNets{

		enum class ActivationType{
			STEP,
			LINEAR,
			RECTIFIER,
			LOGISTIC,
			HYPER_TAN
		};

		template< ActivationType TYPE >
		class ActivationFunction {
		public:

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType evaluate( ScalarType x ){
				MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the ACTIVATION FUNCTION" );
				return ScalarType(0);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType first_derivative( ScalarType x ){
				MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the ACTIVATION FUNCTION" );
				return ScalarType(0);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType second_derivative( ScalarType x ){
				MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the ACTIVATION FUNCTION" );
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::STEP> {
		public:

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType evaluate( ScalarType x ){
				if (signbit(x)){
					return ScalarType(1);
				}else{
					return ScalarType(0);
				}
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType first_derivative( ScalarType x ){
				MLEARN_FORCED_WARNING_MESSAGE( "STEP function is not differentiable!" );
				return ScalarType(0);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType second_derivative( ScalarType x ){
				MLEARN_FORCED_WARNING_MESSAGE( "STEP function is not differentiable!" );
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::LINEAR> {
		public:

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType evaluate( ScalarType x ){
				return x;
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType first_derivative( ScalarType x ){
				return ScalarType(1);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType second_derivative( ScalarType x ){
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::RECTIFIER> {
		public:

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType evaluate( ScalarType x ){
				if ( x > ScalarType(0) )
					return x;
				else
					return ScalarType(0);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType first_derivative( ScalarType x ){
				if ( x > ScalarType(0) )
					return ScalarType(1);
				else
					return ScalarType(0);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType second_derivative( ScalarType x ){
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::LOGISTIC> {
		public:

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType evaluate( ScalarType x ){
				return ScalarType(1)/(ScalarType(1) + std::exp(-x));
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType first_derivative( ScalarType x ){
				ScalarType temp = evaluate(x);
				return temp*(ScalarType(1)-temp);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType second_derivative( ScalarType x ){
				ScalarType temp = evaluate(x);
				return temp*(ScalarType(1)-temp)*(ScalarType(1)-ScalarType(2)*temp);
			}

		};

		template<>
		class ActivationFunction<ActivationType::HYPER_TAN> {
		public:

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType evaluate( ScalarType x ){
				return std::tanh(x);
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType first_derivative( ScalarType x ){
				ScalarType temp = ScalarType(1)/std::cosh(x);
				return temp*temp;
			}

			template< 	typename ScalarType,
						typename = typename std::enable_if< (std::is_floating_point<ScalarType>::value && std::is_signed<ScalarType>::value), void >::type >
			static ScalarType second_derivative( ScalarType x ){
				return ScalarType(-2)*(evaluate(x))*(first_derivative(x));
			}

		};

	}

}

#endif