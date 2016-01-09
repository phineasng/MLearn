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
			SOFT_PLUS,
			LOGISTIC,
			HYPER_TAN
		};

		template< ActivationType TYPE >
		class ActivationFunction {
		public:

			template< 	typename ScalarType >
			static inline ScalarType evaluate( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the ACTIVATION FUNCTION" );
				return ScalarType(0);
			}

			template< 	typename ScalarType >
			static inline ScalarType first_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the ACTIVATION FUNCTION" );
				return ScalarType(0);
			}

			template< 	typename ScalarType >
			static inline ScalarType second_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				MLEARN_FORCED_WARNING_MESSAGE( "EMPTY IMPLEMENTATION of the ACTIVATION FUNCTION" );
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::STEP> {
		public:

			template< 	typename ScalarType >
			static inline ScalarType evaluate( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				if (signbit(x)){
					return ScalarType(1);
				}else{
					return ScalarType(0);
				}
			}

			template< 	typename ScalarType >
			static inline ScalarType first_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				MLEARN_FORCED_WARNING_MESSAGE( "STEP function is not differentiable!" );
				return ScalarType(0);
			}

			template< 	typename ScalarType >
			static inline ScalarType second_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				MLEARN_FORCED_WARNING_MESSAGE( "STEP function is not differentiable!" );
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::LINEAR> {
		public:

			template< 	typename ScalarType >
			static inline ScalarType evaluate( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return x;
			}

			template< 	typename ScalarType >
			static inline ScalarType first_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return ScalarType(1);
			}

			template< 	typename ScalarType >
			static inline ScalarType second_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::RECTIFIER> {
		public:

			template< 	typename ScalarType >
			static inline ScalarType evaluate( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				if ( x > ScalarType(0) )
					return x;
				else
					return ScalarType(0);
			}

			template< 	typename ScalarType >
			static inline ScalarType first_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				if ( x > ScalarType(0) )
					return ScalarType(1);
				else
					return ScalarType(0);
			}

			template< 	typename ScalarType >
			static inline ScalarType second_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return ScalarType(0);
			}

		};

		template<>
		class ActivationFunction<ActivationType::LOGISTIC> {
		public:

			template< 	typename ScalarType >
			static inline ScalarType evaluate( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return ScalarType(1)/(ScalarType(1) + std::exp(-x));
			}

			template< 	typename ScalarType >
			static inline ScalarType first_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				ScalarType temp = evaluate(x);
				return temp*(ScalarType(1)-temp);
			}

			template< 	typename ScalarType >
			static inline ScalarType second_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				ScalarType temp = evaluate(x);
				return temp*(ScalarType(1)-temp)*(ScalarType(1)-ScalarType(2)*temp);
			}

		};

		template<>
		class ActivationFunction<ActivationType::HYPER_TAN> {
		public:

			template< 	typename ScalarType >
			static inline ScalarType evaluate( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return std::tanh(x);
			}

			template< 	typename ScalarType >
			static inline ScalarType first_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				ScalarType temp = ScalarType(1)/std::cosh(x);
				return temp*temp;
			}

			template< 	typename ScalarType >
			static inline ScalarType second_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return ScalarType(-2)*(evaluate(x))*(first_derivative(x));
			}

		};

		template<>
		class ActivationFunction<ActivationType::SOFT_PLUS> {
		public:

			template< 	typename ScalarType >
			static inline ScalarType evaluate( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				if ( x > ScalarType(100) ){
					return x;
				}
				return std::log(1 + std::exp(x));
			}

			template< 	typename ScalarType >
			static inline ScalarType first_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return ActivationFunction<ActivationType::LOGISTIC>::evaluate(x);
			}

			template< 	typename ScalarType >
			static inline ScalarType second_derivative( ScalarType x ){
				static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
				return ActivationFunction<ActivationType::LOGISTIC>::first_derivative(x);
			}

		};
	}

}

#endif