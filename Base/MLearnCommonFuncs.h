/*!
* 	\brief 		MLearn library definitions.
*	\details 	In this header file, common functions are 
*				defined.  
*	\author		phineasng
*					
*/


#ifndef MLEARN_CORE_BASE_COMMON_FUNCS_INCLUDED
#define MLEARN_CORE_BASE_COMMON_FUNCS_INCLUDED

#include <cmath>
#include <type_traits>

namespace MLearn{

	// 0-1 sign function
	template < typename returnType, typename inputType >
	inline returnType ml_zero_one_sign( inputType value ){
		return (returnType)( value > inputType(0) );
	}

	// signum
	template < typename inputType >
	inline inputType ml_signum( inputType value ){
		if ( value > inputType(0) ) return inputType(1);
		if ( value < inputType(0) ) return inputType(-1);
		return inputType(0);
	}

	// exponential
	template < typename inputType >
	inline inputType exponential( inputType value ){
		return exp(value);
	}

	// odd check
	template < typename IntegerType >
	inline bool is_odd(IntegerType x){
		static_assert(std::is_integral<IntegerType>::value,  
			"Only integer types supported. Sorry!");
		return x & 1;
	}

	// exponentiation by squaring
	template < typename BaseType >
	inline BaseType ml_pow(const BaseType& base, const int& exp){
		static_assert(std::is_integral<int>::value, 
			"Only integer values are supported as exponents. Sorry!");
		if ( (exp <= 10) && (exp >= -10) ){ // constants to be optimised 
			BaseType res = 1;
			for (int i = 1; i <= exp; ++i){
				res *= base;
			}
			return res;
		}else{ // exponentiation by squaring
			BaseType value = base;
			BaseType result = 1;
			int exponent = exp;
			if (exponent < 0){
				exponent = -exponent;
				value = BaseType(1)/base;
			}else if(exponent == 0){
				return BaseType(1);
			}
			while (exponent > 1){
				if ( is_odd(exponent) ){
					result *= value;
					value *= value;
					exponent = (exponent - 1) / 2;
				}else{
					value *= value;
					exponent = exponent / 2;
				}
			}
			return value*result;
		}
	}

} // End MLearn namespace

#endif