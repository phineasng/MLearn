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

} // End MLearn namespace

#endif