/*!
* 	\brief 		MLearn library definitions.
*	\details 	In this header file, common functions are 
*				defined.  
*	\author		phineasng
*					
*/


#ifndef MLEARN_CORE_BASE_COMMON_FUNCS_INCLUDED
#define MLEARN_CORE_BASE_COMMON_FUNCS_INCLUDED

namespace MLearn{
	// 0-1 sign function
	template < typename returnType, typename inputType >
	inline returnType ml_zero_one_sign( inputType value ){
		return (returnType)( value > inputType(0) );
	}

} // End MLearn namespace

#endif