/*!
* 	\brief 		MLearn library type traits.
*	\details 	In this header file, useful type traits 
*				are introduced.
*	\author		phineasng
*/

#ifndef MLEARN_CORE_BASE_TYPE_TRAITS_INCLUDED
#define MLEARN_CORE_BASE_TYPE_TRAITS_INCLUDED

namespace MLearn{

	namespace TypeTraits{

 		/*!
		*	\brief 		defines_ScalarType type traits
		*	\details 	It checks if a class T defines a typedef field ScalarType, i.e.
		*				it is possible to define/declare a numeric variable v with 
		*				the statement:
		*				T::ScalarType v; 		
 		*/
 		//template < typename T >
 		//struct defines_ScalarType{
 		//	static constexpr bool value = false;
 		//	operator bool() { return value; }
 		//};

	}

}

#endif