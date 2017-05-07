/*!
* 	\brief 		MLearn library definitions.
*	\details 	In this header file, useful macros are defined
*	\author		phineasng
*					
*/



#ifndef MLEARN_CORE_BASE_MACROS_INCLUDED
#define MLEARN_CORE_BASE_MACROS_INCLUDED

#include <iostream>
#include <exception>

#if defined ( WIN32 )
#define __func__ __FUNCTION__
#endif

#ifndef LOW_ZERO_TOLERANCE
#define LOW_ZERO_TOLERANCE 1e-7
#endif


#ifndef SQRT_2
#define SQRT_2 1.41421356237309504880168872420969807
#endif

#ifndef SQRT_3
#define SQRT_3 1.732050807568877293527446341505
#endif

#ifndef SQRT_5
#define SQRT_5 2.2360679774997896964091736
#endif

#ifndef INV_3
#define INV_3 0.33333333333333333333333333
#endif

// DEBUG MACROS
#ifndef NDEBUG

// ASSERT
#define MLEARN_ASSERT(x,msg)\
	do{\
		if (!(x)){\
			std::cerr << "ERROR in '" << __func__ << "()': " << msg << "\n";\
			std::cerr << "\tAssert failed: " <<  #x << "\n";\
			throw( std::runtime_error( "An error occured. Check your error stream!" ) );\
		}\
	}while(0)

// WARNING
#define MLEARN_WARNING(x,msg)\
	do{\
		if (!(x)){\
			std::cerr << "WARNING in '" << __func__ << "()': " << msg << "\n";\
			std::cerr << "\tCheck failed: " <<  #x << "\n";\
		}\
	}while(0)

// ERROR MESSAGE
#define MLEARN_ERROR_MESSAGE(msg)\
	do{\
		std::cerr << "ERROR in '" << __func__ << "()': " << msg << "\n";\
		throw( std::runtime_error( "An error occured. Check your error stream!" ) );\
	}while(0)

// WARNING MESSAGE
#define MLEARN_WARNING_MESSAGE(msg) do{ std::cerr << "WARNING in '" << __func__ << "()': " << msg << "\n"; }while(0)

#else
#define MLEARN_ASSERT(x,msg)
#define MLEARN_WARNING(x,msg)
#define MLEARN_ERROR_MESSAGE(msg)
#define MLEARN_WARNING_MESSAGE(msg)
#endif // NDEBUG

// FORCED ERROR MESSAGE
#define MLEARN_FORCED_ERROR_MESSAGE(msg)\
	do{\
		std::cerr << "ERROR in '" << __func__ << "()': " << msg << "\n";\
		throw( std::runtime_error( "An error occured. Check your error stream!" ) );\
	}while(0)
// FORCED WARNING MESSAGE
#define MLEARN_FORCED_WARNING_MESSAGE(msg) do{ std::cerr << "WARNING in '" << __func__ << "()': " << msg << "\n"; }while(0)

#endif 