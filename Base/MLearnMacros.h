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