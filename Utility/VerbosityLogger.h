#ifndef MLEARN_VERBOSITY_LOGGER_INCLUDED
#define MLEARN_VERBOSITY_LOGGER_INCLUDED

#include <iostream>
#include <type_traits>

namespace MLearn{

	namespace Utility{
		/*!
		*	\brief 		Simple Logger given verbosity level.
		*	\details	This template utility is used to provide 
		*				an output if VERBOSITY_LEVEL < VERBOSITY_REFERENCE.
		*				The output is directed to the std::clog stream.
		*	\author 	phineasng   
		*/
		template< uint VERBOSITY_LEVEL, uint VERBOSITY_REFERENCE = 0u, typename dummy = void >
		class VerbosityLogger{
		public:

			template < typename T >
			static inline void log(const T& output){}

		};	

		template< uint VERBOSITY_LEVEL, uint VERBOSITY_REFERENCE > 
		class VerbosityLogger< VERBOSITY_LEVEL, VERBOSITY_REFERENCE, typename std::enable_if< (VERBOSITY_LEVEL <= VERBOSITY_REFERENCE), void >::type >{
		public:

			template < typename T >
			static inline void log(const T& output){ std::clog << output ; }

		};

	}

}


#endif