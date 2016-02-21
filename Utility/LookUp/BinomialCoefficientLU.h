#ifndef MLEARN_BINOMIAL_COEFFICIENT_LOOKUP_INCLUDE
#define MLEARN_BINOMIAL_COEFFICIENT_LOOKUP_INCLUDE

// MLearn includes
#include <MLearn/Core>

// STL includes
#include <unordered_map>
#include <type_traits>
#include <cmath>

namespace MLearn{

	namespace Utility{

		namespace LookUp{

			/*
			*	\brief 		LookUp Table for binomial coefficient
			*	\details	NOTE: This lookup table is generated at runtime.
			*				Given N, this class return a MLVector of size N + 1 with all the coefficients C(N,h) for h = 0...N.
			*				If the coefficients for N are not found, they are generated.
			*				It is advised to use this class if the binomial coefficients, for a given N, are required many times.
			*/
			template < typename NUMTYPE, typename KEYTYPE = uint >
			class BinomialCoefficientLU{
				static_assert( std::is_arithmetic<NUMTYPE>::value, "A numeric type is required for the lookup table" );
				static_assert( std::is_integral<KEYTYPE>::value && std::is_unsigned<KEYTYPE>::value, "An unsigned integral is required as key type!" );
			public:
				typedef KEYTYPE 						Key;
				typedef MLVector<NUMTYPE> 				Type;
				typedef std::unordered_map< Key, Type >	TableType;
			public:
				static const Type& getBinomialCoefficients( const Key& N ){
					typename TableType::const_iterator it = binCoeffTable.find(N);
					if ( it == binCoeffTable.end() ){
						return buildBinCoeffs(N);
					}else{
						return (*it).second;
					}
				} 
			protected:
				// make constructor private
				BinomialCoefficientLU(){}
			private:
				static TableType binCoeffTable;
				// Routine to build the bin coeffs
				static const Type& buildBinCoeffs( const Key& N ){
					Type& binCoeffs = binCoeffTable[N] = Type(N+1);
					binCoeffs[0] = 1;
					binCoeffs[N] = 1;
					Key M = Key( std::ceil(float(N)*0.5f) );
					Key i = 1;
					for ( ; i <= M; ++i ){
						binCoeffs[i] = (binCoeffs[i-1]*(N+1-i))/i;
					}
					for ( ; i < N; ++i ){
						binCoeffs[i] = binCoeffs[N-i];
					}
					return binCoeffs;
				}
			};

			template < typename T1, typename T2 >
			typename BinomialCoefficientLU<T1,T2>::TableType BinomialCoefficientLU<T1,T2>::binCoeffTable = typename BinomialCoefficientLU<T1,T2>::TableType();

		};

	}

}


#endif