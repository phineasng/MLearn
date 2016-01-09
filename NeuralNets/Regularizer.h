#ifndef MLEARN_FEEDFORWARD_REGULARIZER_MODE_INCLUDE
#define MLEARN_FEEDFORWARD_REGULARIZER_MODE_INCLUDE

// STL includes
#include <cstdint>
#include <type_traits>

namespace MLearn{

	namespace NeuralNets{

		enum class Regularizer: uint8_t {
			NONE			= 0x00,
			DROPOUT 		= 0x01,
			MAX_NORM 		= 0x02,
			L1 				= 0x04,
			L2 				= 0x08,
			SHARED_WEIGHTS	= 0x10
		};

		typedef typename std::underlying_type<Regularizer>::type RegularizerUnderlyingType;

		constexpr inline Regularizer operator|(Regularizer r1, Regularizer r2){
			return static_cast<Regularizer>(static_cast<RegularizerUnderlyingType>(r1)|static_cast<RegularizerUnderlyingType>(r2));
		}

		inline Regularizer& operator|=(Regularizer& r1, Regularizer r2){
			r1 = static_cast<Regularizer>(static_cast<RegularizerUnderlyingType>(r1)|static_cast<RegularizerUnderlyingType>(r2));
			return r1;
		}

		constexpr inline Regularizer operator&(Regularizer r1, Regularizer r2){
			return static_cast<Regularizer>(static_cast<RegularizerUnderlyingType>(r1)&static_cast<RegularizerUnderlyingType>(r2));
		}

		inline Regularizer& operator&=(Regularizer& r1, Regularizer r2){
			r1 = static_cast<Regularizer>(static_cast<RegularizerUnderlyingType>(r1)&static_cast<RegularizerUnderlyingType>(r2));
			return r1;
		}

		constexpr bool operator==(Regularizer r1, Regularizer r2){
			return static_cast<RegularizerUnderlyingType>(r1) == static_cast<RegularizerUnderlyingType>(r2);
		}

		template< typename ScalarType >
		struct RegularizerOptions{
			static_assert(std::is_floating_point<ScalarType>::value,"The scalar type has to be floating point");
			ScalarType _dropout_prob = 0.5;
			ScalarType _max_norm_limit = 5;
			ScalarType _l1_param = 1e-2;
			ScalarType _l2_param = 1e-2;
		};

	}

}

#endif