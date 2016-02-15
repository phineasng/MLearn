#ifndef MLEARN_RBM_UNIT_TYPE_INCLUDE
#define MLEARN_RBM_UNIT_TYPE_INCLUDE

namespace MLearn{

	namespace NeuralNets{

		namespace RBMSupport{

			enum class RBMUnitType{
				BERNOULLI,
				BINOMIAL,
				GAUSSIAN,
				GAUSSIAN_WITH_VARIANCE
			};

			template < RBMUnitType TYPE >
			struct RBMUnitTypeTraits{
				static constexpr ushort N_params = 0;
				static constexpr ushort N_fixed_params = 0;
				static constexpr bool learn_params = false;
				static constexpr bool fixed_params = false;
			};

			template <>
			struct RBMUnitTypeTraits<RBMUnitType::GAUSSIAN_WITH_VARIANCE>
			{
				static constexpr ushort N_params = 1;
				static constexpr ushort N_fixed_params = 0;
				static constexpr bool learn_params = true;
				static constexpr bool fixed_params = false;
			};

			template <>
			struct RBMUnitTypeTraits<RBMUnitType::BINOMIAL>
			{
				static constexpr ushort N_params = 0;
				static constexpr ushort N_fixed_params = 1;
				static constexpr bool learn_params = false;
				static constexpr bool fixed_params = true;
			};

		}

	}

}


#endif