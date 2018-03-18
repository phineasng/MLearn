#ifndef MLEARN_RBM_CLASS_INCLUDE
#define MLEARN_RBM_CLASS_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include "RBMUnitType.h"
#include "RBMUtils.h"
#include "RBMSampler.h"
#include "RBMCost.h"

// STL includes
#include <type_traits>

namespace MLearn{

	namespace NeuralNets{

		/*!
		*	\brief	RBM class
		*/
		template < 	typename ScalarType,
					RBMSupport::RBMUnitType VISIBLE_TYPE, 
					RBMSupport::RBMUnitType HIDDEN_TYPE>
		class RBM{
		private:
			typedef RBMSupport::RBMSampler< ScalarType,VISIBLE_TYPE,HIDDEN_TYPE > SAMPLER_TYPE;
		public:
			// CONSTRUCTORS
			RBM() = delete;
			template < typename UINT >
			RBM( UINT N_vis, UINT N_hid ):
				parameters( computeNParameters<UINT>(N_vis,N_hid) ),
				sampler( N_vis, N_hid, parameters ),
				gradient_tmp( parameters.size() )
			{
				static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Input values for size should be unsigned integers!" );
			}
			// MODIFIERS
			void setParameters( const Eigen::Ref< MLVector<ScalarType> >& new_parameters ){
				MLEARN_ASSERT( new_parameters.size() == parameters.size(),"Parameters size incompatible! To allow for a size change use the appropriate overload of this function!" );
				parameters = new_parameters;
			}
			template < typename UINT >
			void setParametersAndResize( UINT N_vis, UINT N_hid, const Eigen::Ref< MLVector<ScalarType> >& new_parameters ){
				static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, "Input values for size should be unsigned integers!" );
				MLEARN_ASSERT( new_parameters.size() == computeNParameters<UINT>(N_vis,N_hid),"Input parameters not consistent (in size) with the given size values!" );
				parameters = new_parameters;
				sampler.resizeAndAttachParameters(N_vis,N_hid,parameters);
			}
			template < typename... ARGS >
			void setVisibleDistributionParameters(ARGS... args){
				sampler.setVisibleDistributionParameters(args...);	
			}
			template < typename... ARGS >
			void setHiddenDistributionParameters(ARGS... args){
				sampler.setHiddenDistributionParameters(args...);
			}
			// OBSERVERS
			const MLVector< ScalarType >& getParameters() const{
				return parameters;
			}
			// SAMPLING
			MLVector< ScalarType > sampleRBM( size_t N = 1000 ){
				for (size_t i = 0; i < N; ++i){
					sampler.sampleHFromV();
					sampler.sampleVFromH();
				}
				return sampler.getVisibleUnits();
			}
			MLVector< ScalarType > sampleRBMFromV( const Eigen::Ref<MLVector<ScalarType>>& starting_v, size_t N = 1000 ){
				MLEARN_ASSERT( starting_v.size() == sampler.getVisibleUnits().size(), "Wrong size of starting visible vector!" );
				sampler.sampleHFromV(starting_v);
				sampler.sampleVFromH();
				for (size_t i = 1; i < N; ++i){
					sampler.sampleHFromV();
					sampler.sampleVFromH();
				}
				return sampler.getVisibleUnits();
			}
			// TRAINING
			template < 	RBMSupport::RBMTrainingMode MODE,
						int N_chain,
						Regularizer REG,
						typename MINIMIZER >
			void train( const Eigen::Ref< MLMatrix<ScalarType> >& input, MINIMIZER& minimizer, const RegularizerOptions< ScalarType >& reg_options = RegularizerOptions< ScalarType >() ){

				// create cost function
				RBMSupport::RBMCost< REG, SAMPLER_TYPE, MODE, N_chain > cost( sampler, input, reg_options, gradient_tmp );

				minimizer.minimize(cost,parameters);

			}
		private:
			MLVector< ScalarType > parameters;
			SAMPLER_TYPE sampler;
			// preallocations
			MLVector< ScalarType > gradient_tmp;
			// support functions
			template < typename UINT >
			static UINT computeNParameters(UINT N_vis, UINT N_hid){
				return N_vis*N_hid + ( 1 + RBMSupport::RBMUnitTypeTraits<VISIBLE_TYPE>::N_params )*N_vis + ( 1 + RBMSupport::RBMUnitTypeTraits<HIDDEN_TYPE>::N_params )*N_hid;
			}
		};

	}

}


#endif