#ifndef MLEARN_RBM_SAMPLER_INCLUDE
#define MLEARN_RBM_SAMPLER_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include <MLearn/NeuralNets/Regularizer.h>
#include "RBMUnitType.h"
#include "RBMProcessing.h"
#include "RBMUtils.h"
#include "RBMCost.h"

// STL includes
#include <random>
#include <chrono>
#include <type_traits>

namespace MLearn{

	namespace NeuralNets{

		namespace RBMSupport{


			/*
			*	\brief 		Sampler class
			*	\author 	phineasng
			*/
			template < 	typename ScalarType,
						RBMUnitType VISIBLE_TYPE,
						RBMUnitType HIDDEN_TYPE >
			class RBMSampler{
			public: // Useful typedefs
				typedef std::mt19937 RNG_TYPE;
				typedef typename Distribution<VISIBLE_TYPE,ScalarType>::type VISIBLE_DIST;
				typedef typename Distribution<HIDDEN_TYPE,ScalarType>::type HIDDEN_DIST;
				typedef RBMSampler<ScalarType,VISIBLE_TYPE,HIDDEN_TYPE> SAMPLER_TYPE;
				typedef ScalarType SCALAR;
				static const RBMUnitType VIS_UNIT_TYPE = VISIBLE_TYPE;
				static const RBMUnitType HID_UNIT_TYPE = HIDDEN_TYPE;
			private: // friend declarations
				
				template< typename SAMPLER, bool VISIBLE >
				friend class RBMParser;

				template< 	Regularizer R,
					  		typename RBMSampler,
					  		RBMTrainingMode MODE,
					  		int N >
				friend class RBMCost;

				template < RBMUnitType V, RBMUnitType H >
				friend class FreeEnergy;

				template < RBMUnitType H >
				friend class HiddenIntegral;

			private: // Routines for parameters redefinition
				inline void redefineAdditionalParameters(const Eigen::Ref< const MLVector<ScalarType> >& parameters, std::true_type T1, std::true_type T2 ){
					auto offset = visible_units.size()*hidden_units.size() + visible_units.size() + hidden_units.size();
					auto n_params_vis = visible_units.size()*RBMUnitTypeTraits<VISIBLE_TYPE>::N_params;
					auto n_params_hid = hidden_units.size()*RBMUnitTypeTraits<HIDDEN_TYPE>::N_params;
					new (&additional_parameters_hidden) Eigen::Map< const MLVector<ScalarType> >(parameters.data()+offset,n_params_hid);
					new (&additional_parameters_visible) Eigen::Map< const MLVector<ScalarType> >(parameters.data()+offset+n_params_hid,n_params_vis);
				}
				inline void redefineAdditionalParameters(const Eigen::Ref< const MLVector<ScalarType> >& parameters, std::false_type F, std::true_type T ){
					auto offset = visible_units.size()*hidden_units.size() + visible_units.size() + hidden_units.size();
					auto n_params = hidden_units.size()*RBMUnitTypeTraits<HIDDEN_TYPE>::N_params;
					new (&additional_parameters_hidden) Eigen::Map< const MLVector<ScalarType> >(parameters.data()+offset,n_params);
				}
				inline void redefineAdditionalParameters(const Eigen::Ref< const MLVector<ScalarType> >& parameters, std::true_type T, std::false_type F ){
					auto offset = visible_units.size()*hidden_units.size() + visible_units.size() + hidden_units.size();
					auto n_params = visible_units.size()*RBMUnitTypeTraits<VISIBLE_TYPE>::N_params;
					new (&additional_parameters_visible) Eigen::Map< const MLVector<ScalarType> >(parameters.data()+offset,n_params);
				}
				inline void redefineAdditionalParameters(const Eigen::Ref< const MLVector<ScalarType> >& parameters, std::false_type F1, std::false_type F2 ){}
			public:
				// CONSTRUCTOR
				template < typename UINT >
				RBMSampler( UINT vis_size, UINT hid_size ):
					rng(std::chrono::system_clock::now().time_since_epoch().count()),
					visible_units( vis_size ),
					hidden_units( hid_size ),
					weights( NULL, 0, 0 ),
					bias_visible( NULL, 0 ),
					bias_hidden( NULL, 0 ),
					additional_parameters_visible( NULL, 0 ),
					additional_parameters_hidden( NULL, 0 )
				{
					static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, " Initialization parameters have to be unsigned integers!" );
				}
				template < typename UINT >
				RBMSampler( UINT vis_size, UINT hid_size, const Eigen::Ref< const MLVector<ScalarType> >& parameters):
					rng(std::chrono::system_clock::now().time_since_epoch().count()),
					visible_units( vis_size ),
					hidden_units( hid_size ),
					weights( NULL, 0, 0 ),
					bias_visible( NULL, 0 ),
					bias_hidden( NULL, 0 ),
					additional_parameters_visible( NULL, 0 ),
					additional_parameters_hidden( NULL, 0 )
				{
					static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, " Initialization parameters have to be unsigned integers!" );
					attachParameters(parameters);
				}
				// MODIFIERS
				// attach new parameters
				void attachParameters( const Eigen::Ref< const MLVector<ScalarType> >& parameters ){
					MLEARN_WARNING_MESSAGE("Attaching new parameters: be sure the input vector is not a temporary! (compile with -DNDEBUG to disable this warning)");
					auto N_vis = visible_units.size();
					auto N_hid = hidden_units.size();
					auto offset = N_vis*N_hid;
					MLEARN_ASSERT( parameters.size() == (offset+N_vis+N_hid+N_vis*RBMUnitTypeTraits<VISIBLE_TYPE>::N_params+N_hid*RBMUnitTypeTraits<HIDDEN_TYPE>::N_params), "Wrong number of parameters!" );
					new (&weights) Eigen::Map< const MLMatrix< ScalarType > >( parameters.data(), N_hid, N_vis );
					new (&bias_hidden) Eigen::Map< const MLVector< ScalarType > >( parameters.data() + offset, N_hid );
					new (&bias_visible) Eigen::Map< const MLVector< ScalarType > >( parameters.data() + offset + N_hid, N_vis );
					redefineAdditionalParameters(parameters,VIS_PARAM_TRUTH_TYPE, HID_PARAM_TRUTH_TYPE);

				}
				// resize and attach
				template < typename UINT >
				void resizeAndAttachParameters( UINT vis_size, UINT hid_size, const Eigen::Ref< const MLVector<ScalarType> >& parameters  ){
					static_assert( std::is_integral<UINT>::value && std::is_unsigned<UINT>::value, " Initialization parameters have to be unsigned integers!" );
					visible_units.resize(vis_size);
					hidden_units.resize(hid_size);
					attachParameters(parameters);
				}
				template < typename... ARGS >
				void setVisibleDistributionParameters(ARGS... args){
					static_assert( RBMUnitTypeTraits<VISIBLE_TYPE>::fixed_params, "The distribution does not require parameters to be set!" );
					static_assert( sizeof...(ARGS) == RBMUnitTypeTraits<VISIBLE_TYPE>::N_fixed_params, "Input arguments must be the as many as the settable ones!" );
					DistributionSetter< VISIBLE_TYPE, ScalarType >::set(vis_dist,args...);
				}
				template < typename... ARGS >
				void setHiddenDistributionParameters(ARGS... args){
					static_assert( RBMUnitTypeTraits<HIDDEN_TYPE>::fixed_params, "The distribution does not require parameters to be set!" );
					static_assert( sizeof...(ARGS) == RBMUnitTypeTraits<HIDDEN_TYPE>::N_fixed_params, "Input arguments must be the as many as the settable ones!" );
					DistributionSetter< HIDDEN_TYPE, ScalarType >::set(hid_dist,args...);
				}
				// OBSERVERS
				const MLVector<ScalarType>& getVisibleUnits() const{
					return visible_units;
				}
				const MLVector<ScalarType>& getHiddenUnits() const{
					return hidden_units;
				}
				// SAMPLING
				void sampleHFromV(){
					hidden_units = weights*RBMUnitPreprocessor<VISIBLE_TYPE>::template preprocess<true>(*this) + bias_hidden;
					RBMSampleProcessor<HIDDEN_TYPE>::template process<false>(*this);
				}
				void sampleVFromH(){
					visible_units = weights.transpose()*RBMUnitPreprocessor<HIDDEN_TYPE>::template preprocess<false>(*this) + bias_visible;
					RBMSampleProcessor<VISIBLE_TYPE>::template process<true>(*this);
				}
				void sampleHFromV(const Eigen::Ref< const MLVector<ScalarType> >& starting_v){
					hidden_units = weights*RBMUnitPreprocessor<VISIBLE_TYPE>::template preprocess<true>(*this,starting_v) + bias_hidden;
					RBMSampleProcessor<HIDDEN_TYPE>::template process<false>(*this);
				}
				void sampleVFromH(const Eigen::Ref< const MLVector<ScalarType> >& starting_h){
					visible_units = weights.transpose()*RBMUnitPreprocessor<HIDDEN_TYPE>::template preprocess<false>(*this,starting_h) + bias_visible;
					RBMSampleProcessor<VISIBLE_TYPE>::template process<true>(*this);
				}
			private:
				mutable RNG_TYPE rng;
				mutable VISIBLE_DIST vis_dist;
				mutable HIDDEN_DIST hid_dist;
				static const typename std::integral_constant< bool, RBMUnitTypeTraits<VISIBLE_TYPE>::learn_params >::type VIS_PARAM_TRUTH_TYPE;
				static const typename std::integral_constant< bool, RBMUnitTypeTraits<HIDDEN_TYPE>::learn_params >::type HID_PARAM_TRUTH_TYPE; 
				MLVector< ScalarType > visible_units, hidden_units;
				Eigen::Map< const MLMatrix< ScalarType > > weights;
				Eigen::Map< const MLVector< ScalarType > > bias_visible;
				Eigen::Map< const MLVector< ScalarType > > bias_hidden;
				Eigen::Map< const MLVector< ScalarType > > additional_parameters_visible;
				Eigen::Map< const MLVector< ScalarType > > additional_parameters_hidden;
			};

		}

	}

}


#endif