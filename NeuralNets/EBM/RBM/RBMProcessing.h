#ifndef MLEARN_RBM_PROCESSING_INCLUDE
#define MLEARN_RBM_PROCESSING_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include "RBMUnitType.h"
#include "RBMUtils.h"
#include <MLearn/NeuralNets/ActivationFunction.h>

// STL includes
#include <type_traits>
#include <cmath>

namespace MLearn{

	namespace NeuralNets{

		namespace RBMSupport{

			/*
			*	\brief 	RBM field parser
			*/
			template < typename RBM, bool PROCESS_VISIBLE = true >
			struct RBMParser{
				typedef decltype(RBM::visible_units) UNIT_TYPE;
				typedef decltype(RBM::bias_visible) BIAS_TYPE;
				typedef decltype(RBM::additional_parameters_visible) ADD_TYPE;
				typedef typename RBM::VISIBLE_DIST DIST_TYPE;
				typedef typename RBM::RNG_TYPE RNG_TYPE;
				static inline UNIT_TYPE& getUnits(RBM& rbm){
					return rbm.visible_units;
				}
				static inline BIAS_TYPE& getBiases(RBM& rbm){
					return rbm.bias_visible;
				}
				static inline ADD_TYPE& getAdditionalParameters(RBM& rbm){
					return rbm.additional_parameters_visible;
				}
				static inline DIST_TYPE& getDistribution(RBM& rbm){
					return rbm.vis_dist;
				}
				static inline RNG_TYPE& getRandomGenerator( RBM& rbm ){
					return rbm.rng;
				}
			};

			template < typename RBM >
			struct RBMParser< RBM, false >{
				typedef decltype(RBM::hidden_units) UNIT_TYPE;
				typedef decltype(RBM::bias_hidden) BIAS_TYPE;
				typedef decltype(RBM::additional_parameters_hidden) ADD_TYPE;
				typedef typename RBM::HIDDEN_DIST DIST_TYPE;
				typedef typename RBM::RNG_TYPE RNG_TYPE;
				static inline UNIT_TYPE& getUnits(RBM& rbm){
					return rbm.hidden_units;
				}
				static inline BIAS_TYPE& getBiases(RBM& rbm){
					return rbm.bias_hidden;
				}
				static inline ADD_TYPE& getAdditionalParameters(RBM& rbm){
					return rbm.additional_parameters_hidden;
				}
				static inline DIST_TYPE& getDistribution(RBM& rbm){
					return rbm.hid_dist;
				}
				static inline RNG_TYPE& getRandomGenerator( RBM& rbm ){
					return rbm.rng;
				}
			};

			/*
			*	\brief 	RBM units preprocessors
			*/	
			template < RBMUnitType TYPE >
			struct RBMUnitPreprocessor{

				template < bool PROCESS_VISIBLE,typename RBM >
				using SCALAR = typename RBMParser<RBM,PROCESS_VISIBLE>::UNIT_TYPE::Scalar;

				template < bool PROCESS_VISIBLE,typename RBM >
				static inline const typename RBMParser<RBM,PROCESS_VISIBLE>::UNIT_TYPE& preprocess(RBM& rbm){
					return RBMParser<RBM,PROCESS_VISIBLE>::getUnits(rbm);
				} 
				template < bool PROCESS_VISIBLE,typename RBM >
				static inline const Eigen::Ref<const MLVector<SCALAR<PROCESS_VISIBLE,RBM>>>& preprocess(RBM& rbm, const Eigen::Ref<const MLVector<SCALAR<PROCESS_VISIBLE,RBM>>>& units ){
					return units;
				} 
			};
			template <>
			struct RBMUnitPreprocessor< RBMUnitType::GAUSSIAN_WITH_VARIANCE >{

				template < bool PROCESS_VISIBLE,typename RBM >
				using SCALAR = typename RBMParser<RBM,PROCESS_VISIBLE>::UNIT_TYPE::Scalar;

				template < bool PROCESS_VISIBLE,typename RBM >
				using FIRST_TYPE = typename RBMParser<RBM,PROCESS_VISIBLE>::UNIT_TYPE;

				template < bool PROCESS_VISIBLE,typename RBM >
				using SECOND_TYPE = Eigen::CwiseUnaryOp< INV_EXPONENTIAL<SCALAR<PROCESS_VISIBLE,RBM>>, const typename RBMParser<RBM,PROCESS_VISIBLE>::ADD_TYPE >;

				template < bool PROCESS_VISIBLE,typename RBM >
				using ReturnType = Eigen::CwiseBinaryOp< MULTIPLICATION<SCALAR<PROCESS_VISIBLE,RBM>>, const FIRST_TYPE<PROCESS_VISIBLE,RBM>, const SECOND_TYPE<PROCESS_VISIBLE,RBM>>;
				
				template < bool PROCESS_VISIBLE,typename RBM >
				using AltReturnType = Eigen::CwiseBinaryOp< MULTIPLICATION<SCALAR<PROCESS_VISIBLE,RBM>>, const Eigen::Ref<const MLVector<SCALAR<PROCESS_VISIBLE,RBM>>>, const SECOND_TYPE<PROCESS_VISIBLE,RBM>>;
				

				template < bool PROCESS_VISIBLE,typename RBM >
				static inline ReturnType<PROCESS_VISIBLE,RBM> preprocess(RBM& rbm){
					const auto& units 	= RBMParser<RBM,PROCESS_VISIBLE>::getUnits(rbm);
					const auto& params 	= RBMParser<RBM,PROCESS_VISIBLE>::getAdditionalParameters(rbm);
					typedef SCALAR<PROCESS_VISIBLE,RBM> SCALAR;
					MLEARN_ASSERT( units.size() == params.size(), "Incompatible vectors!" );
					return units.binaryExpr( params.unaryExpr(INV_EXPONENTIAL<SCALAR>()), MULTIPLICATION<SCALAR>() );
				} 

				template < bool PROCESS_VISIBLE,typename RBM >
				static inline AltReturnType<PROCESS_VISIBLE,RBM> preprocess(RBM& rbm, const Eigen::Ref<const MLVector<SCALAR<PROCESS_VISIBLE,RBM>>>& units){
					const auto& params 	= RBMParser<RBM,PROCESS_VISIBLE>::getAdditionalParameters(rbm);
					typedef SCALAR<PROCESS_VISIBLE,RBM> SCALAR;
					MLEARN_ASSERT( units.size() == params.size(), "Incompatible vectors!" );
					return units.binaryExpr( params.unaryExpr(INV_EXPONENTIAL<SCALAR>()), MULTIPLICATION<SCALAR>() );
				} 
			};

			/*
			*	\brief 	RBM sample processors
			*/
			template < RBMUnitType TYPE >
			struct RBMSampleProcessor{
				template < bool PROCESS_VISIBLE,typename RBM >
				static inline void process(RBM& rbm); 
			};

			// BERNOULLI SPECIALIZATION
			template <>
			struct RBMSampleProcessor< RBMUnitType::BERNOULLI >{
				template < bool PROCESS_VISIBLE,typename RBM >
				static inline void process(RBM& rbm){
					auto& units = RBMParser<RBM,PROCESS_VISIBLE>::getUnits(rbm);
					auto& dist 	= RBMParser<RBM,PROCESS_VISIBLE>::getDistribution(rbm);
					auto& rng 	= RBMParser<RBM,PROCESS_VISIBLE>::getRandomGenerator(rbm);
					typedef typename RBMParser<RBM,PROCESS_VISIBLE>::UNIT_TYPE::Scalar SCALAR;
					typedef typename RBMParser<RBM,PROCESS_VISIBLE>::DIST_TYPE::param_type param_type;
					
					auto unary_expr = [&]( const SCALAR& s ){
						return static_cast<SCALAR>(dist(rng,param_type(ActivationFunction<ActivationType::LOGISTIC>::evaluate(s))));
					};
					units = units.unaryExpr( unary_expr );
				}
			};

			// BINOMIAL SPECIALIZATION
			template <>
			struct RBMSampleProcessor< RBMUnitType::BINOMIAL >{
				template < bool PROCESS_VISIBLE,typename RBM >
				static inline void process(RBM& rbm){
					auto& units = RBMParser<RBM,PROCESS_VISIBLE>::getUnits(rbm);
					auto& dist 	= RBMParser<RBM,PROCESS_VISIBLE>::getDistribution(rbm);
					auto& rng 	= RBMParser<RBM,PROCESS_VISIBLE>::getRandomGenerator(rbm);
					typedef typename RBMParser<RBM,PROCESS_VISIBLE>::UNIT_TYPE::Scalar SCALAR;
					typedef typename RBMParser<RBM,PROCESS_VISIBLE>::DIST_TYPE::param_type param_type;
					auto N = dist.t();
					
					auto unary_expr = [&]( const SCALAR& s ){
						return static_cast<SCALAR>(dist(rng,param_type(N,ActivationFunction<ActivationType::LOGISTIC>::evaluate(s))));
					};
					units = units.unaryExpr( unary_expr );
				}
			};

			// GAUSSIAN SPECIALIZATION
			template <>
			struct RBMSampleProcessor< RBMUnitType::GAUSSIAN >{
				template < bool PROCESS_VISIBLE,typename RBM >
				static inline void process(RBM& rbm){
					auto& units = RBMParser<RBM,PROCESS_VISIBLE>::getUnits(rbm);
					auto& dist 	= RBMParser<RBM,PROCESS_VISIBLE>::getDistribution(rbm);
					auto& rng 	= RBMParser<RBM,PROCESS_VISIBLE>::getRandomGenerator(rbm);
					typedef typename RBMParser<RBM,PROCESS_VISIBLE>::UNIT_TYPE::Scalar SCALAR;
					
					auto unary_expr = [&]( const SCALAR& s ){
						return dist(rng) + s;
					};
					units = units.unaryExpr( unary_expr );
				}
			};

			// GAUSSIAN WITH UNKNOWN VARIANCE SPECIALIZATION
			template <>
			struct RBMSampleProcessor< RBMUnitType::GAUSSIAN_WITH_VARIANCE >{
				template < bool PROCESS_VISIBLE,typename RBM >
				static inline void process(RBM& rbm){
					auto& units = RBMParser<RBM,PROCESS_VISIBLE>::getUnits(rbm);
					auto& adds = RBMParser<RBM,PROCESS_VISIBLE>::getAdditionalParameters(rbm);
					auto& dist 	= RBMParser<RBM,PROCESS_VISIBLE>::getDistribution(rbm);
					auto& rng 	= RBMParser<RBM,PROCESS_VISIBLE>::getRandomGenerator(rbm);
					typedef typename RBM::SCALAR SCALAR;
					
					auto binary_expr = [&]( const SCALAR& s1, const SCALAR& s2 ){
						return dist(rng)*std::sqrt(s2) + s1;
					};
					units = units.binaryExpr( adds.unaryExpr( EXPONENTIAL<SCALAR>() ), binary_expr );
				}
			};
		}
	}
}

#endif