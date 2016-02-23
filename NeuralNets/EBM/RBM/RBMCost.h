#ifndef MLEARN_RBM_COST_INCLUDE
#define MLEARN_RBM_COST_INCLUDE

// MLearn includes
#include <MLearn/Core>
#include <MLearn/NeuralNets/Regularizer.h>
#include <MLearn/Optimization/CostFunction.h>
#include <MLearn/NeuralNets/ActivationFunction.h>
#include <MLearn/Utility/LookUp/BinomialCoefficientLU.h>
#include <MLearn/Utility/MemoryPool/MLVectorPool.h>
#include "RBMUtils.h"
#include "RBMProcessing.h"

// STL includes
#include <cmath>

namespace MLearn{

	namespace NeuralNets{

		namespace RBMSupport{

			/*
			*	\brief Hidden distribution integral computer
			*/
			template < RBMUnitType HID_TYPE >
			class HiddenIntegral{
			public:
				template < typename RBM, typename DERIVED >
				static inline typename RBM::SCALAR compute_integral( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBM& rbm ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBM::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					typename RBM::SCALAR integral = 0;
					auto& bias_hidden = rbm.bias_hidden;
					auto binary_op = [](const typename RBM::SCALAR& s1, const typename RBM::SCALAR& s2){
						typename RBM::SCALAR sum = s1 + s2; 
						if (sum > 0)
							return static_cast<typename RBM::SCALAR>( std::log(1+std::exp(sum)) );
						else 
							return static_cast<typename RBM::SCALAR>( std::log(1+std::exp(-sum)) + sum );
					};

					integral = -(interaction_terms.binaryExpr(bias_hidden,binary_op).sum());

					return integral;
				}
				template < typename RBMCOST >
				using DerReturnType = decltype(RBMCOST::bias_hidden);
				template < typename RBMCOST, typename DERIVED >
				static inline DerReturnType<RBMCOST> compute_hidden_derivative( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBMCOST& cost ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBMCOST::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					auto& rbm = cost.rbm;
					auto& bias_hidden = rbm.bias_hidden;
					cost.bias_hidden = -((interaction_terms+bias_hidden).unaryExpr(std::pointer_to_unary_function<typename RBMCOST::SCALAR,typename RBMCOST::SCALAR>(ActivationFunction<ActivationType::LOGISTIC>::evaluate)));
					return cost.bias_hidden;
				}

			};
			// BINOMIAL SPECIALIZATION
			template <>
			class HiddenIntegral<RBMUnitType::BINOMIAL>{
			public:
				template < typename RBM, typename DERIVED >
				static inline typename RBM::SCALAR compute_integral( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBM& rbm ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBM::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					typename RBM::SCALAR integral = 0;
					auto& bias_hidden = rbm.bias_hidden;
					typename RBM::SCALAR N = rbm.hid_dist.t();
					typedef Utility::LookUp::BinomialCoefficientLU< typename RBM::SCALAR > LookUpTable;
					const auto& binCoeffs = LookUpTable::getBinomialCoefficients(N);
					auto unary_op = [&N,&binCoeffs](typename RBM::SCALAR s){
						typename RBM::SCALAR total = 0;
						typename RBM::SCALAR subtotal = 0;
						if ( s > 0 ){
							total = N*s;
							s = -s;
						}
						for (size_t i = 0; i <= size_t(N); ++i){
							subtotal += binCoeffs[i]*std::exp(static_cast<typename RBM::SCALAR>(i)*s);
						}
						total = total + std::log(subtotal);
						return total;
					};

					integral = -((interaction_terms+bias_hidden).unaryExpr(unary_op)).sum();

					return integral;
				}
				template < typename RBMCOST >
				using DerReturnType = decltype(RBMCOST::bias_hidden);
				template < typename RBMCOST, typename DERIVED >
				static inline DerReturnType<RBMCOST> compute_hidden_derivative( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBMCOST& cost ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBMCOST::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					auto& rbm = cost.rbm;
					auto& bias_hidden = rbm.bias_hidden;
					typename RBMCOST::SCALAR N = rbm.hid_dist.t();
					typedef Utility::LookUp::BinomialCoefficientLU< typename RBMCOST::SCALAR > LookUpTable;
					const auto& binCoeffs = LookUpTable::getBinomialCoefficients(N);
					auto unary_op = [&N,&binCoeffs](typename RBMCOST::SCALAR s){
						typename RBMCOST::SCALAR total_num = 0;
						typename RBMCOST::SCALAR total_den = 0;
						typename RBMCOST::SCALAR temp;
						if ( s > 0 ){
							s = -s;
							for (size_t i = 0; i <= size_t(N); ++i){
								temp = binCoeffs[i]*std::exp(static_cast<typename RBMCOST::SCALAR>(N-i)*s);
								total_den += temp;
								total_num += static_cast<typename RBMCOST::SCALAR>(i)*temp;
							}
						}else{
							for (size_t i = 0; i <= size_t(N); ++i){
								temp = binCoeffs[i]*std::exp(static_cast<typename RBMCOST::SCALAR>(i)*s);
								total_den += temp;
								total_num += static_cast<typename RBMCOST::SCALAR>(i)*temp;
							}
						}
						return total_num/total_den;
					};
					cost.bias_hidden = -(interaction_terms+bias_hidden).unaryExpr(unary_op);
					return cost.bias_hidden;
				}

			};

			// GAUSSIAN SPECIALIZATION
			template <>
			class HiddenIntegral<RBMUnitType::GAUSSIAN>{
			public:
				template < typename RBM, typename DERIVED >
				static inline typename RBM::SCALAR compute_integral( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBM& rbm ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBM::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					typename RBM::SCALAR integral = 0;
					auto& bias_hidden = rbm.bias_hidden;
					integral -= interaction_terms.dot( bias_hidden + 0.5*interaction_terms );

					return integral;
				}
				template < typename RBMCOST, typename DERIVED >
				using DerReturnType = Eigen::CwiseBinaryOp< NEGATIVE_SUM<typename RBMCOST::SCALAR>, const decltype(RBMCOST::RBMTYPE::bias_hidden), const DERIVED>;
				template < typename RBMCOST, typename DERIVED >
				static inline DerReturnType<RBMCOST,DERIVED> compute_hidden_derivative( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBMCOST& cost ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBMCOST::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					auto& rbm = cost.rbm;
					cost.bias_hidden = -interaction_terms;
					return rbm.bias_hidden.binaryExpr(interaction_terms,NEGATIVE_SUM<typename RBMCOST::SCALAR>());
				}

			};
			// GAUSSIAN WITH UNKNOWN VARIANCE SPECIALIZATION
			template <>
			class HiddenIntegral<RBMUnitType::GAUSSIAN_WITH_VARIANCE>{
			public:
				template < typename RBM, typename DERIVED >
				static inline typename RBM::SCALAR compute_integral( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBM& rbm ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBM::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					typename RBM::SCALAR integral = 0;
					auto& bias_hidden = rbm.bias_hidden;
					auto& additional_parameters_hidden = rbm.additional_parameters_hidden;
					integral -= additional_parameters_hidden.sum()*0.5 + ((-additional_parameters_hidden).array().exp()*interaction_terms.array()*( bias_hidden.array() + 0.5*interaction_terms.array() )).sum();

					return integral;
				}
				template < typename RBMCOST, typename DERIVED >
				using InnerType = Eigen::CwiseBinaryOp< SUM<typename RBMCOST::SCALAR>, const decltype(RBMCOST::RBMTYPE::bias_hidden), const DERIVED>;
				template < typename RBMCOST, typename DERIVED >
				using DerReturnType = Eigen::CwiseBinaryOp< MULTIPLICATION<typename RBMCOST::SCALAR>, const MLVector<typename RBMCOST::SCALAR>, const InnerType<RBMCOST,DERIVED>>;
				template < typename RBMCOST, typename DERIVED >
				static inline MLVector<typename RBMCOST::SCALAR> compute_hidden_derivative( const Eigen::MatrixBase<DERIVED>& interaction_terms, RBMCOST& cost ){
					static_assert( std::is_same<typename DERIVED::Scalar,typename RBMCOST::SCALAR>::value, "Scalar types must be the same" );
					static_assert( DERIVED::ColsAtCompileTime == 1, "Expected column vector!" );
					auto& rbm = cost.rbm;
					auto& additional_parameters_hidden = rbm.additional_parameters_hidden;
					auto interface = Utility::MemoryPool::MLVectorPool<typename RBMCOST::SCALAR>::get(additional_parameters_hidden.size());
					auto& sigmas = interface.getReference();
					sigmas = - (additional_parameters_hidden).unaryExpr(INV_EXPONENTIAL<typename RBMCOST::SCALAR>());
					cost.bias_hidden = sigmas.cwiseProduct(interaction_terms);
					cost.additional_parameters_hidden.array() = - 0.5 - cost.bias_hidden.cwiseProduct(0.5*interaction_terms + rbm.bias_hidden).array();
					sigmas.array() = sigmas.array()*(rbm.bias_hidden.array() + interaction_terms.array());
					return sigmas;
				}

			};

			/*
			*	\brief	RBM free energy. 
			*/
			template < RBMUnitType VIS_TYPE,RBMUnitType HID_TYPE >
			class FreeEnergy{
			public:
				template< typename RBMCOST, typename DERIVED >
				static inline typename RBMCOST::SCALAR compute_energy( const Eigen::MatrixBase<DERIVED>& input, const RBMCOST& cost){
					typename RBMCOST::SCALAR energy = 0;
					auto& rbm = cost.rbm;
					auto& visible_bias = rbm.bias_visible;
					auto& weights = rbm.weights;

					energy -= input.dot(visible_bias);
					energy += HiddenIntegral< HID_TYPE >::compute_integral( weights*input, rbm );

					return energy;
				}
				template< typename RBMCOST, typename DERIVED >
				static inline void compute_gradient( const Eigen::MatrixBase<DERIVED>& input, RBMCOST& cost){
					auto& rbm = cost.rbm;
					cost.bias_visible = -input;
					auto& weights = rbm.weights;
					cost.weights = HiddenIntegral< HID_TYPE >::compute_hidden_derivative( weights*input, cost )*input.transpose();

					return;
				}
			};
			// GAUSSIAN SPECIALIZATION
			template < RBMUnitType HID_TYPE >
			class FreeEnergy< RBMUnitType::GAUSSIAN, HID_TYPE >{
			public:
				template< typename RBMCOST, typename DERIVED >
				static inline typename RBMCOST::SCALAR compute_energy( const Eigen::MatrixBase<DERIVED>& input, const RBMCOST& cost){
					typename RBMCOST::SCALAR energy = 0;
					auto& rbm = cost.rbm;
					auto& visible_bias = rbm.bias_visible;
					auto& weights = rbm.weights;

					energy += 0.5*(input - visible_bias).squaredNorm();
					energy += HiddenIntegral< HID_TYPE >::compute_integral( weights*input, rbm );

					return energy;
				}
				template< typename RBMCOST, typename DERIVED >
				static inline void compute_gradient( const Eigen::MatrixBase<DERIVED>& input, RBMCOST& cost){
					auto& rbm = cost.rbm;
					auto& visible_bias = rbm.bias_visible;
					cost.bias_visible = visible_bias-input;
					auto& weights = rbm.weights;
					cost.weights = HiddenIntegral< HID_TYPE >::compute_hidden_derivative( weights*input, cost )*input.transpose();

					return;
				}
			};
			// GAUSSIAN WITH UNKNOWN VARIANCE SPECIALIZATION
			template < RBMUnitType HID_TYPE >
			class FreeEnergy< RBMUnitType::GAUSSIAN_WITH_VARIANCE, HID_TYPE >{
			public:
				template< typename RBMCOST, typename DERIVED >
				static inline typename RBMCOST::SCALAR compute_energy( const Eigen::MatrixBase<DERIVED>& input, const RBMCOST& cost){
					typename RBMCOST::SCALAR energy = 0;
					auto& rbm = cost.rbm;
					auto& visible_bias = rbm.bias_visible;
					auto& weights = rbm.weights;
					auto& additional_parameters_visible = rbm.additional_parameters_visible;
					auto interface = Utility::MemoryPool::MLVectorPool<typename RBMCOST::SCALAR>::get(additional_parameters_visible.size());
					auto& processed_add = interface.getReference();
					processed_add = additional_parameters_visible.unaryExpr(INV_EXPONENTIAL<typename RBMCOST::SCALAR>());
					energy += 0.5*(input - visible_bias).cwiseAbs2().dot(processed_add);
					energy += HiddenIntegral< HID_TYPE >::compute_integral( weights*input.binaryExpr(processed_add,MULTIPLICATION<typename RBMCOST::SCALAR>()), rbm );

					return energy;
				}
				template< typename RBMCOST, typename DERIVED >
				static inline void compute_gradient( const Eigen::MatrixBase<DERIVED>& input, RBMCOST& cost){
					auto& rbm = cost.rbm;
					auto& visible_bias = rbm.bias_visible;
					auto& additional_parameters_visible = rbm.additional_parameters_visible;
					MULTIPLICATION<typename RBMCOST::SCALAR> mul_expr;
					auto interface = Utility::MemoryPool::MLVectorPool<typename RBMCOST::SCALAR>::get(additional_parameters_visible.size());
					auto& processed_add = interface.getReference();
					processed_add = additional_parameters_visible.unaryExpr(INV_EXPONENTIAL<typename RBMCOST::SCALAR>());
					
					cost.bias_visible = input.binaryExpr(processed_add,mul_expr);

					auto& weights = rbm.weights;
					const auto& hidden_derivative = HiddenIntegral< HID_TYPE >::compute_hidden_derivative( weights*cost.bias_visible, cost );
					cost.weights = hidden_derivative*cost.bias_visible.transpose();
					cost.additional_parameters_visible = (rbm.weights.transpose()*hidden_derivative).binaryExpr(input,MULTIPLICATION<typename RBMCOST::SCALAR>()) + (input - visible_bias).cwiseAbs2();
					cost.additional_parameters_visible = -cost.additional_parameters_visible.binaryExpr(processed_add,mul_expr);
					cost.additional_parameters_visible = cost.additional_parameters_visible.binaryExpr(additional_parameters_visible,mul_expr);
					cost.bias_visible = cost.bias_visible.binaryExpr(processed_add,mul_expr) - cost.bias_visible;

					return;
				}
			};

			/*
			*
			*/

			/*
			*	\brief	RBM cost function to be minimized
			*	\author phineasng
			*/
			template< Regularizer R,
					  typename RBMSampler,
					  RBMTrainingMode MODE,
					  int N_chain = 1 >
			class RBMCost: public Optimization::CostFunction< RBMCost<R,RBMSampler,MODE,N_chain> >{
			public:	// useful typedefs 
				typedef typename RBMSampler::SCALAR SCALAR;
				static const RBMUnitType HID_TYPE = RBMSampler::HID_UNIT_TYPE;
				static const RBMUnitType VIS_TYPE = RBMSampler::VIS_UNIT_TYPE;
				typedef RBMSampler RBMTYPE;
			public:
				// CONSTRUCTOR
				RBMCost(RBMSampler& refRbm, 
						const Eigen::Ref< MLMatrix<SCALAR> > refInputs,
						const RegularizerOptions<SCALAR>& refOptions,
						MLVector< SCALAR >& ref_allocated_grad ):
					rbm(refRbm),
					inputs(refInputs),
					options(refOptions),
					gradient_tmp(ref_allocated_grad),
					weights( NULL, 0, 0 ),
					bias_visible( NULL, 0 ),
					bias_hidden( NULL, 0 ),
					additional_parameters_visible( NULL, 0 ),
					additional_parameters_hidden( NULL, 0 )
				{
					mapVariables();
				}
				// MODIFIERS
				void remap(){
					mapVariables();
				}
				void setChainInitializationFlag(bool flag){
					chain_initialized = flag;
				}
				// OBSERVERS
				bool getChainInitializationFlag(){
					return chain_initialized;
				}
				// EVALUATE COST FUNCTION
				/*
				*	\brief This is actually the free energy and not a cost related to the "gradient"
				*/
				template < 	typename DERIVED >
				typename DERIVED::Scalar eval( const Eigen::MatrixBase<DERIVED>& x ) const{
					static_assert(std::is_same<SCALAR,typename DERIVED::Scalar>::value, "Scalar types have to be the same!");
					static_assert(DERIVED::ColsAtCompileTime == 1, "Input has to be a column vector (or compatible structure)!");
					MLEARN_ASSERT( x.size() == gradient_tmp.size(), "Dimension not valid!" );
					SCALAR loss = SCALAR(0);
					rbm.attachParameters(x);

					for ( decltype(inputs.cols()) idx = 0; idx < inputs.cols(); ++idx ){
						loss += FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_energy(inputs.col(idx),*this);
					}

					loss /= SCALAR(inputs.cols());

					// L2 reg
					loss += L2Regularization(x,typename std::integral_constant< bool, (Regularizer::L2 & R) == Regularizer::L2 >::type() );
					loss += L1Regularization(x,typename std::integral_constant< bool, (Regularizer::L1 & R) == Regularizer::L1 >::type());

					return loss;
				}
				template< 	typename DERIVED,
					   		typename DERIVED_2 >
				void compute_analytical_gradient(  const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient ) const{
					static_assert( std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types must be the same" );
					static_assert( std::is_same<typename DERIVED::Scalar,SCALAR>::value, "Scalar types must be the same" );
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Expected column vectors!" );
					MLEARN_ASSERT( x.size() == gradient_tmp.size(), "Dimension not valid!" );
					rbm.attachParameters(x);
					gradient = MLVector<SCALAR>::Zero(x.size());

					for ( decltype(inputs.cols()) idx = 0; idx < inputs.cols(); ++idx ){
						sample_for_gradient(idx);
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(inputs.col(idx),*this);
						gradient += gradient_tmp;
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(rbm.visible_units,*this);
						gradient -= gradient_tmp;
					}

					gradient /= SCALAR(inputs.cols());

					L2Regularization(x,gradient,typename std::integral_constant< bool, (Regularizer::L2 & R) == Regularizer::L2 >::type());
					L1Regularization(x,gradient,typename std::integral_constant< bool, (Regularizer::L1 & R) == Regularizer::L1 >::type());

					return;
				}
				template< 	typename IndexType,
							typename DERIVED,
				   			typename DERIVED_2 >
				void compute_stochastic_gradient( const Eigen::MatrixBase<DERIVED>& x, Eigen::MatrixBase<DERIVED_2>& gradient, const MLVector< IndexType >& idx ) const{
					static_assert( std::is_same<typename DERIVED::Scalar,typename DERIVED_2::Scalar>::value, "Scalar types must be the same" );
					static_assert( std::is_same<typename DERIVED::Scalar,SCALAR>::value, "Scalar types must be the same" );
					static_assert( (DERIVED::ColsAtCompileTime == 1) && (DERIVED_2::ColsAtCompileTime == 1), "Expected column vectors!" );
					MLEARN_ASSERT( x.size() == gradient_tmp.size(), "Dimension not valid!" );
					rbm.attachParameters(x);
					gradient = MLVector<SCALAR>::Zero(x.size());

					for ( decltype(idx.size()) i = 0; i < idx.size(); ++i ){
						sample_for_gradient(idx[i]);
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(inputs.col(idx[i]),*this);
						gradient += gradient_tmp;
						FreeEnergy<VIS_TYPE,HID_TYPE>::template compute_gradient(rbm.visible_units,*this);
						gradient -= gradient_tmp;
					}

					gradient /= SCALAR(idx.size());

					L2Regularization(x,gradient,typename std::integral_constant< bool, (Regularizer::L2 & R) == Regularizer::L2 >::type());
					L1Regularization(x,gradient,typename std::integral_constant< bool, (Regularizer::L1 & R) == Regularizer::L1 >::type());

					return;
				}
			private:
				RBMSampler& rbm;
				const Eigen::Ref< MLMatrix<SCALAR> > inputs;
				const RegularizerOptions<SCALAR>& options;
				/// \details For safety (in referencing the needed objects) and more flexibility (using the eigen types),
				/// we plan to reconstruct this cost function class everytime a training algorithm is called.
				/// This means that, in an ONLINE or CROSSVALIDATION setting, the construction may incur in
				/// a not negligible overhead if this class allocates memory or has to build objects which allocates
				/// memory (e.g. Eigen objects).
				/// Therefore, we require the necessary temporaries to be built outside the class
				/// and a reference to them at construction.
				MLVector< SCALAR >& gradient_tmp;	
				mutable Eigen::Map< MLMatrix< SCALAR > > weights;
				mutable Eigen::Map< MLVector< SCALAR > > bias_visible;
				mutable Eigen::Map< MLVector< SCALAR > > bias_hidden;
				mutable Eigen::Map< MLVector< SCALAR > > additional_parameters_visible;
				mutable Eigen::Map< MLVector< SCALAR > > additional_parameters_hidden;
				mutable bool chain_initialized = false;
			private: // friends declarations
				template <RBMUnitType V, RBMUnitType H >
				friend class FreeEnergy;
				template < RBMUnitType H >
				friend class HiddenIntegral;
			private: // helper functions
				void mapVariables(){
					auto N_vis = rbm.visible_units.size();
					auto N_hid = rbm.hidden_units.size();
					auto N_weights = rbm.weights.size();
					auto N_bias_hid = rbm.bias_hidden.size();
					auto N_bias_vis = rbm.bias_visible.size();
					auto N_add_hid = rbm.additional_parameters_hidden.size();
					auto N_add_vis = rbm.additional_parameters_visible.size();
					MLEARN_ASSERT(N_vis==inputs.rows(),"Visible units must be the same size as the training samples!");
					gradient_tmp.resize( N_weights + N_bias_hid + N_bias_vis + N_add_vis + N_add_hid );
					new (&weights) Eigen::Map< MLMatrix<SCALAR> >( gradient_tmp.data(), N_hid, N_vis );
					auto offset = N_weights;
					new (&bias_hidden) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_hid );
					offset += N_hid;
					new (&bias_visible) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_vis );
					offset += N_vis;
					new (&additional_parameters_hidden) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_add_hid );
					offset += N_add_hid;
					new (&additional_parameters_visible) Eigen::Map< MLVector<SCALAR> >( gradient_tmp.data() + offset, N_add_vis );
				}
				// sample point to use in gradient estimation
				template < RBMTrainingMode InnerMode = MODE >
				typename std::enable_if< InnerMode == RBMTrainingMode::CONTRASTIVE_DIVERGENCE,void >::type sample_for_gradient( decltype(inputs.cols()) idx ) const{
					for ( size_t i = 0; i < N_chain; ++i ){
						rbm.sampleHFromV(inputs.col(idx));
						rbm.sampleVFromH();
					}
				}
				template < RBMTrainingMode InnerMode = MODE >
				typename std::enable_if< InnerMode == RBMTrainingMode::PERSISTENT_CONTRASTIVE_DIVERGENCE,void >::type sample_for_gradient( decltype(inputs.cols()) idx ) const{
					if (!chain_initialized){
						rbm.sampleHFromV(inputs.col(idx));
						rbm.sampleVFromH();
						chain_initialized = true;
					}else{
						rbm.sampleHFromV();
						rbm.sampleVFromH();
					}

					for ( size_t i = 1; i < N_chain; ++i ){
						rbm.sampleHFromV();
						rbm.sampleVFromH();
					}
				}
				// regularizations
				// L2 loss
				template< typename DERIVED >
				inline SCALAR L2Regularization( const Eigen::MatrixBase<DERIVED>& x, std::true_type T ) const{
					auto N_vis = rbm.visible_units.size();
					auto N_hid = rbm.hidden_units.size();
					return options._l2_param*x.head( N_vis*N_hid ).cwiseAbs2().sum();
				}
				template< typename DERIVED >
				inline SCALAR L2Regularization( const Eigen::MatrixBase<DERIVED>& x, std::false_type F ) const{ return SCALAR(0); }
				// L2 gradient
				template< typename DERIVED, typename DERIVED_2 >
				inline void L2Regularization( const Eigen::MatrixBase<DERIVED>& x,Eigen::MatrixBase<DERIVED_2>& gradient, std::true_type T ) const{
					auto N = rbm.visible_units.size()*rbm.hidden_units.size();
					gradient.head(N) +=  2*options._l2_param*x.head( N );
				}
				template< typename DERIVED, typename DERIVED_2 >
				inline void L2Regularization( const Eigen::MatrixBase<DERIVED>& x,Eigen::MatrixBase<DERIVED_2>& gradient, std::false_type F ) const{}
				// L1 loss
				template< typename DERIVED >
				inline SCALAR L1Regularization( const Eigen::MatrixBase<DERIVED>& x, std::true_type T ) const{
					auto N_vis = rbm.visible_units.size();
					auto N_hid = rbm.hidden_units.size();
					return options._l1_param*x.head( N_vis*N_hid ).cwiseAbs().sum();
				}
				template< typename DERIVED >
				inline SCALAR L1Regularization( const Eigen::MatrixBase<DERIVED>& x, std::false_type F ) const{ return SCALAR(0); }
				// L2 gradient
				template< typename DERIVED, typename DERIVED_2 >
				inline void L1Regularization( const Eigen::MatrixBase<DERIVED>& x,Eigen::MatrixBase<DERIVED_2>& gradient, std::true_type T ) const{
					auto N = rbm.visible_units.size()*rbm.hidden_units.size();
					gradient.head(N) +=  options._l1_param*x.head( N ).unaryExpr( [](const SCALAR& s){ return static_cast<SCALAR>(s > 0); } );
				}
				template< typename DERIVED, typename DERIVED_2 >
				inline void L1Regularization( const Eigen::MatrixBase<DERIVED>& x,Eigen::MatrixBase<DERIVED_2>& gradient, std::false_type F ) const{}
			};


		};
	}
}

#endif