#ifndef LAYER_BASE_UTILS_MLEARN_H_FILE
#define LAYER_BASE_UTILS_MLEARN_H_FILE

// MLearn includes
#include <MLearn/Core>

// STL includes
#include <cmath>
#include <type_traits>

/*!
	\def DEFINE_LAYER_NAME(name)
	Define a dummy struct to be used as layer name
*/
#define DEFINE_LAYER_NAME(layer_name) struct layer_name { \
	static constexpr char* get_name() { return #layer_name; } \
};

namespace MLearn{
namespace nn{

/*!
	\enum Activation type
*/
enum class ActivationType{
	SIGMOID, 
	RELU, 
	TANH,
	ATAN,
	LEAKY_RELU
};


/*!
	\brief Activation parameters
*/
template <typename Scalar, ActivationType A>
struct ActivationParams{};

/*!
	\brief Activation type traits
*/
template <ActivationType A>
struct ActivationTraits{
	enum { n_parameters = 0 };
	enum { efficient_derivative = false };
};


// dummy version of efficient derivative compute functions 
#define DEFINE_DUMMY_COMPUTE_DERIVATIVE_ACTIVATED()\
	template <typename Scalar>\
	static inline Scalar compute_derivative_activated(const Scalar& f){ \
		throw MLNotImplemented(); \
	}


// dummy version of activation compute functions using parameters
#define DEFINE_DUMMY_COMPUTE_WITH_PARAMETERS_FUNCTION(ACTIVATION)\
	template <typename Scalar>\
	static inline Scalar compute(const Scalar& x, const ActivationParams<Scalar, ACTIVATION>& params){ \
		return compute(x); \
	}\
	template <typename Scalar>\
	static inline Scalar compute_derivative(const Scalar& x, const ActivationParams<Scalar, ACTIVATION>& params){ \
		return compute_derivative(x); \
	}\
	template <typename Scalar>\
	static inline Scalar compute_derivative_activated(const Scalar& f, const ActivationParams<Scalar, ACTIVATION>& params){ \
		return compute_derivative_activated(f); \
	}

#define DEFINE_DUMMY_COMPUTE_DERIVATIVE_ACTIVATED_WITH_PARAMETERS(ACTIVATION)\
	template <typename Scalar>\
	static inline Scalar compute_derivative_activated(const Scalar& f, const ActivationParams<Scalar, ACTIVATION>& params){ \
		return compute_derivative_activated(f); \
	}

/*!
	\brief Activation functor - non-specialised
*/
template <ActivationType A>
struct Activation{
	/*! 
		\brief Compute activation
	*/
	template <typename Scalar>
	static inline Scalar compute(const Scalar& x){ throw MLNotImplemented(); }

	/*! 
		\brief Compute derivative of activation
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative(const Scalar& x){ throw MLNotImplemented(); }

	/*! 
		\brief Compute derivative of activation with given activated value

		This is a more computationally efficient derivative computation in case the activated value is already available.
		E.g. for the sigmoid function, if f is the activated value, then f' = f*(1-f)
	*/
	DEFINE_DUMMY_COMPUTE_DERIVATIVE_ACTIVATED();
	DEFINE_DUMMY_COMPUTE_WITH_PARAMETERS_FUNCTION(A);
};


/*!
	\brief SIGMOID specializations
*/
template <>
struct ActivationTraits<ActivationType::SIGMOID>{
	enum { n_parameters = 0 };
	enum { efficient_derivative = true };
};

template <>
struct Activation<ActivationType::SIGMOID>{
	/*!
	\brief Activation operator
	*/
	template <typename Scalar>
	static inline Scalar compute(const Scalar& x){ return 1./(1. + std::exp(-x)); }

	/*!
	\brief Activation derivative operator
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative(const Scalar& x){ 
		Scalar exp_x = std::exp(-x); 
		Scalar exp_xp1 = exp_x + 1.;
		return exp_x/(exp_xp1*exp_xp1); 
	}

	/*!
	\brief Activation derivate operator - given activate value 
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative_activated(const Scalar& f){ 
		return f*(1. - f);
	}

	DEFINE_DUMMY_COMPUTE_WITH_PARAMETERS_FUNCTION(ActivationType::SIGMOID);
};

/*!
	\brief RELU specializations
*/
template <>
struct Activation<ActivationType::RELU>{
	/*!
	\brief Activation operator
	*/
	template <typename Scalar>
	static inline Scalar compute(const Scalar& x){ return std::max(0., x); }

	/*!
	\brief Activation derivative operator
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative(const Scalar& x){ 
		if (x >= 0.){
			return 1.;
		}else{
			return 0.;
		}
	}

	DEFINE_DUMMY_COMPUTE_DERIVATIVE_ACTIVATED();
	DEFINE_DUMMY_COMPUTE_WITH_PARAMETERS_FUNCTION(ActivationType::RELU);
};

/*!
	\brief TANH specializations
*/
template <>
struct ActivationTraits<ActivationType::TANH>{
	enum { n_parameters = 0 };
	enum { efficient_derivative = true };
};

template <>
struct Activation<ActivationType::TANH>{
	/*!
	\brief Activation operator
	*/
	template <typename Scalar>
	static inline Scalar compute(const Scalar& x){ return std::tanh(x); }

	/*!
	\brief Activation derivative operator
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative(const Scalar& x){ 
		Scalar sech = 1.0/std::cosh(x);
		return sech*sech; 
	}

	/*!
	\brief Activation derivate operator - given activate value 
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative_activated(const Scalar& f){ 
		return (1. - f*f);
	}

	DEFINE_DUMMY_COMPUTE_WITH_PARAMETERS_FUNCTION(ActivationType::TANH);
};

/*!
	\brief ATAN specializations
*/
template <>
struct Activation<ActivationType::ATAN>{
	/*!
	\brief Activation operator
	*/
	template <typename Scalar>
	static inline Scalar compute(const Scalar& x){ return std::atan(x); }

	/*!
	\brief Activation derivative operator
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative(const Scalar& x){ 
		return 1./(x*x + 1.);
	}

	DEFINE_DUMMY_COMPUTE_DERIVATIVE_ACTIVATED();
	DEFINE_DUMMY_COMPUTE_WITH_PARAMETERS_FUNCTION(ActivationType::ATAN);
};


/*!
	\brief RELU specializations

	If no parameters are provided, the classic RELU is computed
*/
template <>
struct ActivationTraits<ActivationType::LEAKY_RELU>{
	enum { n_parameters = 1 };
	enum { efficient_derivative = false };
};


template <typename Scalar>
struct ActivationParams<Scalar, ActivationType::LEAKY_RELU>{
	Scalar a = 0.0;
};

template <>
struct Activation<ActivationType::LEAKY_RELU>{
	/*!
	\brief Activation operator
	*/
	template <typename Scalar>
	static inline Scalar compute(const Scalar& x){ return Activation<ActivationType::RELU>::compute(x); }

	/*!
	\brief Activation derivative operator
	*/
	template <typename Scalar>
	static inline Scalar compute_derivative(const Scalar& x){ 
		return Activation<ActivationType::RELU>::compute_derivative(x);
	}

	template <typename Scalar>
	static inline Scalar compute(const Scalar& x, const ActivationParams<Scalar, ActivationType::LEAKY_RELU>& params){ 
		if (x < 0.){
			return params.a*x;
		}else{
			return x;
		}
	}
	template <typename Scalar>
	static inline Scalar compute_derivative(const Scalar& x, const ActivationParams<Scalar, ActivationType::LEAKY_RELU>& params){ 
		if (x < 0.){
			return params.a;
		}else{
			return 1.;
		}
	}

	DEFINE_DUMMY_COMPUTE_DERIVATIVE_ACTIVATED();
	DEFINE_DUMMY_COMPUTE_DERIVATIVE_ACTIVATED_WITH_PARAMETERS(ActivationType::LEAKY_RELU);
};

/*!
	\brief Helper class to select (at compile-time) which derivative to use
*/
namespace internal{

template <typename Scalar, ActivationType A>
struct ActivationDerivativeBaseWrapper{
	ActivationParams<Scalar, A> _params; 
	/*!
		\brief Constructor
	*/
	ActivationDerivativeBaseWrapper() = default;
	/*!
		\brief Parameters setter
	*/
	void set_params(const ActivationParams<Scalar, A>& in_params){
		_params = in_params;
	}

	/*!
		\brief Parameters getter
	*/
	const ActivationParams<Scalar, A>& get_params() const{
		return _params;
	}
};

template <typename Scalar, ActivationType A, bool efficient>
struct ActivationDerivativeInternalWrapper{};

/*!
	\brief Version using activated values
	\param x pre-activation values
	\param f_x activated values
	\param params parameters
*/
template <typename Scalar, ActivationType A>
struct ActivationDerivativeInternalWrapper<Scalar, A, true>: public ActivationDerivativeBaseWrapper<Scalar, A>{
	using ActivationDerivativeBaseWrapper<Scalar, A>::ActivationDerivativeBaseWrapper;
	using ActivationDerivativeBaseWrapper<Scalar, A>::set_params;
	using ActivationDerivativeBaseWrapper<Scalar, A>::get_params;
	inline Scalar operator()(const Scalar& x, const Scalar& f_x){
		return Activation<A>::compute_derivative_activated(f_x, this->_params);
	}
};

template <typename Scalar, ActivationType A>
struct ActivationDerivativeInternalWrapper<Scalar, A, false>: public ActivationDerivativeBaseWrapper<Scalar, A>{
	using ActivationDerivativeBaseWrapper<Scalar, A>::ActivationDerivativeBaseWrapper;
	using ActivationDerivativeBaseWrapper<Scalar, A>::set_params;
	using ActivationDerivativeBaseWrapper<Scalar, A>::get_params;
	inline Scalar operator()(const Scalar& x, const Scalar& f_x){
		return Activation<A>::compute_derivative(x, this->_params);
	}
};

}

template <typename Scalar, ActivationType A>
using ActivationDerivativeWrapper = 
			internal::ActivationDerivativeInternalWrapper<Scalar, A, ActivationTraits<A>::efficient_derivative>;

}}


#endif // LAYER_BASE_UTILS_MLEARN_H_FILE