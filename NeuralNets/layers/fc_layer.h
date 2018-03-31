#ifndef FC_LAYER_MLEARN_H_FILE
#define FC_LAYER_MLEARN_H_FILE

// STL includes
#include <exception>
#include <functional>
#include <tuple>

// MLearn includes
#include <MLearn/Core>
#include <MLearn/NeuralNets/layers/base/utils.h>
#include <MLearn/NeuralNets/neural_nets.h>

// Eigen includes
#include <Eigen/Core>

namespace MLearn{
namespace nn{

/*
	\brief Fully connected layer

	Only one constructor exposed. (Temporary) Instantiation is the only allowed operation to the end-user.
	The user may access various fields and properties via the friend class NeuralNetwork.
*/
template <typename Scalar, ActivationType A>
class FCLayer{

	MAKE_NEURAL_NETWORK_FRIEND

public:
	/*!
		\brief Constructor
	*/
	FCLayer(int output_dim, bool has_bias = true): _output_dim(output_dim), _has_bias(has_bias), 
			_W(NULL, 0, 0), _b(NULL, 0), _grad_W(NULL, 0, 0), _grad_b(NULL, 0){
		MLEARN_ASSERT(output_dim > 0, "[FCLayer::FCLayer] Non-positive output dim not valid!");
	}

	/*!
		\brief (Hyper-)Parameters setter
	*/
	void set_hyperparams(const ActivationParams<Scalar, A>& in_params){
		_derivative_computer.set_params(in_params);
	}

	/*!
		\brief (Hyper-)Parameters getter
	*/
	const ActivationParams<Scalar, A>& get_hyperparams() const{
		return _derivative_computer.get_params();
	}
protected:
	int _input_dim = -1;
	const int _output_dim = -1;
	int _n_parameters = -1;
	const bool _has_bias = true;
	ActivationDerivativeWrapper<Scalar, A> _derivative_computer;

	MLMatrix<Scalar> _preactivation; //!< Values of neurons before activation function is applied. 
	MLMatrix<Scalar> _temp_gradient; 
	MLMatrix<Scalar> _output;  
	MLMatrix<Scalar> _grad_input; 
	const MLMatrix<Scalar>* _input = NULL;

	Eigen::Map<const MLMatrix<Scalar>> _W;
	Eigen::Map<const MLVector<Scalar>> _b;

	Eigen::Map<MLMatrix<Scalar>> _grad_W;
	Eigen::Map<MLVector<Scalar>> _grad_b;

	// Functions only for friend's classes usage
	/*!
		\brief Copy constructor
	*/
	FCLayer(const FCLayer& other_layer): FCLayer(other_layer._output_dim, other_layer._has_bias){}
	
	/*!
		\brief Move constructor
	*/
	FCLayer(FCLayer&& other_layer): FCLayer(other_layer._output_dim, other_layer._has_bias){}
	/*!
		\brief Input setter
	*/
	void set_input_dim(int input_dim){
		if (_input_dim == -1){
			MLEARN_ASSERT(input_dim > 0, "[FCLayer::set_input_dim] Non-positive input dim not valid!");
			_input_dim = input_dim;
			_n_parameters = _input_dim*_output_dim;
			if (_has_bias){
				_n_parameters += _output_dim;
			}
		}else{
			throw std::logic_error("[FCLayer::set_input_dim] Input dimension cannot be set anymore!");
		}
	}
	/*!
		\brief Weights setter
	*/
	void set_weights(const Scalar *weights){
		MLEARN_ASSERT(_input_dim > 0, "[FCLayer::set_weights] Need to set the input dimension first!");
		MLEARN_ASSERT(weights != NULL, "[FCLayer::set_weights] Input pointer not valid!");
		new (&_W) Eigen::Map<const MLMatrix<Scalar>>(weights,_output_dim, _input_dim);
		if (_has_bias){
			new (&_b) Eigen::Map<const MLVector<Scalar>>(weights + _output_dim*_input_dim, _output_dim);
		}
	}
	void set_grad_weights(Scalar *grad_weights){
		MLEARN_ASSERT(_input_dim > 0, "[FCLayer::set_grad_weights] Need to set the input dimension first!");
		MLEARN_ASSERT(grad_weights != NULL, "[FCLayer::set_grad_weights] Input pointer not valid!");
		new (&_grad_W) Eigen::Map<MLMatrix<Scalar>>(grad_weights,_output_dim, _input_dim);
		if (_has_bias){
			new (&_grad_b) Eigen::Map<MLVector<Scalar>>(grad_weights + _output_dim*_input_dim, _output_dim);
		}
	}
	/*!
		\brief Get number of parameters
	*/
	int get_n_parameters() const{
		return _n_parameters;
	}

	/*!
		\brief Get output dimensions
	*/
	int get_output_dim() const{
		return _output_dim;
	}

	/*!
		\brief Check if layer has bias
	*/
	bool has_bias() const{
		return _has_bias;
	}
	/*!
		\brief Output getter
	*/
	const MLMatrix<Scalar>& get_output() const{
		return _output;
	}
	/*!
		\brief Input gradient getter
	*/
	const MLMatrix<Scalar>& get_grad_input() const{
		return _grad_input;
	}
	/*!
		\brief Forward propagate
	*/
	const MLMatrix<Scalar>& forward_pass(const MLMatrix<Scalar>& input){
		MLEARN_ASSERT(input.rows() == _input_dim, "[FCLayer::forward_pass] Dimensions not consistent!");
		MLEARN_ASSERT(input.cols() > 0, "[FCLayer::forward_pass] Expected at least one sample!");
		_input = &input;

		MLEARN_ASSERT(_W.data() != NULL, "[FCLayer::forward_pass] No memory assigned to W!");
		_preactivation.noalias() = _W*input;
		if (_has_bias){
			MLEARN_ASSERT(_b.data() != NULL, "[FCLayer::forward_pass] No memory assigned to b!");
			_preactivation.colwise() += _b;
		}
		_output.noalias() = _preactivation.unaryExpr(
			std::bind(
				(Scalar(*)(const Scalar&, const ActivationParams<Scalar, A>&)) &Activation<A>::template compute<Scalar>, 
					std::placeholders::_1, _derivative_computer.get_params())); 
		return _output;
	}
	/*!
		\brief Backpropagate
	*/
	void backpropagate(const MLMatrix<Scalar>& out_gradient, bool compute_grad_input = true){
		MLEARN_ASSERT(_input != NULL, "[FCLayer::backpropagate] Forward pass has not been performed!");
		MLEARN_ASSERT(out_gradient.rows() == _output_dim, "[FCLayer::backpropagate] Dimensions not consistent!");
		MLEARN_ASSERT(out_gradient.cols() > 0, "[FCLayer::backpropagate] Expected at least one sample!");
		
		_temp_gradient.noalias() = _preactivation.binaryExpr(_output, _derivative_computer).cwiseProduct(out_gradient);

		if (_has_bias){
			MLEARN_ASSERT(_grad_b.data() != NULL, "[FCLayer::backpropagate] No memory assigned to the b gradient!");
			_grad_b.noalias() = _temp_gradient.rowwise().sum();
		}
		MLEARN_ASSERT(_grad_W.data() != NULL, "[FCLayer::backpropagate] No memory assigned to the W gradient!");
		_grad_W.noalias() = _temp_gradient*(*_input).transpose();

		if (compute_grad_input){
			MLEARN_ASSERT(_W.data() != NULL, "[FCLayer::backpropagate] No memory assigned to W!");
			_grad_input = _W.transpose()*_temp_gradient;
		}
	}
};

}}

#endif // FC_LAYER_MLEARN_H_FILE