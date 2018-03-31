#ifndef NEURAL_NET_MLEARN_H_FILE
#define NEURAL_NET_MLEARN_H_FILE

// STL includes
#include <tuple>

// MLearn includes
#include <MLearn/Core>
#include <MLearn/Optimization/CostFunction.h>

#define MAKE_NEURAL_NETWORK_FRIEND\
	template <typename Scalar, LossType L_t, typename ... LAYERS>\
	friend class NeuralNetwork;\
	template <int _I, typename _S, , LossType _L_t, typename ... _Ls>\
	friend struct internal::_Layer_Handler;\
	template <typename ... _LAYERS>\
	friend class std::tuple;\
	template<int _Idx, typename... _Elements>\
	friend struct std::_Tuple_impl;\
	template<int _Idx, typename _Head, bool _IsEmpty>\
	friend struct std::_Head_base;\

namespace MLearn{
namespace nn{

template <typename Scalar, LossType L_t, typename ... LAYERS>
class NeuralNetwork;

namespace internal{

	// Internal helper functions
	template <int I, typename Scalar, LossType L_t, typename ... LAYERS>
	struct _Layer_Handler{
		static inline void set_input_dim(NeuralNetwork<Scalar, LAYERS...>& net){
			std::get<I>(net._layers).set_input_dim(std::get<I-1>(net._layers).get_output_dim());
			_Layer_Handler<I-1, Scalar, L_t, LAYERS...>::set_input_dim(net);
		}
		static inline int get_n_parameters(NeuralNetwork<Scalar, LAYERS...>& net){
			return std::get<I>(net._layers).get_n_parameters() + 
					_Layer_Handler<I-1, Scalar, L_t, LAYERS...>::get_n_parameters(net);
		}
		static inline void set_weights(NeuralNetwork<Scalar, LAYERS...>& net, const Scalar *weights_array){
			const Scalar *curr_weights = weights_array - std::get<I>(net._layers).get_n_parameters();
			std::get<I>(net._layers).set_weights(curr_weights);
			_Layer_Handler<I-1, Scalar, L_t, LAYERS...>::set_weights(net, curr_weights);
		}
		static inline void set_grad_weights(NeuralNetwork<Scalar, LAYERS...>& net, Scalar *grad_weights_array){
			const Scalar *curr_grad_weights = grad_weights_array - std::get<I>(net._layers).get_n_parameters();
			std::get<I>(net._layers).set_grad_weights(curr_grad_weights);
			_Layer_Handler<I-1, Scalar, L_t, LAYERS...>::set_grad_weights(net, curr_grad_weights);
		}
		static inline void forward_pass(NeuralNetwork<Scalar, LAYERS...>& net, const MLMatrix<Scalar>& input){
			_Layer_Handler<I-1, Scalar, L_t, LAYERS...>::forward_pass(net, input);
			std::get<I>(net._layers).forward_pass(std::get<I-1>(net._layers).get_output());
		}
		static inline void backpropagate(NeuralNetwork<Scalar, LAYERS...>& net, 
										 const MLMatrix<Scalar>& grad_output, 
										 bool compute_input_gradient){

			std::get<I>(net._layers).backpropagate(grad_output, true);
			_Layer_Handler<I-1, Scalar, L_t, LAYERS...>::backpropagate(net, 
																	   std::get<I>(net._layers).get_input_gradient(), 
																	   compute_input_gradient);
		}
	};

	template <typename Scalar, LossType L_t, typename ... LAYERS>
	struct _Layer_Handler<0, Scalar, L_t, LAYERS...>{
		static inline void set_input_dim(NeuralNetwork<Scalar, LAYERS...>& net){}
		static inline int get_n_parameters(NeuralNetwork<Scalar, LAYERS...>& net){
			return std::get<0>(net._layers).get_n_parameters();
		}
		static inline void set_weights(NeuralNetwork<Scalar, LAYERS...>& net, const Scalar *weights_array){
			const Scalar *curr_weights = weights_array - std::get<0>(net._layers).get_n_parameters();
			std::get<0>(net._layers).set_weights(curr_weights);
		}
		static inline void set_grad_weights(NeuralNetwork<Scalar, LAYERS...>& net, Scalar *grad_weights_array){
			Scalar *curr_grad_weights = grad_weights_array - std::get<0>(net._layers).get_n_parameters();
			std::get<0>(net._layers).set_grad_weights(curr_grad_weights);
		}
		static inline void forward_pass(NeuralNetwork<Scalar, LAYERS...>& net, const MLMatrix<Scalar>& input){
			std::get<0>(net._layers).forward_pass(input);
		}
		static inline void backpropagate(NeuralNetwork<Scalar, LAYERS...>& net, 
										 const MLMatrix<Scalar>& grad_output, 
										 bool compute_input_gradient){
			std::get<0>(net._layers).backpropagate(grad_output, compute_input_gradient);
		}
	};
}

/*!
	\brief Neural Network
*/
template <typename Scalar, LossType Loss, typename ... LAYERS>
class NeuralNetwork{
	template <int I, typename S, LossType L_t, typename ... L>
	friend struct internal::_Layer_Handler;

	/*!
		\brief Cost function for training
	*/
	class NeuralNetCost: public CostFunction<NeuralNetCost>{
	public:
		NeuralNetCost(NeuralNetwork<Scalar, LossType, LAYERS...>& net, const MLMatrix<Scalar>& x, const MLMatrix<Scalar>& y):
			_net(net), _input(x), _output(y){}

		TEMPLATED_SIGNATURE_EVAL_FUNCTION(w){
			_net.set_weights(w.data(), false);
			out = _net.forward_pass(x);
			Scalar loss = 0;
			for (int i = 0; i < out.cols(); ++i){
				loss += LossFunction<Loss>::evaluate(out.col(i));
			}
			return loss/Scalar(out.cols());
		}

		TEMPLATED_SIGNATURE_ANALYTICAL_GRADIENT_FUNCTION(w, grad_w){
			grad_w.resize(w.size());

			_net.set_weights(w.data(), false);
			_net.set_grad_weights(grad_w.data());

			forward_and_back_pass(_input, _output);
		}

		TEMPLATED_SIGNATURE_STOCHASTIC_GRADIENT_FUNCTION(w, grad_w, sample_idx){
			grad_w.resize(w.size());

			_net.set_weights(w.data(), false);
			_net.set_grad_weights(grad_w.data());

			_sampled_input.resize(_input.rows(), sample_idx.size());
			_sampled_output.resize(_output.rows(), sample_idx.size());

			for (int i = 0; i < sample_idx.size(); ++i){
				_sampled_input.col(i) = _input.col(sample_idx[i]);
				_sampled_output.col(i) = _output.col(sample_idx[i]);
			}

			forward_and_back_pass(_sampled_input, _sampled_output);
		}
	private:
		MLMatrix<Scalar> _grad_output;
		MLMatrix<Scalar> _sampled_input;
		MLMatrix<Scalar> _sampled_output;
		void forward_and_back_pass(const MLMatrix<Scalar>& input, const MLMatrix<Scalar>& output){
			const MLMatrix<Scalar>& output_hat = _net.forward_pass(input);
			_grad_output.resize(output.rows(), output.cols());
			for (int i = 0; i < output.cols(); ++i){
				LossFunction<Loss>::gradient(output_hat.col(i), output.col(i), _grad_output.col(i));	
			}
			_net.backpropagate(_grad_output, false);
		}
	};
public:
	/*!
		\brief Constructor from const references
	*/
	NeuralNetwork(const LAYERS&... layers): _layers(layers...){}
	
	/*!
		\brief Constructor from rvalue references
	*/
	NeuralNetwork(LAYERS&&... layers): _layers(layers...){}
	
	/*!
		\brief Layer getter
	*/
	template <std::size_t I>
	typename std::tuple_element<I, std::tuple<LAYERS...> >::type& get_layer() {return std::get<I>(_layers);}
	
	/*!
		\brief Layer getter
	*/
	template <std::size_t I>
	const typename std::tuple_element<I, std::tuple<LAYERS...> >::type& get_layer() const {return std::get<I>(_layers);}

	/*!
		\brief Set input dimension
	*/
	void set_input_dim(int dim){
		MLEARN_ASSERT(dim > 0, "[NeuralNetwork::set_input_dim] Non-positive input dim not valid!");
		std::get<0>(_layers).set_input_dim(dim);
		internal::_Layer_Handler<sizeof...(LAYERS)-1, Scalar, LAYERS...>::set_input_dim(*this);
		internal::_Layer_Handler<sizeof...(LAYERS)-1, Scalar, LAYERS...>::get_n_parameters(*this);
	}

	/*!
		\brief Set weights
	*/
	void set_weights(const Scalar* weights_array, bool copy = true){
		MLEARN_ASSERT(weights_array != NULL, "[NeuralNetwork::set_weights] Invalid pointer!");
		internal::_Layer_Handler<sizeof...(LAYERS)-1, Scalar, LAYERS...>::set_weights(*this, 
																				weights_array + _total_n_parameters);
		if (copy){
			_trained_weights = Eigen::Map<MLVector<Scalar>>(weights_array, _total_n_parameters);
		}
	}

	/*!
		\brief Set weights
	*/
	void set_grad_weights(Scalar* grad_weights_array){
		MLEARN_ASSERT(grad_weights_array != NULL, "[NeuralNetwork::set_grad_weights] Invalid pointer!");
		internal::_Layer_Handler<sizeof...(LAYERS)-1, Scalar, LAYERS...>::set_grad_weights(*this, 
																			grad_weights_array + _total_n_parameters);
	}

	/*!
		\brief Forward pass
	*/
	const MLMatrix<Scalar>& forward_pass(const MLMatrix<Scalar>& input){
		internal::_Layer_Handler<sizeof...(LAYERS)-1, Scalar, LAYERS...>::forward_pass(*this, input);
		return std::get<sizeof...(LAYERS)-1>(_layers).get_output();
	}

	/*!
		\brief Backpropagate
	*/
	void backpropagate(const MLMatrix<Scalar>& grad_output, bool compute_input_gradient = false){
		internal::_Layer_Handler<sizeof...(LAYERS)-1, Scalar, LAYERS...>::backpropagate(*this,
																						grad_output, 
																						compute_input_gradient);
	}

	/*!
		\brief Fit to data
	*/
	void fit(const MLMatrix<Scalar>& X, const MLMatrix<Scalar>& Y){

	}

protected:
	std::tuple<LAYERS...> _layers;
	MLVector<Scalar> _trained_weights;
	int _total_n_parameters = -1;
};

/*!
	\brief Util function to construct a network
*/
template <typename Scalar, LossType L_t, typename ... LAYERS>
NeuralNetwork<Scalar, L_t, LAYERS...> make_network(const LAYERS&... layers){
	return NeuralNetwork<Scalar, L_t, LAYERS...>(layers...);
}
/*!
	\brief Util function to construct a network
*/
template <typename Scalar, LossType L_t, typename ... LAYERS>
NeuralNetwork<Scalar, L_t, LAYERS...> make_network(LAYERS&&... layers){
	return NeuralNetwork<Scalar, L_t, LAYERS...>(layers...);
}


}}

#endif // NEURAL_NET_MLEARN_H_FILE