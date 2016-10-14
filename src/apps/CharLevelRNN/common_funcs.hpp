#ifndef COMMON_FUNCS_CHAR_RNN
#define COMMON_FUNCS_CHAR_RNN

#include <iostream>
#include <fstream>

typedef double FLOAT_TYPE;
typedef uint INDEX_TYPE;
uint NEncodedChars = 128;

constexpr MLearn::NeuralNets::ActivationType hidden_act = MLearn::NeuralNets::ActivationType::HYPER_TAN;
constexpr MLearn::NeuralNets::ActivationType output_act = MLearn::NeuralNets::ActivationType::LINEAR;
constexpr MLearn::NeuralNets::RecurrentNets::RNNType cell_type = MLearn::NeuralNets::RecurrentNets::RNNType::GRU;
constexpr MLearn::LossType loss = MLearn::LossType::SOFTMAX_CROSS_ENTROPY;

typedef MLearn::NeuralNets::RecurrentNets::RecurrentLayer< FLOAT_TYPE, INDEX_TYPE, cell_type, hidden_act, output_act > RNNLayer;

bool weightsFromFile(const std::string& weights_file, MLearn::MLVector<FLOAT_TYPE>& weights_, const int expected_n_weights){
	std::ifstream in;
	in.open(weights_file,std::ios::in | std::ios::binary);
	if ( in.fail() ){
		return false;
	}

    in.seekg (0, in.end);
    int length = FLOAT_TYPE(in.tellg())/FLOAT_TYPE(expected_n_weights);
    in.seekg (0, in.beg);

    if (length > expected_n_weights){
    	std::cerr << "Too much data! Check if the stored weights are compatible with your current model!" << std::endl;
    	in.close();
    	return false;
    }

	weights_.resize(expected_n_weights);
	in.read( reinterpret_cast<char*>(&weights_[0]),sizeof(FLOAT_TYPE)*expected_n_weights);

	if (in){
		std::clog << "Weights successfully read!" << std::endl;
	}else{
		std::cerr << "Error in reading weights file!" << std::endl;
		in.close();
		return false;
	}

	in.close();
	return true;
}

bool weightsToFile(const std::string& weights_file, const MLearn::MLVector<FLOAT_TYPE>& _weights){
	std::ofstream out;
	out.open(weights_file,std::ios::out | std::ios::binary);
	if ( out.fail() ){
		return false;
	}
	for ( int i = 0; i < _weights.size(); ++i ){
		out.write( reinterpret_cast<const char*>(&_weights[i]),sizeof(FLOAT_TYPE) );
	}
	out.close();
	return true;
}

#endif