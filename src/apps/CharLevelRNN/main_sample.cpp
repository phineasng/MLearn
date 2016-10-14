#include <MLearn/Core>
#include <MLearn/NeuralNets/RecurrentNets/RecurrentCellType.h>
#include <MLearn/NeuralNets/RecurrentNets/RecurrentLayer.h>
#include <MLearn/NeuralNets/ActivationFunction.h>

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>

#include <boost/program_options.hpp>

#include "common_funcs.hpp"

int main(int ac, char* av[]){

	//
	using namespace MLearn;
	namespace po = boost::program_options;

	// program options
	po::options_description desc("Sample text given a Character Language Model RNN.\nBasic usage: ./CharLevelRNNSample --weight_file <path_to_weight_file>");
	desc.add_options()
    	("help", "See all the available options.")
    	("weight_file", po::value<std::string>(), "Path to a file containing trained weights.")
    	("hidden_size", po::value<uint>(), "Size of the hidden state of the RNN (default: 128).")
    	("sentence_start", po::value<std::string>(), "Start of the text to be sampled (default: 'The next day ')." )
    	("sampling_mode", po::value<bool>(), "Sampling strategy: (0) max probability (1) sample from distribution (default: 1)." )
    	("n_characters", po::value<int>(), "Number of characters to sample (default: 200).");

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")){
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("weight_file")){

		// Default parameters
		uint hidden_size = 128u;
		int n_characters = 200;
		std::string sentence_start = "The next day ";
		bool sample = true;

		// Modify default parameters if necessary
		if (vm.count("hidden_size")){
			hidden_size = vm["hidden_size"].as<uint>();
			std::cout << "User-defined hidden_size: " << hidden_size << std::endl;
		}
		if (vm.count("n_characters")){
			n_characters = vm["n_characters"].as<int>();
			std::cout << "User-defined n_characters: " << n_characters << std::endl;
		}
		if (vm.count("sentence_start") && vm["sentence_start"].as<std::string>().length() > 1){
			sentence_start = vm["sentence_start"].as<std::string>();
			std::cout << "User-defined sentence_start: " << sentence_start << std::endl;
		}
		if (vm.count("sampling_mode")){
			sample = vm["sampling_mode"].as<bool>();
			std::cout << "User-defined sampling_mode: " << sample << std::endl;
		}

		// Create model
		RNNLayer rnn(NEncodedChars,hidden_size,NEncodedChars);
		// -- Set weights
		uint size_weights = rnn.getNWeights();
		MLVector<FLOAT_TYPE> weights;
		bool read_success = weightsFromFile(vm["weight_file"].as<std::string>(),weights,size_weights);
		if (!read_success){
			return 1;
		}
		rnn.attachWeightsToCell(weights);

		MLearn::MLVector<FLOAT_TYPE> input_char(NEncodedChars);
		MLearn::MLVector<FLOAT_TYPE> out_dist(NEncodedChars);
		input_char.setZero();

		// Run the beginning of the sentence
		for ( char& c : sentence_start ){
			input_char.setZero();
			input_char[int(c)] = 1.0;
			out_dist = rnn.forwardpass_step_output(input_char);
		}

		int ch = 0;

		std::mt19937_64 rnd_gen( std::chrono::system_clock::now().time_since_epoch().count() );
		std::cout << std::endl << std::endl << "=========== SAMPLING TEXT ============" << std::endl << sentence_start;
		for ( int curr_char = 0; curr_char < n_characters; ++curr_char ){

			out_dist = out_dist.unaryExpr( std::pointer_to_unary_function< FLOAT_TYPE, FLOAT_TYPE>(exp) );
			out_dist.normalize();

			if ( sample ){
				std::discrete_distribution<int> char_dist(out_dist.data(),out_dist.data()+out_dist.size());
				ch = char_dist(rnd_gen);
			}else{
				out_dist.maxCoeff(&ch);
			}

			std::cout << char(ch);

			input_char.setZero();
			input_char(ch) = 1.0;
			out_dist = rnn.forwardpass_step_output(input_char);

		}

		std::cout << std::endl;
	}else{
		std::cout << desc << std::endl;
		return 1;
	}

	return 0;
}