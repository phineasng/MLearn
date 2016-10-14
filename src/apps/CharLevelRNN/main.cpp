#include <MLearn/Core>
#include <MLearn/Optimization/AdaGrad.h>
#include <MLearn/Utility/DataInterface/SequentialDataRef.h>
#include <MLearn/NeuralNets/RecurrentNets/RecurrentCellType.h>
#include <MLearn/NeuralNets/RecurrentNets/RecurrentLayer.h>
#include <MLearn/NeuralNets/RecurrentNets/SimpleRNNCost.h>
#include <MLearn/NeuralNets/ActivationFunction.h>

#include <iostream>
#include <streambuf>
#include <fstream>
#include <string>
#include <random>
#include <chrono>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "common_funcs.hpp"

uint N_EPOCHS = 100000;
uint MAX_ITER_PER_EPOCH = 10000000;
FLOAT_TYPE master_learning_rate = 0.01;
FLOAT_TYPE GRADIENT_TOLERANCE = 1e-15;
int MAX_SEQUENCE_LENGTH = 100;

typedef MLearn::Utility::DataInterface::SequentialDataRef<FLOAT_TYPE,INDEX_TYPE> SeqData;

bool ASCIITextToCharOneHotEncoder( const std::string& file_name, MLearn::MLMatrix<FLOAT_TYPE>& _one_hot_char ){

	std::ifstream in(file_name);
	if ( in.fail() ){
		return false;
	}

	// Preallocate some space for the matrix
	in.seekg(0, std::ios::end);
	size_t size = in.tellg();
	in.seekg(0);
	_one_hot_char.resize(NEncodedChars,size);
	_one_hot_char.setZero();

	int curr_iter = 0;
	for ( auto it = std::istreambuf_iterator<char>(in); it != std::istreambuf_iterator<char>(); ++it ){
		_one_hot_char(int(*it),curr_iter) = 1.0;
		++curr_iter;
	}

	std::cout << std::endl << std::endl;

	in.close();
	return true;

}


int main(int ac, char* av[]){

	//
	using namespace MLearn;
	namespace po = boost::program_options;

	// program options
	po::options_description desc("Character level RNN.\nBasic usage: ./CharLevelRNN --text_file <path_to_text_file>");
	desc.add_options()
    	("help", "See all the available options.")
    	("text_file", po::value<std::string>(), "Path to an ASCII text file to be used for training.")
    	("hidden_size", po::value<uint>(), "Size of the hidden state of the RNN (default: 128).")
    	("weights_file", po::value<std::string>(), "File containing the weights of a previously trained model (warning: the model has to have the same hidden dimension).")
    	("out_path", po::value<std::string>(), "Path to directory where to store the learned weights (default: current directory).\nThe directory should not terminate with a slash.")
    	("save_every", po::value<uint>(), "Weights will be saved every <save_every> epochs (default: 10).")
    	("data_per_epoch", po::value<uint>(), "How many data samples to create for each epoch (default: 50000).")
    	("data_refresh_rate", po::value<uint>(), "Dataset will be refreshed every <rate> iterations (default: 100).")
    	("batch_size", po::value<uint>(), "Batch size for training (default: 10).");

	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	if (vm.count("help")){
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("text_file")){

		// Default parameters
		uint hidden_size = 128u;
		boost::filesystem::path full_path( boost::filesystem::current_path() );
		std::string out_dir = full_path.string();
		uint save_every = 10u;
		uint batch_size = 10u;
		uint data_per_epoch = 50000u;
		uint data_refresh_rate = 100u;

		// Modify default parameters if necessary
		if (vm.count("hidden_size")){
			hidden_size = vm["hidden_size"].as<uint>();
			std::cout << "User-defined hidden_size: " << hidden_size << std::endl;
		}
		if (vm.count("out_path")){
			out_dir = vm["out_path"].as<std::string>();
			std::cout << "User-defined output directory: " << out_dir << std::endl;
		}
		if (vm.count("save_every")){
			save_every = vm["save_every"].as<uint>();
			std::cout << "User-defined save_every: " << save_every << std::endl;
		}
		if (vm.count("batch_size")){
			batch_size = vm["batch_size"].as<uint>();
			std::cout << "User-defined batch_size: " << batch_size << std::endl;
		}
		if (vm.count("data_per_epoch")){
			data_per_epoch = vm["data_per_epoch"].as<uint>();
			std::cout << "User-defined data_per_epoch: " << data_per_epoch << std::endl;
		}
		if (vm.count("data_refresh_rate")){
			data_refresh_rate = vm["data_refresh_rate"].as<uint>();
			std::cout << "User-defined data_refresh_rate: " << data_refresh_rate << std::endl;
		}



		// output filename template
		std::string out_file = out_dir + "/out_weights_hs" + std::to_string(hidden_size) + "_";
		std::string out_file_ext = ".txt"; 

		// Import text file
		MLMatrix<FLOAT_TYPE> char_matrix;
		std::cout << "Reading text file." << std::endl;
		bool read_success = ASCIITextToCharOneHotEncoder(vm["text_file"].as<std::string>(),char_matrix);
		if (!read_success){
			std::cerr << "Error while reading the text file." << std::endl;
			return 1;
		}
		uint text_length = char_matrix.cols();
		if (text_length < 2){
			std::cerr << "Please use a longer text." << std::endl;
		}

		// Create model
		RNNLayer rnn(NEncodedChars,hidden_size,NEncodedChars);
		// -- Set weights
		uint size_weights = rnn.getNWeights();
		srand((unsigned int) time(0));
		MLVector<FLOAT_TYPE> weights;
		if (vm.count("weights_file")){
			weightsFromFile(vm["weights_file"].as<std::string>(),weights,size_weights);
		}else{
			weights.resize(size_weights);
			weights = 0.5*MLVector<FLOAT_TYPE>::Random( size_weights );
		}
		rnn.attachWeightsToCell(weights);

		// Setup optimization algorithm
		MLearn::Optimization::AdaGrad< MLearn::Optimization::LineSearchStrategy::FIXED,FLOAT_TYPE,INDEX_TYPE,2 > minimizer;
		minimizer.setNSamples(data_per_epoch);
		minimizer.setSizeBatch(batch_size);
		minimizer.setTolerance(GRADIENT_TOLERANCE);
		minimizer.setMaxIter(MAX_ITER_PER_EPOCH);
		minimizer.setMaxEpoch(1);	
		MLearn::Optimization::LineSearch<MLearn::Optimization::LineSearchStrategy::FIXED,FLOAT_TYPE,INDEX_TYPE> strategy(master_learning_rate);
		minimizer.setLineSearchMethod(strategy);

		// create random number generator for the dataset creation
		std::mt19937_64 rnd_gen( std::chrono::system_clock::now().time_since_epoch().count() );
		std::uniform_int_distribution<int> start_point_generator(0,text_length-2);
		std::uniform_int_distribution<INDEX_TYPE> seq_length_generator(0,MAX_SEQUENCE_LENGTH);
		// pre allocation for cost function
		MLVector<FLOAT_TYPE> grad_tmp( size_weights );
		MLMatrix<FLOAT_TYPE> grad_out( NEncodedChars,MAX_SEQUENCE_LENGTH );
			
		MLMatrix<INDEX_TYPE> info(SeqData::n_info_fields,data_per_epoch);
		info.setZero();

		for ( uint iter = 0; iter < N_EPOCHS; ++iter ){

			if ( !( iter % data_refresh_rate ) ){
				// Create sequential data - basically each epoch a new dataset is created
				for ( uint sample_id = 0; sample_id < data_per_epoch; ++sample_id ){
					INDEX_TYPE start_id = start_point_generator(rnd_gen);

					info(0,sample_id) = start_id;	// starting position of the sequence in the text
					info(1,sample_id) = (uint)std::min<int>( seq_length_generator(rnd_gen), (int)text_length - (int)start_id - 1 );	// input sequence length
					info(2,sample_id) = start_id + 1; //starting position of the output sequence
					info(3,sample_id) = info(1,sample_id); // output sequence length
					info(4,sample_id) = 0; // delay between input and output
					info(5,sample_id) = 1;	// flag to indicate if it is necessary to reset the hidden state
				}
			}
			SeqData data(char_matrix,char_matrix,info);
			// Create cost function
			MLearn::NeuralNets::RecurrentNets::SimpleRNNCost<loss,RNNLayer,SeqData> cost(data,rnn,grad_tmp,grad_out);

			minimizer.minimize(cost,weights);

			// after minimization - if necessary, output the weights to file
			if ( !(iter%save_every) ){
				std::string out_file_name = out_file + std::to_string(iter) + out_file_ext;
				bool write_success = weightsToFile(out_file_name, weights);
				if (!write_success){
					std::cerr << "Error while writing to file." << std::endl;
					return 1;
				}
			}
		}


	}else{
		std::cout << desc << std::endl;
		return 1;
	}



	return 0;

}