#ifndef MLEARN_SEQUENTIAL_DATA_INTERFACE
#define MLEARN_SEQUENTIAL_DATA_INTERFACE

// MLearn includes
#include <MLearn/Core>

namespace MLearn{

	namespace Utility{

		namespace DataInterface{

			template < 	typename ScalarType,
						typename IndexType,
						typename DERIVED_DATA_INTERFACE >
			class SequentialDataInterface{
			public:
				const Eigen::Ref< const MLMatrix<ScalarType> > getInput( IndexType idx ) const{
					return static_cast<const DERIVED_DATA_INTERFACE*>(this)->getInput(idx);
				}
				const Eigen::Ref< const MLMatrix<ScalarType> > getOutput( IndexType idx ) const{
					return static_cast<const DERIVED_DATA_INTERFACE*>(this)->getOutput(idx);
				}
				size_t getNSamples() const{
					return info.cols();
				}
				IndexType getDelay( IndexType idx ) const{
					return info(delay_idx,idx);
				}
				IndexType getNOutputSteps( IndexType idx ) const{
					return info(n_outputs_idx,idx);
				}
				bool getReset( IndexType idx ) const{
					return bool(info(reset_idx,idx));
				}
				void setInfo( const Eigen::Ref< const MLMatrix<IndexType> > refInfo ){
					info.resize(refInfo.rows(),refInfo.cols());
					info = refInfo;
				}
			protected:
				SequentialDataInterface(const MLMatrix< IndexType >& _info): info(_info){}
				// info contains the information to parse the dataset
				// it's a matrix 4xN, where N is the number of samples
				// the 0th element indicates the start, in the inputs data, of the sample point
				// the first (row) element indicates the number of inputs elements to consider for the sequence
				// the second indicates the index of the first output element in the output matrix
				// the third is the number of data points in the output sequence
				// the 4th element is the delay between the number of timesteps to wait before starting to output
				// the 5th element indicates if the hidden state, if any, has to be reset
				// example:
				// 		info.col(7) = (5,3,6,2,4)
				//	then,
				//		the input sequence of the 7th sample point starts from column 5 of the input dataset
				//		3 elements are taken for the input sequence
				//		after 2 timesteps of the sequence (e.g. forward step for a RNN), the 
				//			model is expected to start to output the data points from the 6th column of the output dataset
				//	i.e.
				// 		t = 0:   input[5]  ->  
				// 		t = 1:   input[6]  -> 	   
				// 		t = 2:   input[7]  ->  
				// 		t = 3:   		   ->  				(in a RNN, this timestep is considering only his hidden state)
				// 		t = 4:   		   ->	output[6]	  
				// 		t = 5:   		   ->  	output[7]
				// 		
				MLMatrix< IndexType > info;
				const IndexType start_in_idx = 0;
				const IndexType n_inputs_idx = 1;
				const IndexType start_out_idx = 2;
				const IndexType n_outputs_idx = 3;
				const IndexType delay_idx = 4;
				const IndexType reset_idx = 5;
			public:
				static const IndexType n_info_fields;
			};

			template < 	typename ScalarType,
						typename IndexType,
						typename DERIVED_DATA_INTERFACE >
			const IndexType SequentialDataInterface<ScalarType,IndexType,DERIVED_DATA_INTERFACE>::n_info_fields = IndexType(6);

		}

	}

}

#endif