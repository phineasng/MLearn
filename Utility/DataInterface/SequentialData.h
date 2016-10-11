#ifndef MLEARN_SEQUENTIAL_DATA_H
#define MLEARN_SEQUENTIAL_DATA_H

// MLearn includes
#include <MLearn/Core>
#include "SequentialDataInterface.h"

namespace MLearn{

	namespace Utility{

		namespace DataInterface{

			template < 	typename ScalarType,
						typename IndexType >
			class SequentialData: public SequentialDataInterface<ScalarType,IndexType,SequentialData<ScalarType,IndexType>> {
			public:
				typedef ScalarType Scalar;
				typedef IndexType Index;
				typedef SequentialDataInterface<ScalarType,IndexType,SequentialData<ScalarType,IndexType>> Interface;
			public:
				SequentialData(const MLMatrix<ScalarType>& _input,const MLMatrix<ScalarType>& _output,const Eigen::Ref< const MLMatrix<IndexType> > _info):
					Interface(_info),
					input_data_(_input),
					output_data_(_output)
				{}
				const Eigen::Ref< const MLMatrix<ScalarType> > getInput( IndexType idx ) const{
					return input_data_.block(0,Interface::info(Interface::start_in_idx,idx),input_data_.rows(),Interface::info(Interface::n_inputs_idx,idx));
				}
				const Eigen::Ref< const MLMatrix<ScalarType> > getOutput( IndexType idx ) const{
					return output_data_.block(0,Interface::info(Interface::start_out_idx,idx),output_data_.rows(),Interface::info(Interface::n_outputs_idx,idx));
				}
			private:
				MLMatrix<ScalarType> input_data_;
				MLMatrix<ScalarType> output_data_;
			};

		}

	}

}

#endif