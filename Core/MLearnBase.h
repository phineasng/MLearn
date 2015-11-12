/*!
* 	\brief 		MLearn library structures.
*	\details 	In this header file, the basic structures 
*				that will be used in the algorihtm are defined.
*				They are mainly wrappers and extension to the
*				Eigen library structures.
*/

#include <Eigen/Core>

// 

// Matrix typedefs
typedef Eigen::Matrix< double, Dynamic, Dynamic, RowMajor | AutoAlign > 	MLMatrixd;
typedef Eigen::Matrix< float, Dynamic, Dynamic, RowMajor | AutoAlign >  	MLMatrixf;
typedef Eigen::Matrix< short, Dynamic, Dynamic, RowMajor | AutoAlign > 		MLMatrixs;
typedef Eigen::Matrix< int, Dynamic, Dynamic, RowMajor | AutoAlign > 		MLMatrixi;
typedef Eigen::Matrix< long, Dynamic, Dynamic, RowMajor | AutoAlign > 		MLMatrixl;
typedef Eigen::Matrix< long long, Dynamic, Dynamic, RowMajor | AutoAlign > 	MLMatrixll;   
template< typename ScalarType >
using MLMatrix = Eigen::Matrix< ScalarType, Dynamic, Dynamic, RowMajor | AutoAlign >;