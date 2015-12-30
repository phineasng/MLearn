/*!
* 	\brief 		MLearn library structures.
*	\details 	In this header file, the basic structures 
*				that will be used in the algorihtm are defined.
*				They are mainly wrappers and extension to the
*				Eigen library structures.
*	\author		phineasng
*/


#ifndef MLEARN_CORE_BASE_TYPES_INCLUDED
#define MLEARN_CORE_BASE_TYPES_INCLUDED

#include <Eigen/Core>

namespace MLearn{	

	// Matrix typedefs
	template< typename ScalarType >
	using MLMatrix = Eigen::Matrix< ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor | Eigen::AutoAlign >;
	typedef MLMatrix< double > 		MLMatrixd;
	typedef MLMatrix< float >   	MLMatrixf;
	typedef MLMatrix< short >  		MLMatrixs;
	typedef MLMatrix< int >  		MLMatrixi;
	typedef MLMatrix< long >  		MLMatrixl;
	typedef MLMatrix< long long > 	MLMatrixll;   

	// Matrix typedefs (col majors)   
	template< typename ScalarType >
	using MLMatrixRowMajor = Eigen::Matrix< ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >;
	typedef MLMatrixRowMajor< double > 		MLMatrixRowMajord;
	typedef MLMatrixRowMajor< float >   	MLMatrixRowMajorf;
	typedef MLMatrixRowMajor< short >  		MLMatrixRowMajors;
	typedef MLMatrixRowMajor< int >  		MLMatrixRowMajori;
	typedef MLMatrixRowMajor< long >  		MLMatrixRowMajorl;
	typedef MLMatrixRowMajor< long long > 	MLMatrixRowMajorll;   

	// Vector typedefs
	template< typename ScalarType >
	using MLVector = Eigen::Matrix< ScalarType, Eigen::Dynamic, 1, Eigen::ColMajor | Eigen::AutoAlign >;
	typedef MLVector< double > 		MLVectord;
	typedef MLVector< float >   	MLVectorf;
	typedef MLVector< short >  		MLVectors;
	typedef MLVector< int >  		MLVectori;
	typedef MLVector< long >  		MLVectorl;
	typedef MLVector< long long > 	MLVectorll;   

	// Row Vector typedefs   
	template< typename ScalarType >
	using MLRowVector = Eigen::Matrix< ScalarType, 1, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >;
	typedef MLRowVector< double > 		MLRowVectord;
	typedef MLRowVector< float >   		MLRowVectorf;
	typedef MLRowVector< short >  		MLRowVectors;
	typedef MLRowVector< int >  		MLRowVectori;
	typedef MLRowVector< long >  		MLRowVectorl;
	typedef MLRowVector< long long > 	MLRowVectorll;   

}
#endif