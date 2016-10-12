#ifndef MLEARN_DISTANCE_FUNCTION_INCLUDED
#define MLEARN_DISTANCE_FUNCTION_INCLUDED

#include <MLearn/Core>

// Macro for defining new distance function
#define TEMPLATED_DISTANCE_FUNCTION(sample_a,sample_b)\
    template<   typename DERIVED,\
          typename DERIVED_2 >\
 static inline typename DERIVED::Scalar compute(const Eigen::MatrixBase<DERIVED>& sample_a,\
     const Eigen::MatrixBase<DERIVED_2>& sample_b)

namespace MLearn{
  
  namespace Clustering{

    /*!
     *  \brief    SquaredEuclidianDistance
    *   \details  Distance used for clustering, other distance functions can be
    *             implemented based on this template and passed as kernels to
    *             e.g. the KMeans class.
    *   \author   frenaut
    *
    */

    class SquaredEuclidianDistance
    {
    public:
        TEMPLATED_DISTANCE_FUNCTION(a,b)                                         
        {                                                                                    
          static_assert( (DERIVED::ColsAtCompileTime == 1)&&(DERIVED_2::ColsAtCompileTime == 1),
                      "Inputs have to be column vectors");                                           
          static_assert( std::is_floating_point<typename DERIVED::Scalar>::value &&          
                    std::is_same<typename DERIVED::Scalar, typename DERIVED_2::Scalar>::value,     
                          "Scalar types have to be the same and floating point!" );                      
         return (a - b).squaredNorm();                                                      
        }                    

    }; // class SquaredEuclidianDistance

  }// End Clustering namespace

}// End MLearn namespace

#endif
