#ifndef MLEARN_GAUSSIAN_PROCESS_H_FILE
#define MLEARN_GAUSSIAN_PROCESS_H_FILE

// MLearn includes
#include <MLearn/Core>
#include <MLearn/Sampling/GaussianSampling.h>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SVD>

// Boost includes
#include <boost/math/special_functions/erf.hpp>

namespace MLearn{
namespace SP{
namespace GP{

/*!
	\brief Given query covariance and a kernel, compute the covariance 
		   matrix.
    \tparam KERNEL Type of the kernel that exposes a compute() function
*/
template< typename KERNEL, typename DERIVED1, typename DERIVED2 >
static inline void compute_gp_covariance(
		const Eigen::MatrixBase<DERIVED1>& query_pts,
		const KERNEL& kernel,
		Eigen::MatrixBase<DERIVED2>& covariance){
	MLEARN_ASSERT(covariance.rows() == covariance.cols(),
		"Covariance must be a square matrix!");
	MLEARN_ASSERT(query_pts.cols() == covariance.cols(),
		"Covariance size must be the same as the number of query points!");

	for (int j = 0; j < covariance.cols(); ++j){
		for (int i = 0; i <= j; ++i){
			covariance(i,j) = 
				kernel.compute(query_pts.col(i), query_pts.col(j));
			covariance(j,i) = covariance(i,j);
		}
	}
}

/*!
	\brief Class to sample a gaussian process based on the covariance. 
	\details This class can be used in two ways. It provides a static 
			 method to sample a GP given mean function, kernel object and 
			 query_pts. Otherwise, this class can be instantiated and will
			 hold a copy of the mean and the covariance in the underlying 
			 sampler. 
			 This is useful for repeated sampling.
	\tparam Scalar Underlying numeric type
*/
template<typename Scalar>
class GaussianProcess{
private:
	/*! underlying gaussian sampler*/
	Sampling::Gaussian::MultivariateGaussian<Scalar> _sampler;
	static constexpr Scalar _sqrt_2 = 
	Scalar(1.41421356237309504880168872420969807856967187537694807317667973799);
	/*!
		\brief Compute covariance given query points and kernel. 
			   Basically a wrapper function around compute_gp_covariance.
		\params K kernel to be used
		\params query_pts points to use to compute the covariance
	*/
	template <typename KERNEL>
	static inline MLMatrix<Scalar> _compute_covariance(
			const Eigen::Ref<const MLMatrix<Scalar>>& query_pts,
			const KERNEL& K){
		int N = query_pts.cols();
		MLMatrix<Scalar> covariance = MLMatrix<Scalar>::Zero(N, N);
		compute_gp_covariance(query_pts, K, covariance);
		return covariance;
	}
public:
	/*!
		\brief Default constructor
	*/
	GaussianProcess() = default;
	/*!
		\brief Constructor with mean and covariance
	*/
	GaussianProcess(const Eigen::Ref<const MLVector<Scalar>>& mean,
	   const Eigen::Ref<const MLMatrix<Scalar>>& covariance):
		_sampler(mean, covariance){}
	/*!
		\brief Copy constructor
	*/
	GaussianProcess(const GaussianProcess<Scalar>& ref_gp): 
		_sampler(ref_gp._sampler){}
	/*!
		\brief Move constructor
	*/
	GaussianProcess(GaussianProcess<Scalar>&& ref_gp): 
		_sampler(std::move(ref_gp._sampler)){}
	/*!
		\brief Destructor
	*/
	~GaussianProcess() = default;
	/*!
		\brief Copy assignment
	*/
	GaussianProcess<Scalar>& operator=(
			const GaussianProcess<Scalar>& ref_gp){
		_sampler = ref_gp._sampler;
		return (*this);
	}
	/*!
		\brief Move assignment
	*/
	GaussianProcess<Scalar>& operator=(
			GaussianProcess<Scalar>&& ref_gp){
		_sampler = std::move(ref_gp._sampler);
		return (*this);
	}
	/*!
		\brief Get mean
	*/
	const MLVector<Scalar>& mean() const{
		return _sampler.mean();
	}
	/*!
		\brief Get covariance
	*/
	const MLMatrix<Scalar>& covariance() const{
		return _sampler.covariance();
	}
	/*!
		\brief Set mean
	*/
	void set_mean(const Eigen::Ref<const MLVector<Scalar>>& mean){
		_sampler.set_mean(mean);
	}
	/*!
		\brief Set covariance
	*/
	void set_covariance(const Eigen::Ref<const MLMatrix<Scalar>>& covariance){
		_sampler.set_covariance(covariance);
	}
	/*!
		\brief Set distribution
	*/
	void set_distribution(
			const Eigen::Ref<const MLVector<Scalar>>& mean,
			const Eigen::Ref<const MLMatrix<Scalar>>& covariance){
		_sampler.set_distribution(mean, covariance);
	}
	/*!
		\brief Set covariance given query points and kernel
	*/
	template <typename KERNEL>
	void set_covariance(
		const Eigen::Ref<const MLMatrix<Scalar>>& query_pts,
		const KERNEL& K){
		MLMatrix<Scalar> covariance = this->_compute_covariance(query_pts, K);
		_sampler.set_covariance(covariance);
	}
	/*!
		\brief Sample function
	*/
	MLMatrix<Scalar> sample(int N = 10){
		return _sampler.sample(N);
	}
	/*!
		\brief Static sample function
	*/
	template <typename KERNEL>
	inline static MLMatrix<Scalar> sample(
			const Eigen::Ref<const MLVector<Scalar>>& mean,
			const Eigen::Ref<const MLMatrix<Scalar>>& query_pts,
			const KERNEL& K,
			int N = 10){
		MLMatrix<Scalar> covariance = 
			GaussianProcess<Scalar>::_compute_covariance(query_pts, K);
		return Sampling::Gaussian::MultivariateGaussian<Scalar>::sample(
			mean, covariance, N);
	}
	/*!
		\brief Compute confidence intervals
		\param p confidence probability
		\return a 2xN matrix containing the confidence intervals. First row is 
				lower bound, second row is upper bound
	*/
	MLMatrix<Scalar> confidence_interval(const Scalar& p){
		MLMatrix<Scalar> intervals(2, _sampler.mean().rows());
		intervals.row(1) = _sampler.covariance().diagonal().cwiseSqrt();
		intervals.row(1) *= _sqrt_2*boost::math::erf_inv(p);
		intervals.row(0) = -intervals.row(1);
		intervals.rowwise() += _sampler.mean().transpose();
		return intervals;
	}
};


}}}

#endif