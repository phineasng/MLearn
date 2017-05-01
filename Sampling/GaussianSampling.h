#ifndef MLEARN_GAUSSIAN_SAMPLING_ROUTINES_H_FILE
#define MLEARN_GAUSSIAN_SAMPLING_ROUTINES_H_FILE

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

// MLearn includes
#include <MLearn/Core>

// Boost includes
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

// STL includes
#include <type_traits>
#include <chrono>
#include <functional>

namespace MLearn{
namespace Sampling{
namespace Gaussian{

	/*!	\typedef
	*	\brief 	typedef for normal distribution pdf
	*/
	template <typename FLOAT_TYPE>
	using uni_gaussian_pdf = boost::normal_distribution<FLOAT_TYPE>;

	/** @enum
	*	\brief Methods to be used to compute samples from uncorrelated
	*		   multivariate gaussian
	*/
	enum class TransformMethod{
		/** using Cholesky factorization */
		CHOL, 
		/** using SV decomposition */
		EIGENSOLVER
	};

namespace SamplingImpl{

	/*!
		\brief 	Generate seed using current system time
	*/
	uint seed_from_time(){
		return std::chrono::system_clock::now().time_since_epoch().count();
	}

	/*!	\typedef
	*	\brief 	typedef for internal pseudo random numbers generator
	*/
	typedef boost::random::mt19937 RNG_TYPE;

	/*!
	*	\brief	Struct containing the type of the decomposer type
	*/
	template < TransformMethod TM >
	struct SolverTraits{};

	// Cholesky specialization
	template<>
	struct SolverTraits<TransformMethod::CHOL>{
		template < typename Scalar >
		using STYPE = Eigen::LLT< MLMatrix<Scalar> >;
	};

	// EIGENSOLVER specialization
	template<>
	struct SolverTraits<TransformMethod::EIGENSOLVER>{
		template < typename Scalar >
		using STYPE = Eigen::SelfAdjointEigenSolver< MLMatrix<Scalar> >;
	};

	/*!
	*	\brief 	 util function to get the transformation from an already
	*			 decomposed matrix
	*	\details from a matrix decomposed with cholesky 
	*			 (as provided by the eigen library)
	*	\param 	 transform transformation to be computed
	*	\param 	 solver LLT solver
	*/
	template < typename DERIVED1, typename DERIVED2 >
	inline void transform_from_decomposition(
				Eigen::MatrixBase<DERIVED1>& transform,
				const Eigen::LLT<DERIVED2>& solver)
	{
		MLEARN_WARNING(solver.info() == Eigen::Success,
			"Decomposition inaccurate!");
		transform = solver.matrixL();
	}

	/*!
	*	\brief 	 util function to get the transformation from an already
	*			 decomposed matrix
	*	\details from a matrix decomposed with robust cholesky 
	*			 (as provided by the eigen library)
	*			 compute the necessary transformation
	*	\param 	 transform transformation to be computed
	*	\param 	 solver SelfAdjointEigenSolver solver
	*/
	template < typename DERIVED1, typename DERIVED2 >
	inline void transform_from_decomposition(
				Eigen::MatrixBase<DERIVED1>& transform,
				const Eigen::SelfAdjointEigenSolver<DERIVED2>& solver)
	{
		MLEARN_WARNING(solver.info() == Eigen::Success,
			"Decomposition inaccurate!");
		typedef typename DERIVED1::Scalar Scalar;
		transform.fill(0);
		transform.diagonal() = 
			solver.eigenvalues().unaryExpr(
				std::function<Scalar(Scalar)>(round_to_zero<Scalar>)).cwiseSqrt();
		std::cout << transform.diagonal() << std::endl;
		transform = solver.eigenvectors()*transform;
	}

	/*!
	*	\brief 	util function to get the transformation to apply to uncorrelated
	*			normal samples to obtain samples with the given covariance
	*	\param 	transform transformation to be computed
	*	\param 	covariance covariance matrix
	*/
	template < TransformMethod TM, typename DERIVED1, typename DERIVED2>
	inline void transform_from_covariance(
			Eigen::MatrixBase<DERIVED1>& transform,
			const Eigen::MatrixBase<DERIVED2>& covariance){
		typename SolverTraits<TM>::template STYPE< typename DERIVED2::Scalar > 
			cholesky(covariance);
		transform_from_decomposition(transform, cholesky);
	}

	/*!
	*	\brief	Given the mean and the linear transformation, map standard
	*			gaussian samples to samples from the target gaussian
	*	\param 	mean mean of the target distribution
	*	\param 	transform linear transformation to be applied. Note that 
				transform.transpose()*transform should give the covariance 
				matrix of the target distribution
	*	\param  samples samples to be transformed, the transformed sample will 
				be stored in this same matrix
	*/
	template <typename DERIVED1, typename DERIVED2, typename DERIVED3>
	inline void transform_gaussian_samples_with_transform(
			const Eigen::MatrixBase<DERIVED1>& mean,
			const Eigen::MatrixBase<DERIVED2>& transform,
			Eigen::MatrixBase<DERIVED3>& samples){
		MLEARN_ASSERT(mean.rows() == samples.rows(), 
			"Samples must have the same dimension of the mean.");
		MLEARN_ASSERT(transform.rows() == transform.cols(),
			"The transformation must be a square matrix!");
		MLEARN_ASSERT(mean.rows() == transform.rows(),
			"Dimensions of mean and covariance must be consistent!");
		samples = transform*samples;
		samples.colwise() += mean;
	}

	/*!
	*	\brief	Given the mean and the covariance, map standard
	*			gaussian samples to samples from the target gaussian
	*	\param 	mean mean of the target distribution
	*	\param 	covariance covariance of the target gaussian distribution
	*	\param  samples samples to be transformed, the transformed sample will 
				be stored in this same matrix
	*/
	template <TransformMethod TM,
	          typename DERIVED1, 
	          typename DERIVED2, 
	          typename DERIVED3>
	inline void transform_gaussian_samples_with_covariance(
			const Eigen::MatrixBase<DERIVED1>& mean,
			const Eigen::MatrixBase<DERIVED2>& covariance,
			Eigen::MatrixBase<DERIVED3>& samples){
		MLEARN_ASSERT(mean.rows() == samples.rows(), 
			"Samples must have the same dimension of the mean.");
		MLEARN_ASSERT(covariance.rows() == covariance.cols(),
			"The covariance must be a square matrix!");
		MLEARN_ASSERT(mean.rows() == covariance.rows(),
			"Dimensions of mean and covariance must be consistent!");
		MLMatrix<typename DERIVED1::Scalar> transform = 
			MLMatrix<typename DERIVED1::Scalar>::Zero(
				covariance.rows(),
				covariance.cols());
		transform_from_covariance<TM>(transform, covariance);
		samples = transform*samples;
		samples.colwise() += mean;
	}

	/*!
	*	\brief	Helper functor for matrix initialization using i.i.d. standard 
				gaussian
	*	\tparam Scalar floating point type
	*	\tparam RNG random number generator type
	*/
	template < typename Scalar, typename RNG >
	class GaussianInitializer{
		static_assert(std::is_floating_point<Scalar>::value, 
			"First template parameter must be floating point!");
		RNG& _rng;
		mutable uni_gaussian_pdf<Scalar> _pdf;
	public:
		/*!
		*	\brief Constructor
		*	\param rng Random number generator
		*/
		GaussianInitializer(RNG& rng): _rng(rng) {}
		/*!
		*	\brief operator to be called
		*/
		Scalar operator()(Eigen::Index row, Eigen::Index col) const{
			return _pdf(_rng);
		}
	};

} // end of SamplingImpl namespace

/*!
*	\brief 	Sample uncorrelated gaussian samples. Version requiring an RNG.
*	\tparam Scalar Floating point type
*	\param 	dim dimension of the samples to produce
*	\param 	N number of samples to produce
*	\param  rng random number generator
*/
template< typename Scalar, typename RNG >
MLMatrix<Scalar> sample_standard_gaussian(uint dim, uint N, RNG& rng){
	SamplingImpl::GaussianInitializer<Scalar, RNG> initializer(rng);
	return MLMatrix<Scalar>::NullaryExpr(dim, N, initializer);
}

/*!
*	\brief 	Sample uncorrelated gaussian samples. 
*	\tparam Scalar Floating point type
*	\param 	dim dimension of the samples to produce
*	\param 	N number of samples to produce
*/
template< typename Scalar >
MLMatrix<Scalar> sample_standard_gaussian(uint dim, uint N){
	SamplingImpl::RNG_TYPE rng(SamplingImpl::seed_from_time());
	return sample_standard_gaussian<Scalar>(dim, N, rng);
}

/*!
*	\brief Class to produce samples from a gaussian distribution specified
		   by mean and covariance.
	\details This class can be used in two ways. If instantiated, it can be used
			 to sample without decomposing the covariance everytime. Otherwise,
			 it provides a static method that can be used to sample.

*/
template <typename Scalar>
class MultivariateGaussian{
private:
	/*! mean of the distribution*/
	MLVector<Scalar> _mean;
	/*! covariance matrix*/
	MLMatrix<Scalar> _covariance; 
	/*! transform computed from the covariance matrix*/
	MLMatrix<Scalar> _transform; 
	/*! \brief Helper function to compute the linear transformation*/
	template <bool ADAPTIVE = true, TransformMethod TM = TransformMethod::CHOL>
	inline static typename std::enable_if<ADAPTIVE, void>::type 
		_compute_linear_transform(
			const Eigen::Ref<const MLMatrix<Scalar>>& covar,
			MLMatrix<Scalar>& transform){
		Eigen::LLT< MLMatrix<Scalar> > cholesky(covar);
		if (cholesky.info() == Eigen::Success){
			SamplingImpl::transform_from_decomposition(transform, cholesky);
		}else{
			Eigen::JacobiSVD< MLMatrix<Scalar> > svd(covar, Eigen::ComputeThinU | Eigen::ComputeThinV);
			std::cout << svd.singularValues() << std::endl; 
			std::cout << svd.matrixU() << std::endl; 
			std::cout << std::endl;
			std::cout << svd.matrixV() << std::endl; 
			Eigen::SelfAdjointEigenSolver< MLMatrix<Scalar> > eigen(covar);
			std::cout << eigen.eigenvalues() << std::endl; 
			SamplingImpl::transform_from_decomposition(transform, eigen);
		}
	}
	template <bool ADAPTIVE, TransformMethod TM>
	inline static typename std::enable_if<!ADAPTIVE, void>::type 
		_compute_linear_transform(
			const Eigen::Ref<const MLMatrix<Scalar>>& covar,
			MLMatrix<Scalar>& transform){
		SamplingImpl::transform_from_covariance<TM>(transform, covar);
	}
	mutable SamplingImpl::RNG_TYPE rng;
public:
	/*!
		\brief Default constructor 
	*/
	MultivariateGaussian(): rng(SamplingImpl::seed_from_time()) {}
	/*!
		\brief Args constructor
		\param mean mean of the distribution
		\param covar covariance of the distribution 
	*/
	template <bool ADAPTIVE = true, 
			  TransformMethod TM = TransformMethod::CHOL>
	MultivariateGaussian(
			const Eigen::Ref<const MLVector<Scalar>>& mean,
			const Eigen::Ref<const MLMatrix<Scalar>>& covar): 
		_mean(mean), _covariance(covar), _transform(covar.rows(), covar.cols()),
		rng(SamplingImpl::seed_from_time()){
			MLEARN_ASSERT(covar.rows() == mean.rows(), 
				"Dimensions must be consistent!");
			MLEARN_ASSERT(covar.rows() == covar.cols(), 
				"Covariance must be a square matrix!");
		_compute_linear_transform<ADAPTIVE, TM>(covar, _transform);
	}
	/*!
		\brief Copy constructor
		\param mg const reference to a multivariate gaussian
	*/
	MultivariateGaussian(const MultivariateGaussian<Scalar>& mg):
		_mean(mg._mean), _covariance(mg._covariance), _transform(mg._transform),
		rng(SamplingImpl::seed_from_time()) {}
	/*!
		\brief Move constructor
		\param mg const reference to a multivariate gaussian
	*/
	MultivariateGaussian(MultivariateGaussian<Scalar>&& mg):
		_mean(std::move(mg._mean)), _covariance(std::move(mg._covariance)), 
		_transform(std::move(mg._transform)),
		rng(SamplingImpl::seed_from_time()) {}
	/*!
		\brief Copy assignment
	*/
	MultivariateGaussian<Scalar>& 
			operator=(const MultivariateGaussian<Scalar>& mg){
		_mean = mg._mean;
		_covariance = mg._covariance;
		_transform = mg._transform;
		return (*this);
	}
	/*!
		\brief Move assignment
	*/
	MultivariateGaussian<Scalar>& 
			operator=(MultivariateGaussian<Scalar>&& mg){
		_mean = std::move(mg._mean);
		_covariance = std::move(mg._covariance);
		_transform = std::move(mg._transform);
		return (*this);
	}
	/*!
		\brief Destructor
	*/
	~MultivariateGaussian() = default;
	/*!
		\brief Static version of sampling
		\tparam ADAPTIVE (bool) if use an adaptive strategy for the 
				decomposition
		\tparam TM if not adaptive, use the specified decomposition method
		\param mean mean of the distribution
		\param covariance covariance of the distribution
		\param N number of samples to produce (default: 100)
	*/
	template <bool ADAPTIVE = true, 
			  TransformMethod TM = TransformMethod::CHOL>
	inline static MLMatrix<Scalar> sample(
			const Eigen::Ref<const MLVector<Scalar>>& mean,
			const Eigen::Ref<const MLMatrix<Scalar>>& covariance,
			int N = 100){

		MLEARN_ASSERT(mean.rows() == covariance.rows(),
			"Dimensions must be consistent!");
		MLEARN_ASSERT(mean.rows() == covariance.cols(),
			"Dimensions must be consistent!");

		MLMatrix<Scalar> transform = 
			MLMatrix<Scalar>::Zero(covariance.rows(), covariance.cols());
		MultivariateGaussian<Scalar>::_compute_linear_transform
			<ADAPTIVE,TM>(
			covariance, transform);
		MLMatrix<Scalar> samples = 
			sample_standard_gaussian<Scalar>(mean.rows(), N);
		SamplingImpl::transform_gaussian_samples_with_transform(
			mean, transform, samples);
		return samples;
	}

	/*!
		\brief Instantiated version of sampling
		\param N number of samples to produce (default: 100)
	*/
	MLMatrix<Scalar> sample(int N = 100){
		MLMatrix<Scalar> samples = 
			sample_standard_gaussian<Scalar>(_mean.rows(), N);
		SamplingImpl::transform_gaussian_samples_with_transform(
			_mean, _transform, samples);
		return samples;
	}

	/*!
		\brief Set distribution parameters
	*/
	template <bool ADAPTIVE = true, 
			  TransformMethod TM = TransformMethod::CHOL>
	void set_distribution(
			const Eigen::Ref<const MLVector<Scalar>>& mean,
			const Eigen::Ref<const MLMatrix<Scalar>>& covariance){
		MLEARN_ASSERT(mean.rows() == covariance.rows(),
			"Dimensions must be consistent!");
		MLEARN_ASSERT(mean.rows() == covariance.cols(),
			"Dimensions must be consistent!");
		_mean.resize(mean.rows());
		_mean = mean;
		_covariance.resize(covariance.rows(), covariance.cols());
		_covariance = covariance;
		_transform.resize(covariance.rows(), covariance.cols());
		this->_compute_linear_transform<ADAPTIVE, TM>(covariance, _transform);
	}
	/*!
		\brief Get mean
	*/
	const MLVector<Scalar>& mean() const {return _mean;}
	/*!
		\brief Get covariance
	*/
	const MLMatrix<Scalar>& covariance() const {return _covariance;}
	/*!
		\brief Get transform
	*/
	const MLMatrix<Scalar>& transform() const {return _transform;}
	/*!
		\brief Set mean
	*/
	void set_mean(const Eigen::Ref<const MLVector<Scalar>>& mean){
		MLEARN_ASSERT(mean.size() == _mean.size(),
			"Wrong input mean size! Use set_distribution instead if you want to"
			" change size!");
		_mean = mean;
	}
	/*!
		\brief Set covariance
	*/
	template <bool ADAPTIVE = true, 
			  TransformMethod TM = TransformMethod::CHOL>
	void set_covariance(const Eigen::Ref<const MLMatrix<Scalar>>& covariance){
		MLEARN_ASSERT(covariance.cols() == _covariance.cols(),
			"Wrong input size! Use set_distribution instead if you want to "
			"change size!");
		MLEARN_ASSERT(covariance.rows() == _covariance.rows(),
			"Wrong input size! Use set_distribution instead if you want to "
			"change size!");
		_covariance = covariance;
		this->_compute_linear_transform<ADAPTIVE, TM>(_covariance, _transform);
	}
};

}}}

#endif