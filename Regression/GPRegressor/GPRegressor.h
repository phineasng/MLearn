#ifndef GP_REGRESSOR_H_FILE_MLEARN
#define GP_REGRESSOR_H_FILE_MLEARN

// MLearn includes
#include <MLearn/Core>
#include <MLearn/StochasticProcess/GaussianProcess/GP.h>

// Eigen includes
#include <Eigen/Core>

namespace MLearn{
namespace Regression{

/*!
	\brief Model for regression using gaussian processes
	\tparam K kernel of the gaussian process
	\tparam Scalar floating point type to use
	\details This class will fit the posterior
*/
template < typename K, typename Scalar >
class GPRegressor{
private:
	/*! \brief Training target */
	MLVector<Scalar> _y_train;
	/*! \brief Training predictors */
	MLMatrix<Scalar> _X_train;
	/*! \brief Covariance of the training data */
	MLMatrix<Scalar> _covariance_train;
	/*! \brief Gaussian process object used to sample from fitted posterior */
	SP::GP::GaussianProcess<Scalar> _gp;
	/*! \brief Kernel to be used */
	K _kernel;
	/*!	\brief Flag to indicate if it was fitted at least once */
	bool _fitted = false;
	/*!	\brief Flag to indicate if functions can be sampled from the posterior*/
	bool _has_posterior = false;
public:
	/*!
		\brief Default empty constructor
	*/
	GPRegressor() = default;
	/*!
		\brief Constructor with kernel
	*/
	GPRegressor(const K& kernel): _kernel(kernel) {}
	/*!
		\brief Copy constructor
	*/
	GPRegressor(const GPRegressor<K, Scalar>& ref_gpr): 
		_y_train(ref_gpr._y_train),
		_X_train(ref_gpr._X_train),
		_covariance_train(ref_gpr._covariance_train),
		_gp(ref_gpr._gp),
		_kernel(ref_gpr._kernel),
		_fitted(ref_gpr._fitted),
		_has_posterior(ref_gpr._has_posterior)
	{}
	/*!
		\brief Move constructor
	*/
	GPRegressor(GPRegressor<K, Scalar>&& ref_gpr): 
		_y_train(std::move(ref_gpr._y_train)),
		_X_train(std::move(ref_gpr._X_train)),
		_covariance_train(std::move(ref_gpr._covariance_train)),
		_gp(std::move(ref_gpr._gp)),
		_kernel(std::move(ref_gpr._kernel)),
		_fitted(ref_gpr._fitted),
		_has_posterior(ref_gpr._has_posterior)
	{}
	/*!
		\brief Copy assignment
	*/
	GPRegressor<K, Scalar>& operator=(const GPRegressor<K, Scalar>& ref_gpr){
		_y_train = ref_gpr._y_train;
		_X_train = ref_gpr._X_train;
		_covariance_train = ref_gpr._covariance_train;
		_gp = ref_gpr._gp;
		_kernel = ref_gpr._kernel;
		_fitted = ref_gpr._fitted;
		_has_posterior = ref_gpr._has_posterior;
		return *this;
	}
	/*!
		\brief Move assignment
	*/
	GPRegressor<K, Scalar>& operator=(GPRegressor<K, Scalar>&& ref_gpr){
		_y_train = std::move(ref_gpr._y_train);
		_X_train = std::move(ref_gpr._X_train);
		_covariance_train = std::move(ref_gpr._covariance_train);
		_gp = std::move(ref_gpr._gp);
		_kernel = std::move(ref_gpr._kernel);
		_fitted = ref_gpr._fitted;
		_has_posterior = ref_gpr._has_posterior;
		return *this;
	}
	/*!
		\brief Const kernel getter 
	*/
	const K& kernel() const {return _kernel;}
	/*!
		\brief Kernel getter (setter)
	*/
	K& kernel() {return _kernel;}
	/*!
		\brief Query if model has already been fitted
	*/
	bool fitted() const {return _fitted;}
	/*!
		\brief Query if model has already been fitted
	*/
	bool has_posterior() const {return _has_posterior;}
	/*!
		\brief Fit function
		\details X is the matrix containing the points where the GP has to be 
				 fitted. X has shape N_features x N_points. y contains the 
				 target values of the function to be learnt. NOTE: y is a column
				 vector of dimension N_points. This (i.e. the fact that y is 
				 a COLUMN vector) might change in future in case it proves
				 to be a fundamental inconsistency from an user perspective.
		\param X Predictors matrix. X has shape N_features x N_points.
		\param y Target values.
	*/
	void fit(const Eigen::Ref<const MLMatrix<Scalar>>& X, 
			const Eigen::Ref<const MLVector<Scalar>>& y){
		_fitted = true;
		_has_posterior = false;
		_y_train.resize(y.size());
		_X_train.resize(X.rows(), X.cols());
		_covariance_train.resize(X.cols(), X.cols());
		_y_train = y;
		_X_train = X;
		SP::GP::compute_gp_covariance(X, _kernel, _covariance_train);
	}
	/*!
		\brief Fit function with noise
		\details X is the matrix containing the points where the GP has to be 
				 fitted. X has shape N_features x N_points. y contains the 
				 target values of the function to be learnt. NOTE: y is a column
				 vector of dimension N_points. This (i.e. the fact that y is 
				 a COLUMN vector) might change in future in case it proves
				 to be a fundamental inconsistency from an user perspective.
		\param X Predictors matrix. X has shape N_features x N_points.
		\param y Target values.
	*/
	void fit(const Eigen::Ref<const MLMatrix<Scalar>>& X, 
			const Eigen::Ref<const MLVector<Scalar>>& y,
			const Scalar& noise_variance){
		this->fit(X, y);
		_covariance_train.diagonal().array() += noise_variance;
	}
	/*!
		\brief Predict function
		\details Prediction function. This function will return the predicted
				 values given the input query points X.
				 This function will also change the state of the regressor. 
				 After the predict function has been called, it will be possible
				 to sample other functions from the posterior by calling
				 the sample function. It will also be possible to get 
				 confidence intervals using the appropriate function.
	*/
	const MLVector<Scalar>& predict(
			const Eigen::Ref<const MLMatrix<Scalar>>& X){
		if (!_fitted){
			MLEARN_FORCED_ERROR_MESSAGE(
				"The regressor has not been fitted yet!");
		}
		_has_posterior = true;
		MLMatrix<Scalar> marginal_covariance(X.cols(), X.cols());
		MLMatrix<Scalar> cross_covariance(X.cols(), _X_train.cols());
		SP::GP::compute_gp_covariance(X, _kernel, marginal_covariance);
		SP::GP::compute_gp_covariance(X, _X_train, _kernel, cross_covariance);
		MLMatrix<Scalar> temp = 
			cross_covariance*pseudoinverse(_covariance_train);
		_gp.set_distribution(
			temp*_y_train,
			marginal_covariance - temp*cross_covariance.transpose()
		);
		return _gp.mean();
	}
	/*!
		\brief Sample functions from posterior
	*/
	MLMatrix<Scalar> sample(int N = 10) const{
		if (!_has_posterior){
			MLEARN_FORCED_ERROR_MESSAGE("Posterior not available!");
		}
		return _gp.sample(N);
	}
	/*!
		\brief Confidence interval from posterior
	*/
	MLMatrix<Scalar> confidence_interval(const Scalar& p) const{
		if (!_has_posterior){
			MLEARN_FORCED_ERROR_MESSAGE("Posterior not available!");
		}
		return _gp.confidence_interval(p);
	}
	/*!
		\brief Get posterior mean
	*/
	const MLVector<Scalar>& posterior_mean() const{
		if (!_has_posterior){
			MLEARN_FORCED_ERROR_MESSAGE("Posterior not available!");
		}
		return _gp.mean();
	}
	/*!
		\brief Get posterior covariance
	*/
	const MLMatrix<Scalar>& posterior_covariance() const{
		if (!_has_posterior){
			MLEARN_FORCED_ERROR_MESSAGE("Posterior not available!");
		}
		return _gp.covariance();
	}
};


}}

#endif