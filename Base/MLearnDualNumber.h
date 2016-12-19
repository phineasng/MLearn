/*! 
*
*\file 		MLearnDualNumber.h 
*\brief		Implementation of dual numbers for automatic differentiation
*\author	phineasng
*
*/ 

#ifndef MLEARN_CORE_DUAL_NUMBER_INCLUDED
#define MLEARN_CORE_DUAL_NUMBER_INCLUDED

// MLearn includes
#include <MLearn/Base/MLearnTypes.h>

// STL includes
#include <type_traits>
#include <iostream>
#include <cmath>

// Eigen 
#include <Eigen/Core>

namespace MLearn{
	/*!
	*	\brief 		Template class for dual numbers
	*	\note		Regarding the overloaded operators. When the input dual number is a const reference, it is assumed 
	*				that it is correctly initialized.
	*/
	template < typename FLOAT >
	class MLDualNumber{
		static_assert(std::is_floating_point<FLOAT>::value,"Underlying type must be floating point!");
		typedef MLSparseVector<FLOAT> EPS_TYPE;
	private:
		FLOAT v;
		EPS_TYPE partial_eps;
		bool dual_flag = false;  // flag to indicate if number has been initialised for differentiation
	public:
		// Constructors
		MLDualNumber() {}
		MLDualNumber(const MLDualNumber& ref_dual): 
				v(ref_dual.v), 
				partial_eps(ref_dual.partial_eps),
				dual_flag(ref_dual.dual_flag) {}
		MLDualNumber(MLDualNumber&& ref_dual): 
				v(std::move(ref_dual.v)), 
				partial_eps(std::move(ref_dual.partial_eps)),
				dual_flag(std::move(ref_dual.dual_flag)) {}
		MLDualNumber(const FLOAT& x): v(x) {}
		// implicit conversion to float
		operator uint() const { return uint(v); }
		// assignment operators
		MLDualNumber& operator=(const MLDualNumber& ref_dual){
			dual_flag = ref_dual.dual_flag;
			v = ref_dual.v;
			partial_eps = ref_dual.partial_eps;
			return (*this);
		}
		MLDualNumber& operator=(MLDualNumber&& ref_dual){
			dual_flag = std::move(ref_dual.dual_flag);
			v = std::move(ref_dual.v);
			partial_eps = std::move(ref_dual.partial_eps);
			return (*this);
		}
		MLDualNumber& operator=(const FLOAT& ref_val){
			dual_flag = false;
			v = ref_val;
			partial_eps.setZero();
			return (*this);
		}
		// dual number routines
		/*!
		*	\brief 	Initialize number to be a variable for differentiation
		*/
		void initialize(const FLOAT& value, int partial_idx = 0, int n_vars = 1){
			MLEARN_ASSERT( partial_idx < n_vars , "Not consistent values for initialization");
			dual_flag = true;
			v = value;
			partial_eps.resize(n_vars);
			partial_eps.setZero();
			if (partial_idx >= 0) partial_eps.coeffRef(partial_idx) = FLOAT(1.0);
		}
		bool is_dual() const{ return dual_flag; }
		FLOAT value() const{ return v; }
		FLOAT partial_der(int partial_idx) const{ 
			MLEARN_ASSERT(partial_idx < partial_eps.size(), "Requested derivative not available!");
			return partial_eps.coeff(partial_idx); 
		}
		// (friend) misc operators
		inline friend std::ostream& operator<<(std::ostream& os, const MLDualNumber& dual){
			os << dual.v;
			return os;
		}
		// ARITHMETIC OPERATORS
		// += operator
		inline MLDualNumber& operator+=(const MLDualNumber& x){
			v += x.v;
			partial_eps += x.partial_eps;
			return (*this);
		}
		inline MLDualNumber& operator+=(MLDualNumber&& x){
			v += x.v;
			if (x.dual_flag) partial_eps += x.partial_eps;
			return (*this);
		}
		inline MLDualNumber& operator+=(const FLOAT& x){
			v += x;
			return (*this);
		}
		// + operator
		inline friend MLDualNumber operator+(const MLDualNumber& x, const MLDualNumber& y){
			MLDualNumber res(x);
			res += y;
			return res;
		}
		inline friend MLDualNumber operator+(FLOAT x, const MLDualNumber& y){
			MLDualNumber res(y);
			res.v += x;
			return res;
		}
		inline friend MLDualNumber operator+(FLOAT x, MLDualNumber&& y){
			y.v += x;
			return y;
		}
		inline friend MLDualNumber operator+(const MLDualNumber& x,FLOAT y){
			MLDualNumber res(x);
			res.v += y;
			return res;
		}
		inline friend MLDualNumber operator+(MLDualNumber&& x,FLOAT y){
			x.v += y;
			return x;
		}
		inline friend MLDualNumber operator+(MLDualNumber&& x, const MLDualNumber& y){
			if (x.dual_flag){
				x += y;
				return x; 
			}else{
				MLDualNumber res(y);
				res.v += x.v;
				return res;
			}
		}
		inline friend MLDualNumber operator+(const MLDualNumber& x, MLDualNumber&& y){
			if (y.dual_flag){
				y += x;
				return y; 
			}else{
				MLDualNumber res(x);
				res.v += y.v;
				return res;
			}
		}
		inline friend MLDualNumber operator+(MLDualNumber&& x, MLDualNumber&& y){
			if (x.dual_flag){
				x += y;
				return x;
			}else if (y.dual_flag){
				y += x;
				return y;
			}else{
				x.v += y.v;
				return x;
			}
		}		
		// -= operator
		inline MLDualNumber& operator-=(const MLDualNumber& x){
			v -= x.v;
			partial_eps -= x.partial_eps;
			return (*this);
		}
		inline MLDualNumber& operator-=(MLDualNumber&& x){
			v -= x.v;
			if (x.dual_flag) partial_eps -= x.partial_eps;
			return (*this);
		}
		inline MLDualNumber& operator-=(const FLOAT& x){
			v -= x;
			return (*this);
		}
		// - operator
		inline MLDualNumber operator-() const{
			MLDualNumber res(*this);
			res.v *= FLOAT(-1.0);
			res.partial_eps *= FLOAT(-1.0);
			return res;
		}
		inline friend MLDualNumber operator-(const MLDualNumber& x, const MLDualNumber& y){
			MLDualNumber res(x);
			res -= y;
			return res;
		}
		inline friend MLDualNumber operator-(FLOAT x, const MLDualNumber& y){
			MLDualNumber res(y);
			res.v = x - res.v;
			res.partial_eps *= FLOAT(-1.0);
			return res;
		}
		inline friend MLDualNumber operator-(FLOAT x, MLDualNumber&& y){
			y.v = x - y.v;
			if (y.dual_flag) y.partial_eps *= FLOAT(-1.0);
			return y;
		}
		inline friend MLDualNumber operator-(const MLDualNumber& x,FLOAT y){
			MLDualNumber res(x);
			res.v -= y;
			return res;
		}
		inline friend MLDualNumber operator-(MLDualNumber&& x,FLOAT y){
			x.v -= y;
			return x;
		}
		inline friend MLDualNumber operator-(MLDualNumber&& x, const MLDualNumber& y){
			if (x.dual_flag){
				x -= y;
				return x; 
			}else{
				MLDualNumber res(y);
				res.v = x.v - y.v;
				res.partial_eps *= FLOAT(-1.0);
				return res;
			}
		}
		inline friend MLDualNumber operator-(const MLDualNumber& x, MLDualNumber&& y){
			if (y.dual_flag){
				y.v = x.v - y.v;
				y.partial_eps = x.partial_eps - y.partial_eps;
				return y; 
			}else{
				MLDualNumber res(x);
				res.v -= y.v;
				return res;
			}
		}
		inline friend MLDualNumber operator-(MLDualNumber&& x, MLDualNumber&& y){
			if (x.dual_flag){
				x -= y;
				return x;
			}else if (y.dual_flag){
				y.v = x.v - y.v;
				y.partial_eps *= FLOAT(-1.0);
				return y;
			}else{
				x.v -= y.v;
				return x;
			}
		}
		// *= operator
		inline MLDualNumber& operator*=(const MLDualNumber& x){
			partial_eps = x.partial_eps*v + partial_eps*x.v;
			v *= x.v;
			return (*this);
		}
		inline MLDualNumber& operator*=(MLDualNumber&& x){
			partial_eps *= x.v;
			if (x.dual_flag) partial_eps += x.partial_eps*v;
			v *= x.v;
			return (*this);
		}
		inline MLDualNumber& operator*=(const FLOAT& x){
			v *= x;
			partial_eps *= x;
			return (*this);
		}
		// * operator
		inline friend MLDualNumber operator*(const MLDualNumber& x, const MLDualNumber& y){
			MLDualNumber res(x);
			res *= y;
			return res;
		}
		inline friend MLDualNumber operator*(FLOAT x, const MLDualNumber& y){
			MLDualNumber res(y);
			res *= x;
			return res;
		}
		inline friend MLDualNumber operator*(FLOAT x, MLDualNumber&& y){
			y.v *= x;
			if (y.dual_flag) y.partial_eps *= x;
			return y;
		}
		inline friend MLDualNumber operator*(const MLDualNumber& x,FLOAT y){
			MLDualNumber res(x);
			res *= y;
			return res;
		}
		inline friend MLDualNumber operator*(MLDualNumber&& x,FLOAT y){
			x.v *= y;
			if (x.dual_flag) x.partial_eps *= y;
			return x;
		}
		inline friend MLDualNumber operator*(MLDualNumber&& x, const MLDualNumber& y){
			if (x.dual_flag){
				x *= y;
				return x; 
			}else{
				MLDualNumber res(y);
				res.v *= x.v;
				res.partial_eps *= x.v;
				return res;
			}
		}
		inline friend MLDualNumber operator*(const MLDualNumber& x, MLDualNumber&& y){
			if (y.dual_flag){
				y *= x;
				return y; 
			}else{
				MLDualNumber res(x);
				res.v *= y.v;
				res.partial_eps *= y.v;
				return res;
			}
		}
		inline friend MLDualNumber operator*(MLDualNumber&& x, MLDualNumber&& y){
			if (x.dual_flag){
				x *= y;
				return x;
			}else if (y.dual_flag){
				y.v *= x.v;
				y.partial_eps *= x.v;
				return y;
			}else{
				x.v *= y.v;
				return x;
			}
		}
		// /= operator
		inline MLDualNumber& operator/=(const MLDualNumber& x){
			partial_eps = (partial_eps*x.v - x.partial_eps*v)/(x.v*x.v);
			v /= x.v;
			return (*this);
		}
		inline MLDualNumber& operator/=(MLDualNumber&& x){
			partial_eps /= x.v;
			if (x.dual_flag) partial_eps -= x.partial_eps*v/(x.v*x.v);
			v /= x.v;
			return (*this);
		}
		inline MLDualNumber& operator/=(const FLOAT& x){
			v /= x;
			partial_eps /= x;
			return (*this);
		}
		// / operator
		inline friend MLDualNumber operator/(const MLDualNumber& x, const MLDualNumber& y){
			MLDualNumber res(x);
			res /= y;
			return res;
		}
		inline friend MLDualNumber operator/(FLOAT x, const MLDualNumber& y){
			MLDualNumber res(y);
			res.partial_eps *= -x/(res.v*res.v); 
			res.v = x/res.v;
			return res;
		}
		inline friend MLDualNumber operator/(FLOAT x, MLDualNumber&& y){
			if (y.dual_flag) y.partial_eps *= -x/(y.v*y.v);
			y.v /= x;
			return y;
		}
		inline friend MLDualNumber operator/(const MLDualNumber& x,FLOAT y){
			MLDualNumber res(x);
			res /= y;
			return res;
		}
		inline friend MLDualNumber operator/(MLDualNumber&& x,FLOAT y){
			x.v /= y;
			if (x.dual_flag) x.partial_eps /= y;
			return x;
		}
		inline friend MLDualNumber operator/(MLDualNumber&& x, const MLDualNumber& y){
			if (x.dual_flag){
				x /= y;
				return x; 
			}else{
				MLDualNumber res(y);
				res.partial_eps *= -x.v/(res.v*res.v); 
				res.v = x.v/res.v;
				return res;
			}
		}
		inline friend MLDualNumber operator/(const MLDualNumber& x, MLDualNumber&& y){
			if (y.dual_flag){
				y.partial_eps = (x.partial_eps*y.v - y.partial_eps*x.v)/(y.v*y.v);
				y.v = x.v/y.v;
				return y; 
			}else{
				MLDualNumber res(x);
				res.v /= y.v;
				res.partial_eps /= y.v;
				return res;
			}
		}
		inline friend MLDualNumber operator/(MLDualNumber&& x, MLDualNumber&& y){
			if (x.dual_flag){
				x /= y;
				return x;
			}else if (y.dual_flag){
				y.partial_eps *= (-x.v)/(y.v*y.v);
				y.v = x.v/y.v;
				return y;
			}else{
				x.v /= y.v;
				return x;
			}
		}
		// COMMON MATHEMATICAL FUNCTIONS
		// cosine
		inline friend MLDualNumber cos(const MLDualNumber& x){
			MLDualNumber res(x);
			res.partial_eps *= -std::sin(res.v);
			res.v = std::cos(res.v);
			return res;
		} 
		inline friend MLDualNumber cos(MLDualNumber&& x){
			if (x.dual_flag) x.partial_eps *= -std::sin(x.v);
			x.v = std::cos(x.v);
			return x;
		} 
		// sine
		inline friend MLDualNumber sin(const MLDualNumber& x){
			MLDualNumber res(x);
			res.partial_eps *= std::cos(res.v);
			res.v = std::sin(res.v);
			return res;
		} 
		inline friend MLDualNumber sin(MLDualNumber&& x){
			if (x.dual_flag) x.partial_eps *= std::cos(x.v);
			x.v = std::sin(x.v);
			return x;
		} 
		// tangent
		inline friend MLDualNumber tan(const MLDualNumber& x){
			MLDualNumber res(x);
			FLOAT c = std::cos(res.v);
			res.partial_eps /= c*c;
			res.v = std::tan(res.v);
			return res;
		} 
		inline friend MLDualNumber tan(MLDualNumber&& x){
			if (x.dual_flag){
				FLOAT c = std::cos(x.v);
				x.partial_eps /= c*c;	
			} 
			x.v = std::tan(x.v);
			return x;
		} 
		// acos
		inline friend MLDualNumber acos(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::acos(res.v);
			res.partial_eps *= -1.0/std::sqrt(1.0-x.v*x.v);
			return res;
		} 
		inline friend MLDualNumber acos(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= -1.0/std::sqrt(1.0-x.v*x.v);	
			} 
			x.v = std::acos(x.v);
			return x;
		} 
		// asin
		inline friend MLDualNumber asin(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::asin(res.v);
			res.partial_eps *= 1.0/std::sqrt(1.0-x.v*x.v);
			return res;
		} 
		inline friend MLDualNumber asin(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= 1.0/std::sqrt(1.0-x.v*x.v);	
			} 
			x.v = std::asin(x.v);
			return x;
		} 
		// atan
		inline friend MLDualNumber atan(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::atan(res.v);
			res.partial_eps *= 1.0/(1.0+x.v*x.v);
			return res;
		} 
		inline friend MLDualNumber atan(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= 1.0/(1.0+x.v*x.v);	
			} 
			x.v = std::atan(x.v);
			return x;
		} 
		// cosh
		inline friend MLDualNumber cosh(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::cosh(res.v);
			res.partial_eps *= std::sinh(x.v);
			return res;
		} 
		inline friend MLDualNumber cosh(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= std::sinh(x.v);	
			} 
			x.v = std::cosh(x.v);
			return x;
		} 
		// sinh
		inline friend MLDualNumber sinh(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::sinh(res.v);
			res.partial_eps *= std::cosh(x.v);
			return res;
		} 
		inline friend MLDualNumber sinh(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= std::cosh(x.v);	
			} 
			x.v = std::sinh(x.v);
			return x;
		} 
		// tanh
		inline friend MLDualNumber tanh(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::tanh(res.v);
			res.partial_eps *= 1.0 - res.v*res.v;
			return res;
		} 
		inline friend MLDualNumber tanh(MLDualNumber&& x){
			x.v = std::tanh(x.v);
			if (x.dual_flag){
				x.partial_eps *= 1.0 - x.v*x.v;	
			} 
			return x;
		} 
		// acosh
		inline friend MLDualNumber acosh(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::acosh(res.v);
			res.partial_eps *= 1.0/(std::sqrt(x.v - 1.0)*std::sqrt(x.v + 1.0));
			return res;
		} 
		inline friend MLDualNumber acosh(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= 1.0/(std::sqrt(x.v - 1.0)*std::sqrt(x.v + 1.0));	
			} 
			x.v = std::acosh(x.v);
			return x;
		} 
		// asinh
		inline friend MLDualNumber asinh(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::asinh(res.v);
			res.partial_eps *= 1.0/std::sqrt(x.v*x.v + 1.0);
			return res;
		} 
		inline friend MLDualNumber asinh(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= 1.0/std::sqrt(x.v*x.v + 1.0);	
			} 
			x.v = std::asinh(x.v);
			return x;
		} 
		// atanh
		inline friend MLDualNumber atanh(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::atanh(res.v);
			res.partial_eps *= 1.0/(1.0 - x.v*x.v);
			return res;
		} 
		inline friend MLDualNumber atanh(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps *= 1.0/(1.0 - x.v*x.v);	
			} 
			x.v = std::atanh(x.v);
			return x;
		} 
		// exp
		inline friend MLDualNumber exp(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::exp(res.v);
			res.partial_eps *= res.v;
			return res;
		} 
		inline friend MLDualNumber exp(MLDualNumber&& x){
			x.v = std::exp(x.v);
			if (x.dual_flag){
				x.partial_eps *= x.v;	
			} 
			return x;
		} 
		// log
		inline friend MLDualNumber log(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::log(res.v);
			res.partial_eps /= x.v;
			return res;
		} 
		inline friend MLDualNumber log(MLDualNumber&& x){
			if (x.dual_flag){
				x.partial_eps /= x.v;	
			} 
			x.v = std::log(x.v);
			return x;
		} 
		// sqrt
		inline friend MLDualNumber sqrt(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::sqrt(res.v);
			res.partial_eps *= 0.5/res.v;
			return res;
		} 
		inline friend MLDualNumber sqrt(MLDualNumber&& x){
			x.v = std::sqrt(x.v);
			if (x.dual_flag){
				x.partial_eps *= 0.5/x.v;	
			} 
			return x;
		} 
		// abs
		inline friend MLDualNumber abs(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::abs(res.v);
			if ( std::signbit(x.v) ){
				res.partial_eps *= FLOAT(-1.0);
			}
			return res;
		} 
		inline friend MLDualNumber abs(MLDualNumber&& x){
			x.v = std::abs(x.v);
			if ((x.dual_flag) && std::signbit(x.v)){
				x.partial_eps *= FLOAT(-1.0);	
			} 
			return x;
		} 
		// ceil
		inline friend MLDualNumber ceil(const MLDualNumber& x){
			MLDualNumber res(x);
			res.v = std::ceil(res.v);
			return res;
		} 
		inline friend MLDualNumber ceil(MLDualNumber&& x){
			x.v = std::ceil(x.v);
			return x;
		} 
		// check routines
		inline friend bool isfinite(const MLDualNumber& x){
			return std::isfinite(x.v);
		}
		inline friend bool isinf(const MLDualNumber& x){
			return std::isinf(x.v);
		}
		inline friend bool isnan(const MLDualNumber& x){
			return std::isnan(x.v);
		}
		inline friend bool isnormal(const MLDualNumber& x){
			return std::isnormal(x.v);
		}
		inline friend bool signbit(const MLDualNumber& x){
			return std::signbit(x.v);
		}
		// comparison routines
		inline friend bool operator<(const MLDualNumber& x, const MLDualNumber& y){
			return x.v < y.v;
		}
		inline friend bool operator<(const MLDualNumber& x, FLOAT y){
			return x.v < y;
		}
		inline friend bool operator<(FLOAT x, const MLDualNumber& y){
			return x < y.v;
		}
		inline friend bool operator>(const MLDualNumber& x, const MLDualNumber& y){
			return x.v > y.v;
		}
		inline friend bool operator>(const MLDualNumber& x, FLOAT y){
			return x.v > y;
		}
		inline friend bool operator>(FLOAT x, const MLDualNumber& y){
			return x > y.v;
		}
		inline friend bool isgreater(const MLDualNumber& x, const MLDualNumber& y){
			return std::isgreater(x.v,y.v);
		}
		inline friend bool isgreater(const MLDualNumber& x, FLOAT y){
			return std::isgreater(x.v,y);
		}
		inline friend bool isgreater(FLOAT x, const MLDualNumber& y){
			return std::isgreater(x,y.v);
		}
		inline friend bool isgreaterequal(const MLDualNumber& x, const MLDualNumber& y){
			return std::isgreaterequal(x.v,y.v);
		}
		inline friend bool isgreaterequal(const MLDualNumber& x, FLOAT y){
			return std::isgreaterequal(x.v,y);
		}
		inline friend bool isgreaterequal(FLOAT x, const MLDualNumber& y){
			return std::isgreaterequal(x,y.v);
		}
		inline friend bool isless(const MLDualNumber& x, const MLDualNumber& y){
			return std::isless(x.v,y.v);
		}
		inline friend bool isless(const MLDualNumber& x, FLOAT y){
			return std::isless(x.v,y);
		}
		inline friend bool isless(FLOAT x, const MLDualNumber& y){
			return std::isless(x,y.v);
		}
		inline friend bool islessequal(const MLDualNumber& x, const MLDualNumber& y){
			return std::islessequal(x.v,y.v);
		}
		inline friend bool islessequal(const MLDualNumber& x, FLOAT y){
			return std::islessequal(x.v,y);
		}
		inline friend bool islessequal(FLOAT x, const MLDualNumber& y){
			return std::islessequal(x,y.v);
		}
		inline friend bool islessgreater(const MLDualNumber& x, const MLDualNumber& y){
			return std::islessgreater(x.v,y.v);
		}
		inline friend bool islessgreater(const MLDualNumber& x, FLOAT y){
			return std::islessgreater(x.v,y);
		}
		inline friend bool islessgreater(FLOAT x, const MLDualNumber& y){
			return std::islessgreater(x,y.v);
		}
		inline friend bool isunordered(const MLDualNumber& x, const MLDualNumber& y){
			return std::isunordered(x.v,y.v);
		}
		inline friend bool isunordered(const MLDualNumber& x, FLOAT y){
			return std::isunordered(x.v,y);
		}
		inline friend bool isunordered(FLOAT x, const MLDualNumber& y){
			return std::isunordered(x,y.v);
		}
	};

	template< typename FLOAT >
	struct enables_autodiff< MLDualNumber<FLOAT> >{
		static const bool value = true;
	};

}

namespace std{
	template < typename FLOAT >
	struct is_floating_point< MLearn::MLDualNumber<FLOAT> >: is_floating_point<FLOAT> {};
}

namespace Eigen{
	template< typename FLOAT > struct NumTraits<MLearn::MLDualNumber<FLOAT>>: NumTraits<FLOAT> {
		// TODO(phineasng): find better costs (make Eigen::HugeCost work)
		enum { 
			ReadCost = 5,
			AddCost = 5,
			MulCost = 5,
		};

		typedef MLearn::MLDualNumber<FLOAT> Real;
		typedef MLearn::MLDualNumber<FLOAT> NonInteger;
		typedef MLearn::MLDualNumber<FLOAT> Nested;
	
	};
}


#endif