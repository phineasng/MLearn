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

namespace MLearn{
	/*!
	*	\brief 		Template class for dual numbers
	*	\note		Regarding the overloaded operators. When the input dual number is a const reference, it is assumed 
	*				that it is correctly initialized.
	*/
	template < typename FLOAT >
	class MLDualNumber{
		static_assert(std::is_floating_point<FLOAT>::value,"Underlying type must be floating point!");
		typedef MLVector<FLOAT> EPS_TYPE;
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
		operator FLOAT() const { return v; }
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
		void initialize(const FLOAT& value, size_t partial_idx = 0, size_t n_vars = 1){
			MLEARN_ASSERT( partial_idx < n_vars , "Not consistent values for initialization");
			dual_flag = true;
			v = value;
			partial_eps = EPS_TYPE::Zero(n_vars);
			partial_eps[partial_idx] = FLOAT(1.0);
		}
		bool is_dual() const{ return dual_flag; }
		FLOAT value() const{ return v; }
		FLOAT partial_der(size_t partial_idx) const{ 
			MLEARN_ASSERT(partial_idx < partial_eps.size(), "Requested derivative not available!");
			return partial_eps[partial_idx]; 
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
				res.v += x;
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
			res.v -= x;
			return res;
		}
		inline friend MLDualNumber operator-(FLOAT x, MLDualNumber&& y){
			y.v -= x;
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
				res.v -= x;
				return res;
			}
		}
		inline friend MLDualNumber operator-(const MLDualNumber& x, MLDualNumber&& y){
			if (y.dual_flag){
				y -= x;
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
				y -= x;
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
				y *= x;
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
		 


	private:
		bool dual_flag = false;  // flag to indicate if number has been initialised for differentiation
		FLOAT v;
		EPS_TYPE partial_eps;
	};
}


#endif