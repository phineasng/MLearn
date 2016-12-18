#include <iostream>

#include <MLearn/Core>
#include <Eigen/Core>

int main(int argc, char* argv[]){

	typedef double FLOAT_TYPE;
	using namespace MLearn;
	typedef MLDualNumber<FLOAT_TYPE> ml_dual_float;

	MLDualNumber<FLOAT_TYPE> x, y, z;
	x.initialize(0.25,0,2);
	y.initialize(0.6,1,2);
	z.initialize(1e-15,0,1);
	MLDualNumber<FLOAT_TYPE> f = x+y;
	MLDualNumber<FLOAT_TYPE> f2 = x*y;
	MLDualNumber<FLOAT_TYPE> f3 = x-y;
	MLDualNumber<FLOAT_TYPE> f4 = x*x*y + y*y*y*y*y;
	MLDualNumber<FLOAT_TYPE> f5 = x/y;
	MLDualNumber<FLOAT_TYPE> f6 = 1.5/x;
	MLDualNumber<FLOAT_TYPE> f7 = MLDualNumber<FLOAT_TYPE>(5.0)/x;
	MLDualNumber<FLOAT_TYPE> f8 = y/(x*x);
	MLDualNumber<FLOAT_TYPE> f9 = x/MLDualNumber<FLOAT_TYPE>(50.0);
	MLDualNumber<FLOAT_TYPE> f10 = x/10.0;
	MLDualNumber<FLOAT_TYPE> f11 = f3/f5 - f10*f2;
	MLDualNumber<FLOAT_TYPE> f12 = cos(-x*y);
	MLDualNumber<FLOAT_TYPE> f13 = cos(y)*cos(y) + sin(y)*sin(y);
	MLDualNumber<FLOAT_TYPE> f14 = cos(y)*cos(y);
	MLDualNumber<FLOAT_TYPE> f15 = 1.0 - sin(y)*sin(y);
	MLDualNumber<FLOAT_TYPE> f16 = 2.0*sin(y)*cos(y);
	MLDualNumber<FLOAT_TYPE> f17 = sin(2*y);
	MLDualNumber<FLOAT_TYPE> f18 = sin(x*y);
	MLDualNumber<FLOAT_TYPE> f19 = tan(x*y)*cos(x*y);
	MLDualNumber<FLOAT_TYPE> f20 = sin(z)/z;
	MLDualNumber<FLOAT_TYPE> f21 = cos(z)/(1.0-z*z*0.5);
	MLDualNumber<FLOAT_TYPE> f22 = asin(x*y)*x;
	MLDualNumber<FLOAT_TYPE> f23 = acos(y)*x;
	MLDualNumber<FLOAT_TYPE> f24 = atanh(x)*x;
	MLDualNumber<FLOAT_TYPE> f25 = exp(x*x*y);
	MLDualNumber<FLOAT_TYPE> f26 = 2*x*y*exp(x*x*y);
	MLDualNumber<FLOAT_TYPE> f27 = x*x*exp(x*x*y);
	MLDualNumber<FLOAT_TYPE> f28 = log(x*x*y);
	MLDualNumber<FLOAT_TYPE> f29 = 2.0/x;
	MLDualNumber<FLOAT_TYPE> f30 = 1.0/y;
	MLDualNumber<FLOAT_TYPE> f31 = sqrt(x*x*y);
	MLDualNumber<FLOAT_TYPE> f32 = x*y/sqrt(x*x*y);
	MLDualNumber<FLOAT_TYPE> f33 = x*x*0.5/sqrt(x*x*y);
	MLDualNumber<FLOAT_TYPE> f34 = abs(f31-f18);
	
	FLOAT_TYPE a = 1.52;

	x = y;


	std::cout << "x dual flag: " << x.is_dual() << std::endl;
	std::cout << "a: " << a << std::endl;
	std::cout << "x value: " << x << std::endl;
	std::cout << "x partial der 1: " << x.partial_der(1) << std::endl;

	std::cout << "f = x+y: " << f << std::endl;
	std::cout << "d(x+y)/dx: " << f.partial_der(0) << std::endl;
	std::cout << "d(x+y)/dy: " << f.partial_der(1) << std::endl;
	std::cout << "x+a: " << (x+a) << std::endl;
	std::cout << "a+y: " << (a+y) << std::endl;
	std::cout << "x+1.0: " << (x+1.0) << std::endl;
	std::cout << "5.0+y: " << (5.0+y) << std::endl;
	std::cout << "5.0+4.0: " << MLDualNumber<FLOAT_TYPE>(5.0) + MLDualNumber<FLOAT_TYPE>(4.0) << std::endl;
	std::cout << "5.0+x: " << MLDualNumber<FLOAT_TYPE>(5.0) + x << std::endl;
	std::cout << "x+4.0: " << x + MLDualNumber<FLOAT_TYPE>(4.0) << std::endl;
	std::cout << "4.0+4.0: " << MLDualNumber<FLOAT_TYPE>(4.0) + 4.0 << std::endl;

	std::cout << "f2 = x*y: " << f2 << std::endl;
	std::cout << "d(f2)/dx: " << f2.partial_der(0) << std::endl;
	std::cout << "d(f2)/dy: " << f2.partial_der(1) << std::endl;
	std::cout << "x*a: " << (x*a) << std::endl;
	std::cout << "a*y: " << (a*y) << std::endl;
	std::cout << "x*1.5: " << (x*1.5) << std::endl;
	std::cout << "5.0*y: " << (5.0*y) << std::endl;
	std::cout << "5.0*4.0: " << MLDualNumber<FLOAT_TYPE>(5.0) * MLDualNumber<FLOAT_TYPE>(4.0) << std::endl;
	std::cout << "5.0*x: " << MLDualNumber<FLOAT_TYPE>(5.0) * x << std::endl;
	std::cout << "x*4.0: " << x * MLDualNumber<FLOAT_TYPE>(4.0) << std::endl;
	std::cout << "4.0*4.0: " << MLDualNumber<FLOAT_TYPE>(4.0) * 4.0 << std::endl;

	std::cout << "f3 = x-y: " << f3 << std::endl;
	std::cout << "d(f3)/dx: " << f3.partial_der(0) << std::endl;
	std::cout << "d(f3)/dy: " << f3.partial_der(1) << std::endl;
	std::cout << "x-a: " << (x-a) << std::endl;
	std::cout << "a-y: " << (a-y) << std::endl;
	std::cout << "x-1.5: " << (x-1.5) << std::endl;
	std::cout << "5.0-y: " << (5.0-y) << std::endl;
	std::cout << "5.0-4.0: " << MLDualNumber<FLOAT_TYPE>(5.0) - MLDualNumber<FLOAT_TYPE>(4.0) << std::endl;
	std::cout << "5.0-x: " << MLDualNumber<FLOAT_TYPE>(5.0) - x << std::endl;
	std::cout << "x-4.0: " << x - MLDualNumber<FLOAT_TYPE>(4.0) << std::endl;
	std::cout << "4.0-4.0: " << MLDualNumber<FLOAT_TYPE>(4.0) - 4.0 << std::endl;

	std::cout << "f4 = x*x*y + y*y*y*y*y: " << f4 << std::endl;
	std::cout << "d(f4)/dx: " << f4.partial_der(0) << std::endl;
	std::cout << "d(f4)/dy: " << f4.partial_der(1) << std::endl;

	std::cout << "f5: " << f5 << std::endl;
	std::cout << "d(f5)/dx: " << f5.partial_der(0) << std::endl;
	std::cout << "d(f5)/dy: " << f5.partial_der(1) << std::endl;

	std::cout << "f6: " << f6 << std::endl;
	std::cout << "d(f6)/dx: " << f6.partial_der(0) << std::endl;

	std::cout << "f7: " << f7 << std::endl;
	std::cout << "d(f7)/dx: " << f7.partial_der(0) << std::endl;

	std::cout << "f8: " << f8 << std::endl;
	std::cout << "d(f8)/dx: " << f8.partial_der(0) << std::endl;
	std::cout << "d(f8)/dy: " << f8.partial_der(1) << std::endl;

	std::cout << "f9: " << f9 << std::endl;
	std::cout << "d(f9)/dx: " << f9.partial_der(0) << std::endl;

	std::cout << "f10: " << f10 << std::endl;
	std::cout << "d(f10)/dx: " << f10.partial_der(0) << std::endl;
	std::cout << "d(f10)/dy: " << f10.partial_der(1) << std::endl;

	std::cout << "f11: " << f11 << std::endl;
	std::cout << "d(f11)/dx: " << f11.partial_der(0) << std::endl;
	std::cout << "d(f11)/dy: " << f11.partial_der(1) << std::endl;

	std::cout << "f12: " << f12 << std::endl;
	std::cout << "d(f12)/dx: " << f12.partial_der(0) << std::endl;
	std::cout << "d(f12)/dy: " << f12.partial_der(1) << std::endl;

	std::cout << "f13: " << f13 << std::endl;
	std::cout << "d(f13)/dx: " << f13.partial_der(0) << std::endl;
	std::cout << "d(f13)/dy: " << f13.partial_der(1) << std::endl;

	std::cout << "f14: " << f14 << std::endl;
	std::cout << "d(f14)/dx: " << f14.partial_der(0) << std::endl;
	std::cout << "d(f14)/dy: " << f14.partial_der(1) << std::endl;

	std::cout << "f15: " << f15 << std::endl;
	std::cout << "d(f15)/dx: " << f15.partial_der(0) << std::endl;
	std::cout << "d(f15)/dy: " << f15.partial_der(1) << std::endl;

	std::cout << "f16: " << f16 << std::endl;
	std::cout << "d(f16)/dx: " << f16.partial_der(0) << std::endl;
	std::cout << "d(f16)/dy: " << f16.partial_der(1) << std::endl;

	std::cout << "f17: " << f17 << std::endl;
	std::cout << "d(f17)/dx: " << f17.partial_der(0) << std::endl;
	std::cout << "d(f17)/dy: " << f17.partial_der(1) << std::endl;

	std::cout << "f18: " << f18 << std::endl;
	std::cout << "d(f18)/dx: " << f18.partial_der(0) << std::endl;
	std::cout << "d(f18)/dy: " << f18.partial_der(1) << std::endl;

	std::cout << "f19: " << f19 << std::endl;
	std::cout << "d(f19)/dx: " << f19.partial_der(0) << std::endl;
	std::cout << "d(f19)/dy: " << f19.partial_der(1) << std::endl;

	std::cout << "f20: " << f20 << std::endl;
	std::cout << "d(f20)/dx: " << f20.partial_der(0) << std::endl;

	std::cout << "f21: " << f21 << std::endl;
	std::cout << "d(f21)/dx: " << f21.partial_der(0) << std::endl;

	std::cout << "f22: " << f22 << std::endl;
	std::cout << "d(f22)/dx: " << f22.partial_der(0) << std::endl;
	std::cout << "d(f22)/dy: " << f22.partial_der(1) << std::endl;

	std::cout << "f23: " << f23 << std::endl;
	std::cout << "d(f23)/dx: " << f23.partial_der(0) << std::endl;
	std::cout << "d(f23)/dy: " << f23.partial_der(1) << std::endl;

	std::cout << "f24: " << f24 << std::endl;
	std::cout << "d(f24)/dx: " << f24.partial_der(0) << std::endl;

	std::cout << "f25: " << f25 << std::endl;
	std::cout << "d(f25)/dx: " << f25.partial_der(0) << std::endl;
	std::cout << "d(f25)/dy: " << f25.partial_der(1) << std::endl;

	std::cout << "f26: " << f26 << std::endl;
	std::cout << "d(f26)/dx: " << f26.partial_der(0) << std::endl;
	std::cout << "d(f26)/dy: " << f26.partial_der(1) << std::endl;
	
	std::cout << "f27: " << f27 << std::endl;
	std::cout << "d(f27)/dx: " << f27.partial_der(0) << std::endl;
	std::cout << "d(f27)/dy: " << f27.partial_der(1) << std::endl;
	
	std::cout << "f28: " << f28 << std::endl;
	std::cout << "d(f28)/dx: " << f28.partial_der(0) << std::endl;
	std::cout << "d(f28)/dy: " << f28.partial_der(1) << std::endl;
	
	std::cout << "f29: " << f29 << std::endl;
	std::cout << "d(f29)/dx: " << f29.partial_der(0) << std::endl;
	std::cout << "d(f29)/dy: " << f29.partial_der(1) << std::endl;
	
	std::cout << "f30: " << f30 << std::endl;
	std::cout << "d(f30)/dx: " << f30.partial_der(0) << std::endl;
	std::cout << "d(f30)/dy: " << f30.partial_der(1) << std::endl;
	
	std::cout << "f31: " << f31 << std::endl;
	std::cout << "d(f31)/dx: " << f31.partial_der(0) << std::endl;
	std::cout << "d(f31)/dy: " << f31.partial_der(1) << std::endl;
	
	std::cout << "f32: " << f32 << std::endl;
	std::cout << "d(f32)/dx: " << f32.partial_der(0) << std::endl;
	std::cout << "d(f32)/dy: " << f32.partial_der(1) << std::endl;
	
	std::cout << "f33: " << f33 << std::endl;
	std::cout << "d(f33)/dx: " << f33.partial_der(0) << std::endl;
	std::cout << "d(f33)/dy: " << f33.partial_der(1) << std::endl;

	std::cout << "f34: " << f34 << std::endl;
	std::cout << "d(f34)/dx: " << f34.partial_der(0) << std::endl;
	std::cout << "d(f34)/dy: " << f34.partial_der(1) << std::endl;
	
	Eigen::NumTraits<MLDualNumber<FLOAT_TYPE>> traits = Eigen::NumTraits<MLDualNumber<FLOAT_TYPE>>();
	std::cout << "trait.IsInteger = " << traits.IsInteger << std::endl;
	std::cout << "trait.AddCost = " << traits.AddCost << std::endl;
	std::cout << "trait.MulCost = " << traits.MulCost << std::endl;

	MLVector<ml_dual_float> x0(10);

	for (size_t i = 0; i < 10; ++i){
		x0[i].initialize(i,i,10);
	} 

	ml_dual_float cost_at_x0 = cost(x0);

	std::cout << "Gradient" << std::endl;
	for (size_t i = 0; i < 10; ++i){
		std::cout << i <<")" << cost_at_x0.partial_der(i) << std::endl;
	}	

	return 0;
}