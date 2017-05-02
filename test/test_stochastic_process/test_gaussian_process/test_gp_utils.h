#ifndef TEST_GP_UTILS_H_FILE_MLEARN
#define TEST_GP_UTILS_H_FILE_MLEARN

// Eigen includes
#include <Eigen/Core>

// MLearn includes
#include <MLearn/Core>

namespace TestUtils{

	class MockKernel{
	public:
		KERNEL_COMPUTE_TEMPLATE_START(x,y)
			return x.dot(y);
		KERNEL_COMPUTE_TEMPLATE_END
	};

}

#endif