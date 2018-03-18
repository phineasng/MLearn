#ifndef ML_EXCEPTION_H_FILE
#define ML_EXCEPTION_H_FILE

// STL includes
#include <exception>

class MLNotImplemented : public std::logic_error
{
public:
    MLNotImplemented() : std::logic_error("Function not implemented"){};
};

#endif // ML_EXCEPTION_H_FILE