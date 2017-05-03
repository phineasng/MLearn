# Gaussian Processes demo

This demo will plot a gaussian process with a user-specified kernel.

## Setup 

Be sure to run the [setup]() steps.
Compile specifically this demo from the build directory with `make GPSamplingDemo`

## How to run the demo

This demo can be run from the correct subfolder with the command
`./GPSamplingDemo`. If no option is provided a simple linear function will be plotted.
Otherwise you can provided a numeric value with the option `--kernel` to specify which kernel 
to use. Here is a list of available kernels in this demo (you can print this information also
by using the option `--help`):
```bash
./GPSamplingDemo --help
--kernel arg          Value in [0, 10].This value indicates which kernel will
                        be used. Available kernels:
                        LINEAR (0)
                        POLYNOMIAL (1)
                        RBF (2)
                        LAPLACIAN (3)
                        ABEL (4)
                        CONSTANT (5)
                        MIN (6)
                        MATERN_32 (7)
                        MATERN_52 (8)
                        RATIONAL_QUADRATIC (9)
                        PERIODIC (10)
```