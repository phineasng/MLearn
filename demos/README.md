# MLearn Demos

## Dependencies

To run the script you need to have [CMake](https://cmake.org/) installed.
The dependencies of the library will be installed by the script if required.
Further dependencies for the demo (the script will always try to install these):
- [gnuplot](http://www.gnuplot.info/)
- [gnuplot-iostream](http://stahlke.org/dan/gnuplot-iostream/)
- [boost compiled libraries](http://www.boost.org/)
- [CImg](http://cimg.eu/)

## Setup

To setup the correct dependencies, you just need to run the `setup.sh`
from this folder, i.e. `/path/to/MLearn/demos`. 
If the options are correctly set, the script should _automatically_ perform these steps:
1. Create a build directory
2. Download/setup dependencies
  - If you don't have the necessary dependencies (or you're not sure), you can specify the flags `--eigen`, `--boost` or `--all` to require the installation of respectively the Eigen library, Boost or both.
  - If either Eigen or Boost are already installed, you should specify respectively `--eigen_path` (the folder containing the `Eigen/` folder) or `--boost_path` (the folder you specified when installing boost with the `--prefix` flag - usually this defaults to `/usr/local/`)
  - `gnuplot` and `gnuplot-iostream` will be installed in _any_ case (super user rights are required for `gnuplot`). 
  - As above, `CImg` will be installed in _any_ case (in this case, this just means downloading the library and moving the header file in the correct folder). 
3. Run cmake

The script can be provided with these options:
```bash
  --clean              : clean build directory.
  --eigen              : download and setup eigen.
  --boost              : download and setup boost.
  --all                : download and setup both boost and eigen.
  --eigen_path         : root directory where to find eigen (where 
                         your Eigen/ folder is stored).
  --boost_path         : root directory where to find boost (where you
                         installed headers and libraries using the 
                         install scripts provided by the boost library).
  -h                   : print this help.

```
*NOTE*: the scripts will try to install gnuplot with `sudo` so you will probably 
have to input your password. 

For example, to install all the dependencies from a clean build directory:
```bash
bash setup.sh --clean --all
```

Now you can `cd` into the build folder `demo_build`. If you run the `make`
command here, you will build all the demos. You can also specify a single demo
to be built, e.g. `make MultivariateSamplingDemo`.

## Available demos
Here I will try to maintain a list of available demos:
1. [Multivariate Gaussian sampling](https://github.com/phineasng/MLearn/tree/master/demos/demo_sampling)
2. [Gaussian Processes sampling](https://github.com/phineasng/MLearn/tree/master/demos/demo_gaussian_process)
3. [Gaussian Process Regression](https://github.com/phineasng/MLearn/tree/master/demos/demo_regression)
3. [3-layer NN on MNIST](https://github.com/phineasng/MLearn/tree/master/demos/demo_mnist)

## Troubleshooting
* You might get a linking error while compiling these demos. Possible causes are:
  * You haven't compiled/installed the binaries of the _boost libraries_.
  * You have compiled the _boost libraries_ using a different compiler as the one used during compilation of these demos.
* `Eigen`-related errors.
  * If you get any `Éigen` error, the most probable cause is that you are including a different version of the Eigen library. Please be sure that only the supported version is included in your `ÍNCLUDE_PATH`
