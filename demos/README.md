# MLearn Demos

## Dependencies

To run the script you need to have [CMake](https://cmake.org/) installed.
The dependencies of the library will be installed by the script if required.
Further dependencies for the demo (the script will always try to install these):
- [gnuplot](http://www.gnuplot.info/)
- [gnuplot-iostream](http://stahlke.org/dan/gnuplot-iostream/)
- [boost compiled libraries](http://www.boost.org/)

## Setup

To setup the correct dependencies, you just need to run the `setup.sh`
from this folder, i.e. `/path/to/MLearn/demos`. 
If the options are correctly set, the script should perform these steps:
1. Create a build directory
2. Download/setup dependencies
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

For example, to install all the dependencies from a clean build directory:
```bash
bash setup.sh --clean --all
```

Now you can `cd` into the build folder `demo_build`. If you run the `make`
command here, you will build all the demos. You can also specify a single demo
to be built, e.g. `make MultivariateSamplingDemo`.

## Available demos
Here I will try to maintain a list of available demos:
1. [Multivariate Gaussian sampling]()
2. [Gaussian Processes sampling]()
