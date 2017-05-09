# Gaussian Process Regression demo

This demo will perform regression using gaussian process. The function to be fitted
is `x*sin(x) + exp(-x^2)`.

## Setup 

Be sure to run the [setup](https://github.com/phineasng/MLearn/tree/master/demos) steps.
Compile specifically this demo from the build directory with `make GPRegressionDemo`.

## How to run the demo

This demo can be run from the correct subfolder with the command
`./GPRegressionDemo`. You can control the noise level of the samples
with the `--noise` flag.

```bash
./GPRegressionDemo --help
--noise arg           Noise variance (default: 0.0).
```
![](https://github.com/phineasng/MLearn/blob/master/demos/demo_regression/img/gp_regression.png)
