# MLearn
[![Build Status](https://travis-ci.org/phineasng/MLearn.svg?branch=master)](https://travis-ci.org/phineasng/MLearn)
[![Coverage Status](https://coveralls.io/repos/github/phineasng/MLearn/badge.svg?branch=master)](https://coveralls.io/github/phineasng/MLearn?branch=master)
## Description
This template library is a personal project to refresh and deepen my knowledge of Machine Learning theory and algorithms.  

## Algorithms implemented
- Machine Learning
  1. Gaussian processes
  2. Neural networks (Fully connected Networks)
  3. K-means (untested)
- Optimization (untested)
  1. SGD and variants (untested)
  2. BFGS (untested)

## Dependencies
* c++11 ( g++-4.9.2 )
* Eigen == 3.3.3
* Boost == 1.64.0 (headers-only)

## Demos
- [Demo](https://github.com/phineasng/MLearn/tree/master/demos) 
  - A bash script is provided to setup eveything necessary to run the demos.

## Snapshots from experiments
* Gaussian Process regression. Image from this [demo](https://github.com/phineasng/MLearn/tree/master/demos/demo_regression)
![](https://github.com/phineasng/MLearn/blob/master/demos/demo_regression/img/gp_regression.png)
* Filters learned by a Bernoulli-Bernoulli RBM on the MNIST dataset (19 epochs)
![](https://github.com/phineasng/MLearn/blob/master/misc/imgs/BernoulliBernoulliRBM_19epochs.png)
* Classification on MNIST with a 3-layer NN. Image from this [demo](https://github.com/phineasng/MLearn/tree/master/demos/demo_mnist)
![](https://github.com/phineasng/MLearn/blob/master/demos/demo_mnist/img/demo_mnist.png)

### Disclaimer
When I started this library it was mostly to have some fun implementing deep learning methods and C++ templates.
One consequence is that I haven't spent much time testing those algorithms, but they seem to be working. 
In the future, I will slowly redesign those parts and add the relevant tests.  
