# MLearn
[![Build Status](https://travis-ci.org/phineasng/MLearn.svg?branch=master)](https://travis-ci.org/phineasng/MLearn)
[![Coverage Status](https://coveralls.io/repos/github/phineasng/MLearn/badge.svg?branch=master)](https://coveralls.io/github/phineasng/MLearn?branch=master)
## Description
This template library is a personal project to refresh and deepen my knowledge of Machine Learning theory and algorithms.  

## Algorithms implemented
- Machine Learning
  1. Gaussian processes
  2. Neural networks (Feedforward Networks, RNN, RBM) (untested)
  3. K-means (untested)
- Optimization
  1. SGD and variants (untested)
  2. BFGS

## Dependencies
* c++11 ( g++-4.9.2 )
* Eigen == 3.3.3
* Boost == 1.64.0 (headers-only)

## Demos
- [Demo]() 
  -A bash script is provided to setup eveything necessary to run the demos.
- [Examples]() 
  - Mostly boilerplate code. While I am providing a cmake file for each examples, 
  they might be cumbersome to run due to lack of documentation, not provided data, etc...
  Contact me in case you want to run those examples and you encounter some problems. 

## Snapshots from experiments
* Filters learned by a Bernoulli-Bernoulli RBM on the MNIST dataset (19 epochs)
![](https://github.com/phineasng/MLearn/blob/master/misc/imgs/BernoulliBernoulliRBM_19epochs.png)

### Disclaimer
When I started this library it was mostly to have some fun implementing deep learning methods and C++ templates.
One consequence is that I haven't spent much time testing those algorithms, but they seem to be working. 
In the future, I will slowly redesign those parts and add the relevant tests.  
