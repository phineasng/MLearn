# MLearn
[![Build Status](https://travis-ci.org/phineasng/MLearn.svg?branch=master)](https://travis-ci.org/phineasng/MLearn)
[![Coverage Status](https://coveralls.io/repos/github/phineasng/MLearn/badge.svg?branch=travis_and_coveralls)](https://coveralls.io/github/phineasng/MLearn?branch=travis_and_coveralls)
## Description
This library is a personal project to refresh and deepen the knowledge of Machine Learning theory and algorithms. 
Due to a recently developed interest in deep learning, the library may reveal a particular focus on topics from this subfield.

## Algorithms implemented
1. Perceptron
2. FeedForward Nets (backprop + L1,L2,Dropout regularizations)
3. RBM

## Dependencies
* c++11 ( library tested using g++-4.9.2 )
* Eigen >= 3.2.7

### Dependencies for the examples
* openCV

## Contributors
* phineasng
* frenaut

## Snapshots from experiments
* Filters learned by a Bernoulli-Bernoulli RBM on the MNIST dataset (19 epochs)
![](https://github.com/phineasng/MLearn/blob/master/misc/imgs/BernoulliBernoulliRBM_19epochs.png)