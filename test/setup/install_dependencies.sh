#!/bin/bash

# get dependencies from repos

# - Eigen
export TMP_EIGEN=$TMP_DIR/Eigen
mkdir $TMP_EIGEN
wget https://bitbucket.org/eigen/eigen/get/f3a22f35b044.zip -O $TMP_EIGEN'/eigen_repo.zip'
unzip -qq $TMP_EIGEN'/eigen_repo.zip' -d $TMP_EIGEN
mv $TMP_EIGEN'/eigen-eigen-f3a22f35b044/Eigen' $TEST_INCLUDE_PATH'/Eigen'

# - Coveralls
export TMP_COVERALLS=$TMP_DIR/CoverallsCMake
mkdir $TMP_COVERALLS
wget https://github.com/JoakimSoderberg/coveralls-cmake/archive/master.zip -O $TMP_COVERALLS'/coveralls.zip'
unzip -qq $TMP_COVERALLS'/coveralls.zip' -d $TMP_COVERALLS
mv $TMP_COVERALLS'/coveralls-cmake-master/cmake'/* $TEST_CMAKE'/'

# - Catch (testing framework)
wget https://github.com/philsquared/Catch/releases/download/v1.9.1/catch.hpp -O $TEST_INCLUDE_PATH'/'