#!/bin/bash

# get dependencies from the repos
# - Eigen
mkdir $TMP_DIR'/Eigen'
wget https://bitbucket.org/eigen/eigen/get/f3a22f35b044.zip -O $TMP_DIR'/Eigen/eigen_repo.zip'
unzip $TMP_DIR'/Eigen/eigen_repo.zip' -d $TMP_DIR'/Eigen/'
sudo mv $TMP_DIR'/Eigen/eigen-eigen-f3a22f35b044/Eigen' $INCLUDE_PATH'/Eigen'