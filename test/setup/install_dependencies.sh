#!/bin/bash

# get dependencies from the repos
# - Eigen
echo 'Install Eigen ..'
echo '.. creating temporary directory to save zip ..'
sudo mkdir $TMP_DIR'/Eigen'
echo '.. downloading ..'
sudo wget https://bitbucket.org/eigen/eigen/get/f3a22f35b044.zip -O $TMP_DIR'/Eigen/eigen_repo.zip'
echo '.. unzipping ..'
sudo unzip $TMP_DIR'/Eigen/eigen_repo.zip' -d $TMP_DIR'/Eigen/'
echo '.. moving to include ..'
sudo mv $TMP_DIR'/Eigen/eigen-eigen-f3a22f35b044/Eigen' /usr/include/Eigen
echo '.. done!'