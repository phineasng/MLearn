cmake_minimum_required(VERSION 3.0.2)
project(TestGaussianProcess)

add_executable(TestGaussianProcess test_main.cpp)
add_test(NAME GaussianProcess COMMAND TestGaussianProcess)