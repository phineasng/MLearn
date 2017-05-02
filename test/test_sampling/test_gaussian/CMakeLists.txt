cmake_minimum_required(VERSION 3.0.2)
project(TestGaussianSampling)

add_executable(TestGaussianSampling test_main.cpp)
add_test(NAME GaussianSampling COMMAND TestGaussianSampling)