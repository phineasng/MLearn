cmake_minimum_required(VERSION 3.0.2)
project(TestKernels)

add_executable(TestKernels test_main.cpp)
add_test(NAME Kernels COMMAND TestKernels)