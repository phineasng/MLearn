cmake_minimum_required(VERSION 3.0.2)
project(TestRegression)

add_executable(TestRegression test_main.cpp)
add_test(NAME Regression COMMAND TestRegression)