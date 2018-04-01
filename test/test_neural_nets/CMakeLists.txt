cmake_minimum_required(VERSION 3.0.2)
project(TestNeuralNets)

add_executable(TestNeuralNets test_main.cpp)
add_test(NAME NeuralNets COMMAND TestNeuralNets)