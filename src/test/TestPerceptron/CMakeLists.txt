cmake_minimum_required(VERSION 3.0.2)
project(TestPerceptron)

include_directories("${PROJECT_SOURCE_DIR}/../../../../")
add_definitions(-std=c++11)

add_executable(Perceptron main.cpp)