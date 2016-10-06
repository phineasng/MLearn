cmake_minimum_required(VERSION 3.0.2)
project(ExampleKMeans)
include_directories("${PROJECT_SOURCE_DIR}/../../../../")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_definitions(-std=c++11 -Wall -O3)

add_executable(KMeansExample main.cpp)
