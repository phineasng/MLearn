cmake_minimum_required(VERSION 3.0.2)
project(ExampleFeedforwardNet)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../)
find_package(Eigen3 REQUIRED)

include_directories("${EIGEN3_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../../../../")
add_definitions(-std=c++11 -Wall)

add_executable(FeedForwardNet main.cpp)