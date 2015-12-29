cmake_minimum_required(VERSION 3.0.2)
project(TestCostFunction)

include_directories("${PROJECT_SOURCE_DIR}/../../../../")
add_definitions(-std=c++11 -Wall -O3)

add_executable(CustomCost main.cpp)