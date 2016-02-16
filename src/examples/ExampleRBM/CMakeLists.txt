cmake_minimum_required(VERSION 3.0.2)
project(ExampleRBM)

find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/../../../../")
add_definitions(-std=c++11 -Wall -DNDEBUG -O3)

add_executable(RBM main.cpp)

add_executable(RBMMNIST mainMNIST.cpp)
target_link_libraries(RBMMNIST ${OpenCV_LIBS})