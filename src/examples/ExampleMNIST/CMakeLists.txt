cmake_minimum_required(VERSION 3.0.2)
project(ExampleMNIST)

find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/../../../../")
add_definitions(-std=c++11 -Wall -DNDEBUG -O3)

add_executable(TrainMLP mainTrain.cpp)
target_link_libraries(TrainMLP ${OpenCV_LIBS})

add_executable(TestMLP mainTest.cpp)
target_link_libraries(TestMLP ${OpenCV_LIBS})