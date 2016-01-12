cmake_minimum_required(VERSION 3.0.2)
project(ExampleFeedforwardNet)

find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/../../../../")
add_definitions(-std=c++11 -Wall -DNDEBUG -O3)

add_executable(FeedForwardNet main.cpp)
target_link_libraries(FeedForwardNet ${OpenCV_LIBS})