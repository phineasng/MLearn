cmake_minimum_required(VERSION 3.0.2)

project(ExampleFaceCat)
find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}/../../../../")
add_definitions(-std=c++11 -Wall -O3)

add_executable(CatFaceAutoEncoderTrain main.cpp)
target_link_libraries(CatFaceAutoEncoderTrain ${OpenCV_LIBS})

add_executable(CatFaceAutoEncoderVisualize main2.cpp)
target_link_libraries(CatFaceAutoEncoderVisualize ${OpenCV_LIBS})
