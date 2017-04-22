cmake_minimum_required(VERSION 3.0.2)
project(TestRoot)

set(MLEARN_ROOT_DIR "..")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MLEARN_ROOT_DIR}/cmake)
include_directories(${MLEARN_ROOT_DIR})
include_directories($ENV{INCLUDE_PATH})

add_subdirectory(dummy_test_eigen)