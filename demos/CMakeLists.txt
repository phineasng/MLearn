cmake_minimum_required(VERSION 3.0.2)
project(MLearnDemos)

set(MLEARN_ROOT_DIR "../")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MLEARN_ROOT_DIR}/cmake)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O2 -Wall")

find_package(Boost REQUIRED filesystem program_options system iostreams)
include_directories($ENV{EIGEN3_ROOT})
include_directories($ENV{DEP_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(../${MLEARN_ROOT_DIR})
link_directories(${Boost_LIBRARY_DIRS})

add_subdirectory(demo_sampling)
add_subdirectory(demo_gaussian_process)