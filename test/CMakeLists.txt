cmake_minimum_required(VERSION 3.0.2)
project(TestRoot)

option(COVERALLS "Turn on coveralls support" OFF)
option(COVERALLS_UPLOAD "Upload the generated coveralls json" ON)

set(MLEARN_ROOT_DIR "..")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MLEARN_ROOT_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} $ENV{TEST_CMAKE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (COVERALLS)
	include(Coveralls)
	file(GLOB_RECURSE SRCS $ENV{TRAVIS_BUILD_DIR}/*.h)
	string(REGEX REPLACE "$ENV{TEST_INCLUDE_PATH}/[^;]+;?" "" SRCS "${SRCS}")
	string(REGEX REPLACE "$ENV{TEST_SRC_PATH}/[^;]+;?" "" SRCS "${SRCS}")
	coveralls_turn_on_coverage()
	coveralls_setup("${SRCS}" ${COVERALLS_UPLOAD})
endif()

include_directories(${MLEARN_ROOT_DIR})
include_directories($ENV{TEST_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_subdirectory(test_kernels)