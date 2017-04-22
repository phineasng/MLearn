cmake_minimum_required(VERSION 3.0.2)
project(TestRoot)

option(COVERALLS "Turn on coveralls support" OFF)
option(COVERALLS_UPLOAD "Upload the generated coveralls json" ON)

set(MLEARN_ROOT_DIR "..")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MLEARN_ROOT_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} $ENV{TEST_CMAKE})

if (COVERALLS)
	include(Coveralls)
	file(GLOB_RECURSE SRCS *.cpp)
	coveralls_turn_on_coverage()
	coveralls_setup("${SRCS}" ${COVERALLS_UPLOAD})
endif()

include_directories(${MLEARN_ROOT_DIR})
include_directories($ENV{TEST_INCLUDE_PATH})

add_subdirectory(dummy_test_eigen)