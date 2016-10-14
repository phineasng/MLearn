cmake_minimum_required(VERSION 3.0.2)

project(CharLevelRNN)

FIND_PACKAGE(Boost 1.62.0 COMPONENTS program_options system filesystem REQUIRED)

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
include_directories("${PROJECT_SOURCE_DIR}/../../../../")

add_definitions(-std=c++11 -Wall -O3)

add_executable(CharLevelRNN main.cpp)
TARGET_LINK_LIBRARIES(CharLevelRNN ${Boost_LIBRARIES} )
add_executable(CharLevelRNNSample main_sample.cpp)
TARGET_LINK_LIBRARIES(CharLevelRNNSample ${Boost_LIBRARIES} )

