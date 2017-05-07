cmake_minimum_required(VERSION 3.0.2)
project(TestBase)

add_executable(TestBase test_main.cpp)
add_test(NAME Base COMMAND TestBase)