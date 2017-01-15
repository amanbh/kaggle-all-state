cmake_minimum_required(VERSION 2.8)
project(GP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(GP ${SOURCE_FILES})