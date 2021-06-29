cmake_minimum_required(VERSION 3.16.3)

project(xgrad CXX)

option(XGRAD_BUILD_TEST "build xgrad tests if ON" OFF)
option(XGRAD_BUILD_ALL "build all if ON" OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp)

if(XGRAD_BUILD_TEST OR XGRAD_BUILD_ALL)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests)
endif()
