cmake_minimum_required(VERSION 3.16.3)

project(xgrad CXX)

file(GLOB_RECURSE xgrad_src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xgrad/*.cpp)

include(../cmake/XGradAddCompileOptions.cmake)

add_library(xgrad ${xgrad_src})
target_include_directories(xgrad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
xgrad_add_compile_options(xgrad)
