cmake_minimum_required(VERSION 3.16.3)

project(test_xgrad CXX)

file(GLOB_RECURSE test_xgrad_src
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include(../../cmake/XGradAddCompileOptions.cmake)

add_executable(test_xgrad ${test_xgrad_src})
target_link_libraries(test_xgrad PRIVATE xgrad CppUTest)
xgrad_add_compile_options(test_xgrad)
