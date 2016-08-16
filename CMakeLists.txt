cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
add_compile_options(-std=c++11 -O3 -fopenmp -fPIC)
project(rayOptimization)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# PRINT macro
macro(print)
  message(STATUS ${ARGN})
endmacro(print)

set(POINT_CLOUD_UTIL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/point_cloud_util")
set(SOURCE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src")

# include paths
include_directories(
  ${POINT_CLOUD_UTIL_INCLUDE_DIR}
  ${SOURCE_INCLUDE_DIR})

# Create the actual project
add_subdirectory(src)

# add libs
add_subdirectory(lib) 

MESSAGE(STATUS "compilation flags: " ${CMAKE_CXX_FLAGS})
