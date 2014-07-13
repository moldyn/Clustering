cmake_minimum_required (VERSION 2.8)
project (supervised_clustering) 

# TODO: set march, fast-math, avx, fma, etc.

set (CMAKE_CXX_COMPILER "g++")
set (CMAKE_CXX_FLAGS "-std=c++11 -march=native -ftree-vectorize -ffast-math -O3 -fopenmp -Wall -Wextra")

#set (CMAKE_CXX_COMPILER "icpc")
#set (CMAKE_CXX_FLAGS "-std=c++11 -openmp -O3")
#set (CMAKE_CXX_FLAGS "-std=c++11 -openmp -O3 -prof-gen=srcpos")

find_package (Boost COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

add_executable(density_clustering density_clustering.cpp)
target_link_libraries(density_clustering ${Boost_LIBRARIES})

# COMPILER: g++ or icc
# DC_CPU_ACCELERATION: SSE2 SSE4_1 AVX NONE
# DC_XEON_PHI: ON OFF

configure_file(config.hpp.cmake.in ${CMAKE_BINARY_DIR}/generated/config.hpp)
include_directories(${CMAKE_BINARY_DIR}/generated/)

add_executable(test_read_coords test_read_coords.cpp)

