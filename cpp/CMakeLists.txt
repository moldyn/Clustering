cmake_minimum_required (VERSION 2.8)
project (supervised_clustering) 

# enable profiling flags for gprof
#set (CMAKE_CXX_FLAGS "-std=c++0x -fopenmp -pg -g")



# TODO: set march, fast-math, avx, fma, etc.

set (CMAKE_CXX_COMPILER "g++")
set (CMAKE_CXX_FLAGS "-std=c++11 -march=native -mavx -ftree-vectorize -fopenmp -ffast-math -O3 -Wall -Wextra")

#set (CMAKE_CXX_COMPILER "icpc")
#set (CMAKE_CXX_FLAGS "-std=c++11 -openmp -O3")
#set (CMAKE_CXX_FLAGS "-std=c++11 -openmp -O3 -prof-gen=srcpos")

find_package (Boost COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

add_executable(density_clustering density_clustering.cpp)
target_link_libraries(density_clustering ${Boost_LIBRARIES})


#add_executable(neighborhood neighborhood.cpp)
#target_link_libraries(neighborhood ${Boost_LIBRARIES})
#
#add_executable(density density.cpp)
#target_link_libraries(density ${Boost_LIBRARIES})
