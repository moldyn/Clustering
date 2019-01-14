# Copyright (c) 2015, Florian Sittel (www.lettis.net)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 2.8)
project (clustering) 

# do we want static libraries?
# When STATIC_LINKING is set, cmake looks for libraries called *.a.
if(STATIC_LINKING)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif(STATIC_LINKING)

# Gnu compiler setup
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  message("using Gnu compiler")
  # use C++ 2011 standard
  set (CMAKE_CXX_FLAGS "-std=c++11")
  # profiling
  #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
  # optimization
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  # parallelization
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  # warnings
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-declarations")

  if (NATIVE_COMPILATION)
    message("native compilation: this code will ONLY run on THIS machine")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
  set (DC_MEM_ALIGNMENT 32)
  if (CPU_ACCELERATION)
    if (${CPU_ACCELERATION} STREQUAL "SSE2")
      message("using SSE2")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
      set (DC_MEM_ALIGNMENT 16)

    elseif (${CPU_ACCELERATION} STREQUAL "SSE4_1")
      message("using SSE4_1")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
      set (DC_MEM_ALIGNMENT 16)

    elseif (${CPU_ACCELERATION} STREQUAL "SSE4_2")
      message("using SSE4_2")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
      set (DC_MEM_ALIGNMENT 16)

    elseif (${CPU_ACCELERATION} STREQUAL "AVX")
      message("using AVX")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
      set (DC_MEM_ALIGNMENT 32)
    else()
      message(SEND_ERROR "unknown instruction set: ${CPU_ACCELERATION}")
    endif()
  else()
    message("using standard CPU acceleration with memory aligned at ${DC_MEM_ALIGNMENT} bytes")
  endif()
else()
  message(SEND_ERROR "currently no other compiler than the GCC is supported. sorry.")
endif()



configure_file(config.hpp.cmake.in ${CMAKE_BINARY_DIR}/generated/config.hpp)
include_directories(${CMAKE_BINARY_DIR}/generated/)

find_package (Boost 1.49 COMPONENTS program_options system filesystem REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

add_subdirectory(src/coords_file)


set(CLUSTERING_SRCS src/clustering.cpp
                    src/density_clustering.cpp
                    src/density_clustering_common.cpp
                    src/mpp.cpp
                    src/network_builder.cpp
                    src/state_filter.cpp
                    src/coring.cpp
                    src/noise.cpp
                    src/tools.cpp
                    src/logger.cpp)

set(CLUSTERING_LIBS ${Boost_LIBRARIES} coords_file)

if(${USE_CUDA})
  message(STATUS "using CUDA")
  find_package(CUDA REQUIRED)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11;-O3;-DVERBOSE;-Xcompiler -fopenmp")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=compute_30")
  # debugging
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g -G")

  set (CLUSTERING_SRCS ${CLUSTERING_SRCS}
                       src/density_clustering_cuda.cu
                       src/density_clustering_cuda_kernels.cu)

  set (CLUSTERING_LIBS ${CLUSTERING_LIBS} ${CUDA_LIBRARIES})
  include_directories (${CUDA_INCLUDE_DIRS})
  add_definitions(-DUSE_CUDA)
endif()


set (PROGNAME clustering)


if (${USE_CUDA})
  #  set (PROGNAME ${PROGNAME}.cuda)
  cuda_add_executable(${PROGNAME} ${CLUSTERING_SRCS})
else()
  add_executable(${PROGNAME} ${CLUSTERING_SRCS})
endif()

target_link_libraries(${PROGNAME} ${CLUSTERING_LIBS})


# set -static, when STATIC_LINKING is TRUE and set LINK_SEARCH_END_STATIC
# to remove the additional -bdynamic from the linker line.
if(STATIC_LINKING)
    set(CMAKE_EXE_LINKER_FLAGS "-static")
    set_target_properties(${PROGNAME} PROPERTIES LINK_SEARCH_END_STATIC 1)
endif(STATIC_LINKING)


install(TARGETS ${PROGNAME} RUNTIME DESTINATION .)

# create source-doc with doxygen
add_custom_target(doc
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  COMMAND doxygen doxygen.config)

