cmake_minimum_required (VERSION 2.8)
project (supervised_clustering) 

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  message("using Clang compiler (ATTENTION: only rudimentary support)")
  set (DC_MEM_ALIGNMENT 32)
  set (CMAKE_CXX_FLAGS "-std=c++11 -O2 -fopenmp -Wall -ffast-math -fvectorize")



elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  message("using Gnu compiler")
  if (DC_NATIVE_COMPILATION)
    message("native compilation: this code will ONLY run on THIS machine")
    set (CMAKE_CXX_FLAGS "-std=c++0x -march=native -ftree-vectorize -ftree-vectorizer-verbose=1 -ffast-math -O3 -fopenmp -Wall -Wextra -Wno-unknown-pragmas")
  else()
    set (CMAKE_CXX_FLAGS "-std=c++0x -ftree-vectorize -ffast-math -O2 -fopenmp -Wall -Wextra -Wno-unknown-pragmas")
  endif()
  if (DC_USE_XEON_PHI)
    message(SEND_ERROR "XeonPhi not supported with Gnu compiler")
  endif()
  set (DC_MEM_ALIGNMENT 32)
  if (DC_CPU_ACCELERATION)
    if (${DC_CPU_ACCELERATION} STREQUAL "SSE2")
      message("using SSE2")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
      set (DC_MEM_ALIGNMENT 16)

    elseif (${DC_CPU_ACCELERATION} STREQUAL "SSE4_1")
      message("using SSE4_1")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
      set (DC_MEM_ALIGNMENT 16)

    elseif (${DC_CPU_ACCELERATION} STREQUAL "SSE4_2")
      message("using SSE4_2")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
      set (DC_MEM_ALIGNMENT 16)

    elseif (${DC_CPU_ACCELERATION} STREQUAL "AVX")
      message("using AVX")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
      set (DC_MEM_ALIGNMENT 32)
    else()
      message(SEND_ERROR "unknown instruction set: ${DC_CPU_ACCELERATION}")
    endif()
  else()
    message("using standard CPU acceleration with memory aligned at ${DC_MEM_ALIGNMENT} bytes")
  endif()



elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  message("using Intel compiler")

  if (DC_XEON_PHI_NATIVE)
    set (CMAKE_CXX_FLAGS "-std=c++11 -mmic -openmp -O3")
    set (DC_MEM_ALIGNMENT 32)
  else()
    set (CMAKE_CXX_FLAGS "-std=c++11 -openmp -xAVX -vec-report=1 -O3")
    set (DC_MEM_ALIGNMENT 32)
  endif()

endif()

find_package (Boost COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

add_executable(density_clustering density_clustering.cpp)
target_link_libraries(density_clustering ${Boost_LIBRARIES})

configure_file(config.hpp.cmake.in ${CMAKE_BINARY_DIR}/generated/config.hpp)
include_directories(${CMAKE_BINARY_DIR}/generated/)

#add_executable(test_read_coords test_read_coords.cpp)

