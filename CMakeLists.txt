cmake_minimum_required (VERSION 2.8)
project (clustering) 

# do we want static libraries?
# When DC_STATIC_LINKING is set, cmake looks for libraries called *.a.
if(DC_STATIC_LINKING)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif(DC_STATIC_LINKING)

# Gnu compiler setup
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  message("using Gnu compiler")
  # use standard C++2011 ('0x' synonymous, used for compatibility with older compilers)
  set (CMAKE_CXX_FLAGS "-std=c++0x")
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

  if (DC_NATIVE_COMPILATION)
    message("native compilation: this code will ONLY run on THIS machine")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
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
else()
  message(SEND_ERROR "currently no other compiler than the GCC is supported. sorry.")
endif()

if (DC_USE_OPENCL)
  if (DC_USE_MPI)
    message(SEND_ERROR "sorry, MPI and OpenCL are at the moment mutually exclusive. might change with a newer release.")
  endif()
  set (DC_USE_OPENCL "#define DC_USE_OPENCL")
  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -l OpenCL")
  find_package(OpenCL REQUIRED)
  if(OPENCL_HAS_CPP_BINDINGS)
  	message( "using OpenCL for GPGPU computing")
    # generate kernel headers with python
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/kernel)
    execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/generate_header.py
                                      ${CMAKE_SOURCE_DIR}/pops.cl
                                      ${CMAKE_BINARY_DIR}/kernel/pops.h)
    include_directories(${CMAKE_BINARY_DIR})
  else(OPENCL_HAS_CPP_BINDINGS)
  	message(SEND_ERROR "No OpenCL CPP bindings found" )
  endif(OPENCL_HAS_CPP_BINDINGS)
endif(DC_USE_OPENCL)


if(DC_USE_MPI)
  set (DC_USE_MPI "#define DC_USE_MPI")
  find_package(MPI REQUIRED)
  if (MPI_CXX_FOUND)
    message("using MPI for parallelization")
  else()
    message(SEND_ERROR "no MPI for C++ found!")
  endif()
endif()

configure_file(config.hpp.cmake.in ${CMAKE_BINARY_DIR}/generated/config.hpp)
include_directories(${CMAKE_BINARY_DIR}/generated/)


find_package (Boost 1.49 COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

add_subdirectory(coords_file)

set(CLUSTERING_SRCS clustering.cpp
                    density_clustering.cpp
                    density_clustering_common.cpp
                    mpp.cpp
                    network_builder.cpp
                    state_filter.cpp
                    coring.cpp
                    tools.cpp
                    logger.cpp)

if(DC_USE_MPI)
  include_directories("${MPI_CXX_INCLUDE_PATH}")
  add_executable(clustering_mpi ${CLUSTERING_SRCS} density_clustering_mpi.cpp)
  target_link_libraries(clustering_mpi ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} coords_file)
  set(CLUSTERING_BINARIES ${CLUSTERING_BINARIES} clustering_mpi)
elseif(DC_USE_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIRS})
  include_directories(${CMAKE_SOURCE_DIR})
  add_executable(clustering_gpu ${CLUSTERING_SRCS} density_clustering_opencl.cpp)
  target_link_libraries(clustering_gpu ${Boost_LIBRARIES} coords_file)
  set(CLUSTERING_BINARIES ${CLUSTERING_BINARIES} clustering_gpu)
else()
  add_executable(clustering ${CLUSTERING_SRCS})
  target_link_libraries(clustering ${Boost_LIBRARIES} coords_file)
  set(CLUSTERING_BINARIES ${CLUSTERING_BINARIES} clustering)
endif()

# set -static, when STATIC_LINKING is TRUE and set LINK_SEARCH_END_STATIC
# to remove the additional -bdynamic from the linker line.
if(DC_STATIC_LINKING)
    set(CMAKE_EXE_LINKER_FLAGS "-static")
    set_target_properties(clustering PROPERTIES LINK_SEARCH_END_STATIC 1)
#    set_target_properties(network_builder PROPERTIES LINK_SEARCH_END_STATIC 1)
#    set_target_properties(state_filter PROPERTIES LINK_SEARCH_END_STATIC 1)
#    set_target_properties(coring PROPERTIES LINK_SEARCH_END_STATIC 1)
endif(DC_STATIC_LINKING)


install(TARGETS ${CLUSTERING_BINARIES} RUNTIME DESTINATION .)


