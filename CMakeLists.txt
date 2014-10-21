cmake_minimum_required (VERSION 2.8)
project (clustering) 

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

# Intel compiler setup
#elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
#  message("using Intel compiler")
#  set (CMAKE_CXX_FLAGS "-std=c++11 -openmp -xAVX -vec-report=1 -O3")
#  set (DC_MEM_ALIGNMENT 32)
#endif()

#if (DC_USE_OPENCL)
#  set (DC_USE_OPENCL "#define DC_USE_OPENCL")
#  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
#  find_package(OpenCL REQUIRED)
#  include_directories(${OPENCL_INCLUDE_DIRS})
#  if( OPENCL_HAS_CPP_BINDINGS )
#  	message( "OpenCL has CPP bindings. Full include is: " ${OPENCL_INCLUDE_DIRS} )
#  else( OPENCL_HAS_CPP_BINDINGS )
#  	message( "No OpenCL CPP bindings found" )
#  endif( OPENCL_HAS_CPP_BINDINGS )
#  include_directories("${CMAKE_SOURCE_DIR}")
#else()
#  set (DC_USE_OPENCL "")
#endif(DC_USE_OPENCL)


if(DC_USE_MPI)
  set (DC_USE_MPI "#define DC_USE_MPI")
  find_package(MPI REQUIRED)
  if (MPI_CXX_FOUND)
    message("using MPI for parallelization")
  else()
    message(SEND_ERROR "no MPI for C++ found!")
  endif()
endif()


find_package (Boost 1.49 COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

configure_file(config.hpp.cmake.in ${CMAKE_BINARY_DIR}/generated/config.hpp)
include_directories(${CMAKE_BINARY_DIR}/generated/)

set(CLUSTERING_SRCS clustering.cpp density_clustering.cpp density_clustering_common.cpp mpp.cpp tools.cpp logger.cpp)
set(CLUSTERING_BINARIES network_builder state_filter)

add_executable(network_builder network_builder.cpp tools.cpp logger.cpp)
target_link_libraries(network_builder ${Boost_LIBRARIES})

add_executable(state_filter state_filter.cpp)
target_link_libraries(state_filter ${Boost_LIBRARIES})

if(DC_USE_MPI)
  include_directories("${MPI_CXX_INCLUDE_PATH}")
  add_executable(clustering_mpi ${CLUSTERING_SRCS} density_clustering_mpi.cpp)
  target_link_libraries(clustering_mpi ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set(CLUSTERING_BINARIES ${CLUSTERING_BINARIES} clustering_mpi)
else()
  add_executable(clustering ${CLUSTERING_SRCS})
  target_link_libraries(clustering ${Boost_LIBRARIES})
  set(CLUSTERING_BINARIES ${CLUSTERING_BINARIES} clustering)
endif()

install(TARGETS ${CLUSTERING_BINARIES} RUNTIME DESTINATION .)

