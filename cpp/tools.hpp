#pragma once

#include "config.hpp"

#include <string>
#include <vector>
#include <tuple>
#include <memory>

/// needed for aligned memory allocation for Xeon Phi, SSE or AVX
#if defined(__INTEL_COMPILER)
 #include <malloc.h>
#else
 #include <mm_malloc.h>
#endif

#if defined(__INTEL_COMPILER_)
  #define ASSUME_ALIGNED(c) __assume_aligned( (c), DC_MEM_ALIGNMENT)
#else
  #define ASSUME_ALIGNED(c) (c) = (float*) __builtin_assume_aligned( (c), DC_MEM_ALIGNMENT)
#endif

// read coordinates from space-separated ASCII file.
// will write data with precision of MY_FLOAT into memory.
// format: [row * n_cols + col]
// return value: tuple of {data (unique_ptr<NUM> with custom deleter), n_rows (size_t), n_cols (size_t)}.
template <typename NUM>
std::tuple<NUM*, std::size_t, std::size_t>
read_coords(std::string filename,
            std::vector<std::size_t> usecols = std::vector<std::size_t>());

template <typename NUM>
void
free_coords(NUM* coords);

// template implementations
#include "tools.hxx"

