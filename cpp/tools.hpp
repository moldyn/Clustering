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

struct CoordsDeleter {
  void operator()(void* x) {
    _mm_free(x);
  }
};

template<typename NUM>
using CoordsPointer = std::unique_ptr<NUM, CoordsDeleter>;

// read coordinates from space-separated ASCII file.
// will write data with precision of MY_FLOAT into memory.
// format: [row * n_cols + col]
// return value: tuple of {data (unique_ptr<NUM> with custom deleter), n_rows (size_t), n_cols (size_t)}.
template <typename NUM>
std::tuple<CoordsPointer<NUM>, std::size_t, std::size_t>
read_coords(std::string filename,
            std::vector<std::size_t> usecols = {});

// template implementations
#include "tools.hxx"

