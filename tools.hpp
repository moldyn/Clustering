/*
Copyright (c) 2015, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "config.hpp"

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <memory>
#include <iostream>

/// needed for aligned memory allocation for Xeon Phi, SSE or AVX
#if defined(__INTEL_COMPILER)
 #include <malloc.h>
#else
 #include <mm_malloc.h>
#endif

#if defined(__INTEL_COMPILER)
  #define ASSUME_ALIGNED(c) __assume_aligned( (c), DC_MEM_ALIGNMENT)
#else
  #define ASSUME_ALIGNED(c) (c) = (float*) __builtin_assume_aligned( (c), DC_MEM_ALIGNMENT)
#endif

namespace Clustering {
namespace Tools {

using Neighborhood = std::map<std::size_t, std::pair<std::size_t, float>>;

//TODO doc
void
write_pops(std::string fname, std::vector<std::size_t> pops);

//TODO doc
void
write_fes(std::string fname, std::vector<float> fes);

//TODO doc
std::vector<std::size_t>
read_clustered_trajectory(std::string filename);

//TODO doc
void
write_clustered_trajectory(std::string filename, std::vector<std::size_t> traj);

//TODO doc
template <typename NUM>
std::vector<NUM>
read_single_column(std::string filename);

//TODO doc
template <typename NUM>
void
write_single_column(std::string filename, std::vector<NUM> dat, bool with_scientific_format=false);

template <typename KEY, typename VAL>
void
write_map(std::string filename, std::map<KEY, VAL> mapping);

std::vector<float>
read_free_energies(std::string filename);

//TODO doc
std::pair<Neighborhood, Neighborhood>
read_neighborhood(const std::string fname);

//TODO doc
void
write_neighborhood(const std::string fname,
                   const Neighborhood& nh,
                   const Neighborhood& nh_high_dens);

//TODO doc
std::map<std::size_t, std::size_t>
microstate_populations(std::vector<std::size_t> traj);

// read coordinates from space-separated ASCII file.
// will write data with precision of NUM-type into memory.
// format: [row * n_cols + col]
// return value: tuple of {data (unique_ptr<NUM> with custom deleter), n_rows (size_t), n_cols (size_t)}.
template <typename NUM>
std::tuple<NUM*, std::size_t, std::size_t>
read_coords(std::string filename,
            std::vector<std::size_t> usecols = std::vector<std::size_t>());

template <typename NUM>
void
free_coords(NUM* coords);

std::string
stringprintf(const std::string& str, ...);

template <typename NUM>
NUM
string_to_num(const std::string &s);

} // end namespace 'Tools'
} // end namespace 'Clustering'

// template implementations
#include "tools.hxx"

