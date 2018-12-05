/*
Copyright (c) 2015, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
//! additional tools used throughout the *clustering* package
namespace Tools {
  //! matches neighbor's frame id to distance
  using Neighbor = std::pair<std::size_t, float>;
  //! map frame id to neighbors
  using Neighborhood = std::map<std::size_t, Clustering::Tools::Neighbor>;
  //! write populations as column into given file
  void
  write_pops(std::string fname, std::vector<std::size_t> pops, std::string header_comment);
  //! write free energies as column into given file
  void
  write_fes(std::string fname, std::vector<float> fes, std::string header_comment);
  //! read states from trajectory (given as plain text file)
  std::vector<std::size_t>
  read_clustered_trajectory(std::string filename);
  //! write state trajectory into plain text file
  void
  write_clustered_trajectory(std::string filename, std::vector<std::size_t> traj,
                             std::string header_comment);
  //! read single column of numbers from given file. number type (int, float, ...) given as template parameter
  template <typename NUM>
  std::vector<NUM>
  read_single_column(std::string filename);
  //! write single column of numbers to given file. number type (int, float, ...) given as template parameter
  template <typename NUM>
  void
  write_single_column(std::string filename, std::vector<NUM> dat,
                      std::string header_comment, bool with_scientific_format=false);
  //! write key-value map to plain text file with key as first and value as second column
  template <typename KEY, typename VAL>
  void
  write_map(std::string filename, std::map<KEY, VAL> mapping,
            std::string header_comment, bool val_then_key=false);
  //! read free energies from plain text file
  std::vector<float>
  read_free_energies(std::string filename);
  //! read neighborhood info from plain text file
  //! (two different neighborhoods: nearest neighbor (NN) and NN with higher density)
  std::pair<Neighborhood, Neighborhood>
  read_neighborhood(const std::string fname);
  //! write neighborhood info to plain text file
  //! (two different neighborhoods: nearest neighbor (NN) and NN with higher density)
  void
  write_neighborhood(const std::string fname,
                     const Neighborhood& nh,
                     const Neighborhood& nh_high_dens,
                     std::string header_comment);
  //! compute microstate populations from clustered trajectory
  std::map<std::size_t, std::size_t>
  microstate_populations(std::vector<std::size_t> traj);
  //! read coordinates from space-separated ASCII file.
  //! will write data with precision of NUM-type into memory.
  //! format: [row * n_cols + col]
  //! return value: tuple of {data (unique_ptr<NUM> with custom deleter), n_rows (size_t), n_cols (size_t)}.
  template <typename NUM>
  std::tuple<NUM*, std::size_t, std::size_t>
  read_coords(std::string filename,
              std::vector<std::size_t> usecols = std::vector<std::size_t>());
  //! free memory pointing to coordinates
  template <typename NUM>
  void
  free_coords(NUM* coords);
  //! return std::vector with coords sorted along first dimension.
  //! uses row-based addressing (row*n_cols+col).
  template <typename NUM>
  std::vector<NUM>
  dim1_sorted_coords(const NUM* coords
                   , std::size_t n_rows
                   , std::size_t n_cols);
  //! separate into equally sized boxes and return min values of
  //! first dimension for given box.
  //! used for pruning in CUDA-accelerated code.
  template <typename NUM>
  std::vector<NUM>
  boxlimits(const std::vector<NUM>& xs
          , std::size_t boxsize
          , std::size_t n_rows
          , std::size_t n_cols);
  //! return indices of min and max boxes around value for given radius.
  template <typename NUM>
  std::pair<std::size_t, std::size_t>
  min_max_box(const std::vector<NUM>& limits
            , NUM val
            , NUM radius);
  //! return minimum multiplicator to fulfill result * mult >= orig
  unsigned int
  min_multiplicator(unsigned int orig
                  , unsigned int mult);
  //! printf-version for std::string
  std::string
  stringprintf(const std::string& str, ...);
  //! convert std::string to number of given template format
  template <typename NUM>
  NUM
  string_to_num(const std::string &s);
  //! return distinct elements of vector
  template <typename T>
  std::vector<T>
  unique_elements(std::vector<T> xs);
} // end namespace 'Tools'
} // end namespace 'Clustering'

// template implementations
#include "tools.hxx"

