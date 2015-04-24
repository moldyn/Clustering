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
#include "density_clustering.hpp"

#include <vector>

#include <boost/program_options.hpp>

namespace Clustering {
namespace Density {
//! MPI implementations of compute intensive functions.
namespace MPI {
  //! identify MPI process 0 as main process
  const int MAIN_PROCESS = 0;
  //! MPI implementation of
  //! Clustering::Density::calculate_populations(const float* coords, const std::size_t n_rows, const std::size_t n_cols, const float radius)
  std::vector<std::size_t>
  calculate_populations(const float* coords,
                        const std::size_t n_rows,
                        const std::size_t n_cols,
                        const float radius,
                        const int mpi_n_nodes,
                        const int mpi_node_id);
  //! MPI implementation of
  //! Clustering::Density::calculate_populations(const float* coords, const std::size_t n_rows, const std::size_t n_cols, const std::vector<float> radii)
  std::map<float, std::vector<std::size_t>>
  calculate_populations(const float* coords,
                        const std::size_t n_rows,
                        const std::size_t n_cols,
                        std::vector<float> radii,
                        const int mpi_n_nodes,
                        const int mpi_node_id);
  //! MPI implementation of
  //! Clustering::Density::nearest_neighbors
  std::tuple<Neighborhood, Neighborhood>
  nearest_neighbors(const float* coords,
                    const std::size_t n_rows,
                    const std::size_t n_cols,
                    const std::vector<float>& free_energy,
                    const int mpi_n_nodes,
                    const int mpi_node_id);
  //! MPI implementation of
  //! Clustering::Density::high_density_neighborhood
  std::set<std::size_t>
  high_density_neighborhood(const float* coords,
                            const std::size_t n_cols,
                            const std::vector<FreeEnergy>& sorted_fe,
                            const std::size_t i_frame,
                            const std::size_t limit,
                            const float max_dist,
                            const int mpi_n_nodes,
                            const int mpi_node_id);
  //! MPI implementation of
  //! Clustering::Density::main
  void
  main(boost::program_options::variables_map args);
} // end namespace MPI
} // end namespace Density
} // end namespace Clustering

