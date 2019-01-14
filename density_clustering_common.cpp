/*
Copyright (c) 2015-2019, Florian Sittel (www.lettis.net) and Daniel Nagel
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

#include "logger.hpp"
#include "density_clustering_common.hpp"
#include <iomanip>

#ifdef DC_USE_MPI
  #include "density_clustering_mpi.hpp"
#endif

namespace Clustering {
namespace Density {

  std::vector<std::size_t>
  screening(const std::vector<float>& free_energy
          , const Neighborhood& nh
          , const float free_energy_threshold
          , const float* coords
          , const std::size_t n_rows
          , const std::size_t n_cols
          , const std::vector<std::size_t> initial_clusters
#ifdef DC_USE_MPI
          , const int mpi_n_nodes
          , const int mpi_node_id
#endif
                            ) {
#ifdef DC_USE_MPI
    using namespace Clustering::Density::MPI;
#endif
    std::vector<std::size_t> clustering;
    std::size_t first_frame_above_threshold;
    double sigma2;
    std::vector<FreeEnergy> fe_sorted;
    std::set<std::size_t> visited_frames;
    std::size_t distinct_name;
    // data preparation
    std::tie(clustering
           , first_frame_above_threshold
           , sigma2
           , fe_sorted
           , visited_frames
           , distinct_name) = prepare_initial_clustering(free_energy
                                                       , nh
                                                       , free_energy_threshold
                                                       , n_rows
                                                       , initial_clusters);
    Clustering::logger(std::cout) << "    " << std::setw(6)
                                  << Clustering::Tools::stringprintf("%.3f", free_energy_threshold)
                                  << " " << std::setw(9)
                                  << Clustering::Tools::stringprintf("%i", first_frame_above_threshold)
                                  << std::endl;
#ifdef DC_USE_MPI
    if (mpi_node_id == MAIN_PROCESS) {
#endif
      screening_log(sigma2
                  , first_frame_above_threshold
                  , fe_sorted);
#ifdef DC_USE_MPI
    }
#endif
    // indices inside this loop are in order of sorted(!) free energies
    bool neighboring_clusters_merged = false;
    //TODO: this while loop shouldn't be necessary, resp.
    //      will always finish trivially after 2 runs, since nothing will
    //      happen as all frames will have been visitied...
    while ( ! neighboring_clusters_merged) {
      neighboring_clusters_merged = true;
#ifdef DC_USE_MPI
      if (mpi_node_id == MAIN_PROCESS) {
#endif
//        logger(std::cout) << "initial merge iteration" << std::endl;
#ifdef DC_USE_MPI
      }
#endif
      for (std::size_t i=0; i < first_frame_above_threshold; ++i) {
        if (visited_frames.count(i) == 0) {
          visited_frames.insert(i);
          // all frames/clusters in local neighborhood should be merged ...
#ifdef DC_USE_MPI
          using hdn_mpi = Clustering::Density::MPI::high_density_neighborhood;
          std::set<std::size_t> local_nh = hdn_mpi(coords,
                                                   n_cols,
                                                   fe_sorted,
                                                   i,
                                                   first_frame_above_threshold,
                                                   4*sigma2,
                                                   mpi_n_nodes,
                                                   mpi_node_id);
#else
          //TODO use box-assisted search on
          //     fe_sorted coords for 'high_density_neighborhood'
          std::set<std::size_t> local_nh = high_density_neighborhood(coords,
                                                                     n_cols,
                                                                     fe_sorted,
                                                                     i,
                                                                     first_frame_above_threshold,
                                                                     4*sigma2);
#endif
          neighboring_clusters_merged = lump_initial_clusters(local_nh
                                                            , distinct_name
                                                            , clustering
                                                            , fe_sorted
                                                            , first_frame_above_threshold)
                                     && neighboring_clusters_merged;
        }
      } // end for
    } // end while
    return normalized_cluster_names(first_frame_above_threshold
                                  , clustering
                                  , fe_sorted);
  }

} // end namespace Density
} // end namespace Clustering

