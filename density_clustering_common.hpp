#pragma once

#include "config.hpp"
#include "density_clustering.hpp"

/*
 * this module holds common implementations for both, single-node and MPI code.
 * it is used for functions that are too similar for both versions and have
 * only slight variations that can be implemented by 'ifdef'-guards.
 */

namespace Clustering {
  namespace Density {

    //TODO doc
    std::vector<std::size_t>
    initial_density_clustering(const std::vector<float>& free_energy
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
                               );

  } // end namespace Density
} // end namespace Clustering

