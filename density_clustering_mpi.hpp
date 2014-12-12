#pragma once

#include "config.hpp"
#include "density_clustering.hpp"

#include <vector>

#include <boost/program_options.hpp>

namespace Clustering {
namespace Density {
namespace MPI {

const int MAIN_PROCESS = 0;

//TODO doc

std::vector<std::size_t>
calculate_populations(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const float radius,
                      const int mpi_n_nodes,
                      const int mpi_node_id);

std::map<float, std::vector<std::size_t>>
calculate_populations(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      std::vector<float> radii,
                      const int mpi_n_nodes,
                      const int mpi_node_id);

std::tuple<Neighborhood, Neighborhood>
nearest_neighbors(const float* coords,
                  const std::size_t n_rows,
                  const std::size_t n_cols,
                  const std::vector<float>& free_energy,
                  const int mpi_n_nodes,
                  const int mpi_node_id);

// returns neighborhood set of single frame.
// all ids are sorted in free energy.
std::set<std::size_t>
high_density_neighborhood(const float* coords,
                          const std::size_t n_cols,
                          const std::vector<FreeEnergy>& sorted_fe,
                          const std::size_t i_frame,
                          const std::size_t limit,
                          const float max_dist,
                          const int mpi_n_nodes,
                          const int mpi_node_id);

void
main(boost::program_options::variables_map args);

} // end namespace MPI
} // end namespace Density
} // end namespace Clustering

