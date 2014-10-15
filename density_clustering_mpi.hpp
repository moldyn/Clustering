#pragma once

#include "config.hpp"

#include <vector>

#include <boost/program_options.hpp>

namespace Clustering {
namespace Density {
namespace MPI {

std::vector<std::size_t>
calculate_populations(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const float radius,
                      const int mpi_n_nodes,
                      const int mpi_node_id);

void
main(boost::program_options::variables_map args);

} // end namespace MPI
} // end namespace Density
} // end namespace Clustering

