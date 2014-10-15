
#include "density_clustering_mpi.hpp"
#include "density_clustering.hpp"

namespace DC_MPI {

  namespace { // local
    std::size_t n_nodes;
    std::size_t node_id;
  } // end local namespace
  
  std::vector<std::size_t>
  calculate_populations(const float* coords,
                        const std::size_t n_rows,
                        const std::size_t n_cols,
                        const float radius) {
    //TODO
  }

  void
  density_main(boost::program_options::variables_map args) {
    //TODO
  }

} // end namespace DC_MPI

