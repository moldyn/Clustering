
#include "mpp.hpp"

namespace Clustering {
  namespace MPP {
    SparseMatrixF
    get_transition_counts(std::vector<std::size_t> clusters,
                          std::size_t n_lag_steps) {
      //TODO: allocate enough space for 'cm'
      SparseMatrixF cm;
      for (std::size_t i=0; i < clusters.size() - n_lag_steps; ++i) {
        std::size_t from = clusters[i];
        std::size_t to = clusters[i+n_lag_steps];
        cm(from, to) += 1;
      }
      return cm;
    }

    SparseMatrixF
    row_normalized_transition_probabilities(SparseMatrixF count_matrix) {
      SparseMatrixF transition_matrix;
      //TODO

      return transition_matrix;
    }
  } // end namespace MPP

  std::vector<std::size_t>
  most_probable_path(std::vector<std::size_t> initial_clusters,
                     float q_min,
                     std::size_t lagtime) {
    //TODO
  }
  
} // end namespace Clustering

