#pragma once

#include <vector>
#include <map>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace Clustering {
  namespace MPP {
    using SparseMatrixF = boost::numeric::ublas::mapped_matrix<float>;

    SparseMatrixF
    row_normalized_transition_probabilities(SparseMatrixF count_matrix);

  } // end namespace Clustering::MPP

  std::vector<std::size_t>
  most_probable_path(std::vector<std::size_t> initial_clusters,
                     float q_min,
                     std::size_t lagtime);

} // end namespace Clustering

