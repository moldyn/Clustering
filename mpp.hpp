#pragma once

#include <vector>
#include <map>
#include <set>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace Clustering {
  namespace MPP {
    using SparseMatrixF = boost::numeric::ublas::mapped_matrix<float>;

    // count transitions from one to the other cluster with certain lag
    // and return as count matrix (row/col := from/to)
    SparseMatrixF
    transition_counts(std::vector<std::size_t> trajectory, std::size_t n_lag_steps);

    // compute transition matrix from counts by normalization of rows
    SparseMatrixF
    row_normalized_transition_probabilities(SparseMatrixF count_matrix, std::set<std::size_t> microstate_names);

    // compute immediate future (i.e. without lag) of every state from highest probable transitions
    std::map<std::size_t, std::size_t>
    single_step_future_state(SparseMatrixF transition_matrix, std::set<std::size_t> cluster_names, float q_min);

    // for every state, compute most probable path by following
    // the 'future_state'-mapping recursively
    std::map<std::size_t, std::vector<std::size_t>>
    most_probable_path(std::map<std::size_t, std::size_t> future_state, std::set<std::size_t> cluster_names);

    // compute cluster populations
    std::map<std::size_t, std::size_t>
    microstate_populations(std::vector<std::size_t> clusters, std::set<std::size_t> cluster_names);
  } // end namespace Clustering::MPP
} // end namespace Clustering

