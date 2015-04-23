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

#include <vector>
#include <map>
#include <set>
#include <stdexcept>

#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "tools.hpp"

namespace Clustering {
  namespace MPP {
    using SparseMatrixF = boost::numeric::ublas::mapped_matrix<float>;
    using Neighborhood = Clustering::Tools::Neighborhood;

    // count transitions from one to the other cluster with certain lag
    // and return as count matrix (row/col := from/to)
    SparseMatrixF
    transition_counts(std::vector<std::size_t> trajectory, std::vector<std::size_t> concat_limits, std::size_t n_lag_steps);

    // compute transition matrix from counts by normalization of rows
    SparseMatrixF
    row_normalized_transition_probabilities(SparseMatrixF count_matrix, std::set<std::size_t> microstate_names);

    // compute immediate future (i.e. without lag) of every state from highest probable transitions;
    // exclude self-transitions.
    std::map<std::size_t, std::size_t>
    single_step_future_state(SparseMatrixF transition_matrix,
                             std::set<std::size_t> cluster_names,
                             float q_min,
                             std::map<std::size_t, float> min_free_energy);

    // for every state, compute most probable path by following
    // the 'future_state'-mapping recursively
    std::map<std::size_t, std::vector<std::size_t>>
    most_probable_path(std::map<std::size_t, std::size_t> future_state, std::set<std::size_t> cluster_names);

    // compute cluster populations
    std::map<std::size_t, std::size_t>
    microstate_populations(std::vector<std::size_t> clusters, std::set<std::size_t> cluster_names);

    // assign every state the lowest free energy value
    // of all of its frames.
    std::map<std::size_t, float>
    microstate_min_free_energy(const std::vector<std::size_t>& clustering,
                               const std::vector<float>& free_energy);

    //TODO doc
    std::map<std::size_t, std::size_t>
    path_sinks(std::vector<std::size_t> clusters,
               std::map<std::size_t, std::vector<std::size_t>> mpp,
               SparseMatrixF transition_matrix,
               std::set<std::size_t> cluster_names,
               float q_min,
               std::vector<float> free_energy,
               Neighborhood nh_high_dens);

    // lump states based on path sinks and return new trajectory.
    // new microstates will have IDs of sinks.
    std::vector<std::size_t>
    lumped_trajectory(std::vector<std::size_t> trajectory,
                      std::map<std::size_t, std::size_t> sinks);

    // run clustering for given Q_min value
    std::tuple<std::vector<std::size_t>, std::map<std::size_t, std::size_t>>
    fixed_metastability_clustering(std::vector<std::size_t> initial_trajectory,
                                   std::vector<std::size_t> concat_limits,
                                   float q_min,
                                   std::size_t lagtime,
                                   std::vector<float> free_energy,
                                   Neighborhood nearest_neighbors);

    /*
     * run MPP clustering
     *  args:
     *   input (string)
     *   lagtime (int)
     *   qmin-from (float)
     *   qmin-to (float)
     *   qmin-step (float)
     */
    void
    main(boost::program_options::variables_map args);
  } // end namespace Clustering::MPP
} // end namespace Clustering

