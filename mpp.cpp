
#include "mpp.hpp"

namespace Clustering {
  namespace MPP {
    SparseMatrixF
    transition_counts(std::vector<std::size_t> trajectory,
                      std::size_t n_lag_steps) {
      std::size_t i_max = (*std::max_element(trajectory.begin(), trajectory.end()));
      SparseMatrixF count_matrix(i_max, i_max);
      for (std::size_t i=0; i < trajectory.size() - n_lag_steps; ++i) {
        std::size_t from = trajectory[i];
        std::size_t to = trajectory[i+n_lag_steps];
        count_matrix(from, to) += 1;
      }
      return count_matrix;
    }

    SparseMatrixF
    row_normalized_transition_probabilities(SparseMatrixF count_matrix,
                                            std::set<std::size_t> cluster_names) {
      std::size_t n_rows = count_matrix.size1();
      std::size_t n_cols = count_matrix.size2();
      SparseMatrixF transition_matrix(n_rows, n_cols);
      for (std::size_t i: cluster_names) {
        std::size_t row_sum = 0;
        for (std::size_t j=0; j < n_cols; ++j) {
          row_sum += count_matrix(i,j);
        }
        if (row_sum > 0) {
          for (std::size_t j=0; j < n_cols; ++j) {
            if (count_matrix(i,j) != 0) {
              transition_matrix(i,j) = count_matrix(i,j) / row_sum;
            }
          }
        }
      }
      return transition_matrix;
    }

    std::map<std::size_t, std::size_t>
    single_step_future_state(SparseMatrixF transition_matrix,
                             std::set<std::size_t> cluster_names,
                             float q_min,
                             std::map<std::size_t, float> min_free_energy) {
      std::map<std::size_t, std::size_t> future_state;
      for (std::size_t i: cluster_names) {
        std::vector<std::size_t> candidates;
        float max_trans_prob = 0.0f;
        if (transition_matrix(i,i) >= q_min) {
          // self-transition is greater than stability measure: stay.
          candidates = {i};
        } else {
          for (std::size_t j: cluster_names) {
            if (transition_matrix(i,j) > max_trans_prob) {
              max_trans_prob = transition_matrix(i,j);
              candidates = {i};
            } else if (transition_matrix(i,j) == max_trans_prob && max_trans_prob > 0.0f) {
              candidates.push_back(i);
            }
          }
        }
        if (candidates.size() == 1) {
          future_state[i] = candidates[0];
        } else {
          // multiple candidates: choose the one with lowest Free Energy
          auto min_fe_compare = [&](std::size_t i, std::size_t j) {
            return min_free_energy[i] < min_free_energy[j];
          };
          future_state[i] = (*std::min_element(candidates.begin(), candidates.end(), min_fe_compare));
        }
      }
      return future_state;
    }

    std::map<std::size_t, std::vector<std::size_t>>
    most_probable_path(std::map<std::size_t, std::size_t> future_state,
                       std::set<std::size_t> cluster_names) {
      std::map<std::size_t, std::vector<std::size_t>> mpp;
      for (std::size_t i: cluster_names) {
        std::vector<std::size_t> path = {i};
        std::set<std::size_t> visited = {i};
        std::size_t next_state = future_state[i];
        // abort when path 'closes' in a loop, i.e.
        // when a state has been revisited
        while (visited.count(next_state) == 0) {
          path.push_back(next_state);
          visited.insert(next_state);
          next_state = future_state[next_state];
        }
        mpp[i] = path;
      }
      return mpp;
    }

    std::map<std::size_t, std::size_t>
    microstate_populations(std::vector<std::size_t> clusters,
                           std::set<std::size_t> cluster_names) {
      std::map<std::size_t, std::size_t> pops;
      for (std::size_t i: cluster_names) {
        pops[i] = std::count(clusters.begin(), clusters.end(), i);
      }
      return pops;
    }

    std::map<std::size_t, std::size_t>
    path_sinks(std::vector<std::size_t> clusters,
               std::map<std::size_t, std::vector<std::size_t>> mpp,
               SparseMatrixF transition_matrix,
               std::set<std::size_t> cluster_names,
               float q_min,
               std::vector<float> free_energy) {
      std::map<std::size_t, std::size_t> pops = microstate_populations(clusters, cluster_names);
      std::map<std::size_t, float> min_free_energy = microstate_min_free_energy(clusters, free_energy);
      std::map<std::size_t, std::size_t> sinks;
      for (std::size_t i: cluster_names) {
        std::vector<std::size_t> metastable_states;
        for (std::size_t j: mpp[i]) {
          // check: are there stable states?
          if (transition_matrix(j,j) > q_min) {
            metastable_states.push_back(j);
          }
        }
        if (metastable_states.size() == 0) {
          // no stable state: use all in path as candidates
          metastable_states = mpp[i];
        }
        // helper function: compare states by their population
        auto pop_compare = [&](std::size_t i, std::size_t j) -> bool {
          return pops[i] < pops[j];
        };
        // find sink candidate state by population
        auto candidate = std::max_element(metastable_states.begin(), metastable_states.end(), pop_compare);
        std::size_t max_pop = pops[*candidate];
        std::set<std::size_t> sink_candidates = {*candidate};
        metastable_states.erase(candidate);
        // there may be several states with same (max.) population,
        // collect them all into one set
        while (sink_candidates.count(*candidate) == 1) {
          candidate = std::max_element(metastable_states.begin(), metastable_states.end(), pop_compare);
          metastable_states.erase(candidate);
          if (pops[*candidate] == max_pop) {
            sink_candidates.insert(*candidate);
          }
        }
        // helper function: compare states by their min. Free Energy
        auto min_fe_compare = [&](std::size_t i, std::size_t j) -> bool {
          return min_free_energy[i] < min_free_energy[j];
        };
        // select sink either as the one with highest population ...
        if (sink_candidates.size() == 1) {
          sinks[i] = (*sink_candidates.begin());
        } else {
          // or as the one with lowest Free Energy, if several microstates
          // have the same high population
          sinks[i] = (*std::min_element(sink_candidates.begin(), sink_candidates.end(), min_fe_compare));
        }
      }
      return sinks;
    }

    // basins: all microstates that fall into given sink
    std::map<std::size_t, std::vector<std::size_t>>
    basins(std::map<std::size_t, std::size_t> sinks) {
      std::map<std::size_t, std::vector<std::size_t>> basins;
      for (auto state_sink: sinks) {
        if (basins.count(state_sink.second) == 0) {
          basins[state_sink.second] = {state_sink.first};
        } else {
          basins[state_sink.second].push_back(state_sink.first);
        }
      }
      return basins;
    }

    // lump states based on path sinks and return new trajectory.
    // new microstates will have IDs of sinks.
    std::vector<std::size_t>
    lumped_trajectory(std::vector<std::size_t> trajectory,
                      std::map<std::size_t, std::size_t> sinks) {
      for (std::size_t& state: trajectory) {
        state = sinks[state];
      }
      return trajectory;
    }

    // run clustering for given Q_min value
    std::vector<std::size_t>
    metastable_clustering(std::vector<std::size_t> initial_trajectory,
                          float q_min,
                          std::size_t lagtime) {
      std::set<std::size_t> microstate_names(initial_trajectory.begin(), initial_trajectory.end());
      std::vector<std::size_t> traj = initial_trajectory;
      const uint MAX_ITER=100;
      for (uint iter=0; iter < MAX_ITER; ++iter) {
        // get transition probabilities
        SparseMatrixF trans_prob = row_normalized_transition_probabilities(
                                     transition_counts(traj, lagtime),
                                     microstate_names);
        // get immediate future
        std::map<std::size_t, std::size_t> future_state = single_step_future_state(trans_prob, microstate_names, q_min);
        // compute MPP
        std::map<std::size_t, std::vector<std::size_t>> mpp = most_probable_path(future_state, microstate_names);
        //TODO get sinks



        //TODO lump traj
        //TODO check convergence
      }
    }
  } // end namespace MPP
} // end namespace Clustering

