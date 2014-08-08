#pragma once

#include "project_data.h"

//// map transitions of states i -> [j1: 0.1, j2: 0.3, ...] by priority queue.
//// highest prob. of transition == first element in queue.
//class TransitionMatrix {
// public:
//  TransitionMatrix();
//  void add(std::size_t i, std::size_t j, double tprob);
//  TransitionProbability most_probable_transition(std::size_t i);
// private:
//  static constexpr auto comp = [](TransitionProbability a, TransitionProbability b) -> bool
//                                 { return (a.second > b.second); };
//  std::map<std::size_t,
//           std::priority_queue<TransitionProbability,
//                               std::vector<TransitionProbability>,
//                               decltype(TransitionMatrix::comp)>> trans_matrix;
//};

void run_mpp(ProjectData& proj,
             std::size_t n_microstates,
             std::size_t lagtime,
             std::vector<std::size_t> break_points,
             std::size_t n_threads);

