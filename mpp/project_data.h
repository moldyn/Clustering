#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <utility>

struct MicroStates {
  std::size_t n_microstates;
  std::vector<std::size_t> trajectory;
  // centers as matrix: n_microstates * n_cols; addressing: i*n_cols+j
  std::vector<double> centers;
  std::size_t n_data_cols;
  double cluster_variance;
};

struct LumpedStates {
  std::size_t n_states;
  std::vector<std::size_t> trajectory;
  std::map<std::size_t, std::size_t> population;
};

struct TransitionProbability {
  std::size_t from;
  std::size_t to;
  double prob;
  // define a < b operator for internal sorting in priority queue, etc.
  // this way, TransitionProbabilities can directly be compared to each other.
  friend bool operator<(TransitionProbability a, TransitionProbability b) {
    return a.prob < b.prob;
  }
};

typedef std::pair<std::size_t, std::size_t> Transition;
typedef std::map<Transition, double> TransitionMatrix;
typedef std::map<std::size_t, std::size_t> FutureMap;
typedef std::map<std::size_t, std::size_t> PopulationMap;
typedef std::map<std::size_t, std::vector<double>> ClusterCenters;
typedef std::map<std::size_t, std::vector<std::size_t>> MostProbablePaths;

struct Transitions {
  TransitionMatrix transition_matrix;
  std::set<std::size_t> states;
  std::map<std::size_t, double> most_probable_transition_value;
};


struct ProjectData {
  std::string coord_file;
  std::map<std::size_t, MicroStates> microstates;
};

// load project data from binary project file
ProjectData load_project(std::string fname);

// save project data to binary project file
void save_project(std::string fname, ProjectData proj);

