
#include "mpp.h"

#include <cassert>
#include <limits>

namespace {


double geometric_distance_squared(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size());
  double d=0.0;
  for (std::size_t i=0; i < a.size(); ++i) {
    d += (a[i]-b[i]) * (a[i]-b[i]);
  }
  return d;
}

ClusterCenters convert_to_vectors(const std::vector<double>& centers, std::size_t n_cols) {
  ClusterCenters cl_c;
  std::size_t n_rows = centers.size() / n_cols;

  for (std::size_t r=0; r < n_rows; ++r) {
    std::vector<double> v;
    for (std::size_t c=0; c < n_cols; ++c) {
      v.push_back(centers[r*n_cols+c]);
    }
    cl_c[r] = v;
  }

  return cl_c;
}


std::map<std::size_t, std::size_t> state_populations(std::vector<size_t> trajectory) {
  std::map<std::size_t, std::size_t> p;
  for (std::size_t i=0; i < trajectory.size(); ++i) {
    ++p[i];
  }
  return p;
}

Transitions transitions(const std::vector<size_t>& traj,
                        const std::size_t lagtime,
                        std::vector<std::size_t>& break_points) {
  std::map<std::size_t, std::size_t> total_transitions;
  std::map<std::size_t, double> most_probable_transition_value;
  TransitionMatrix t_mat;
  auto next_break = break_points.begin();
  //TODO: break_points as TO-values in outer FOR loop?
  if (traj.size() > lagtime) {
    for (std::size_t i=0; i < traj.size()-lagtime; ++i) {
      // treat trajectory breaks (e.g. for concatenated trajectories)
      if (next_break != break_points.end()) {
        if (i < (*next_break)
        &&  (i+lagtime) >= (*next_break)) {
          continue;
        } else if (i == (*next_break)) {
          ++next_break;
          continue;
        } 
      }
      ++total_transitions[traj[i]];
      t_mat[Transition(traj[i], traj[i+lagtime])] += 1.0;
    }
  }
  std::set<std::size_t> states;
  for (auto it=t_mat.begin(); it != t_mat.end(); ++it) {
    std::size_t state_from = it->first.first;
    // collect all state ids
    states.insert(state_from);
    // normalization
    it->second /= total_transitions[state_from];
    // update most probable transition value for current state
    if (most_probable_transition_value.count(state_from) == 0
    ||  it->second > most_probable_transition_value[state_from]) {
      most_probable_transition_value[state_from] = it->second;
    }
  }
  return {t_mat, states, most_probable_transition_value};
}

FutureMap future_map(const Transitions& trans,
                     const ClusterCenters& centers,
                     double q_min) {
  FutureMap f;
  // helper function to get transition probability from transition-matrix.
  // if transition not found: probability = 0.
  auto prob = [&](Transition t) -> double {
    return (trans.transition_matrix.count(t) == 1)
            ? trans.transition_matrix.find(t)->second
            : 0.0;
  };
  for (auto state=trans.states.begin(); state != trans.states.end(); ++state) {
    Transition self_trans = Transition(*state, *state);
    if (prob(self_trans) >= q_min) {
      f[*state] = *state;
    }
  }
  // find transitions with highest probability for states not fulfilling q_min yet.
  auto highest_prob = [&](std::size_t state) -> double {
    return (trans.most_probable_transition_value.count(state) == 1)
            ? trans.most_probable_transition_value.find(state)->second
            : 0.0;
  };
  std::map<std::size_t, double> min_center_dist;
  // go through all available transitions ...
  for (auto t=trans.transition_matrix.begin(); t != trans.transition_matrix.end(); ++t) {
    std::size_t state_from = t->first.first;
    // ... and check, if transition is one with highest probability for a certain state
    if ((f.count(state_from) == 0)
    &&  (prob(t->first) == highest_prob(state_from))) {
      std::size_t state_to = t->first.second;
      double dist = geometric_distance_squared(centers.find(state_from)->second, centers.find(state_to)->second);
      // in case of several 'optimal' transitions (in means of probability)
      // we take the one with nearest cluster center
      if ((min_center_dist.count(state_from) == 0)
      ||  (dist < min_center_dist[state_from])) {
        f[state_from] = t->first.second;
        min_center_dist[state_from] = dist;
      }
    }
  }
  return f;
}

MostProbablePaths calculate_mpp(FutureMap f) {
  MostProbablePaths mpp;
  std::map<std::size_t, std::set<std::size_t>> visited_states;

  for (auto it=f.begin(); it != f.end(); ++it) {
    std::size_t state_now = it->first;
    if (mpp.count(state_now) == 0) {
      // initialize MPP of state with state itself
      mpp[state_now] = {state_now};
      // keep track of visited states and build their respective
      // MPP 'in parallel'. many of the states should merge their paths,
      // thus this should lead to many paths generated at the same time.
      if (visited_states.count(state_now) == 0) {
        visited_states[state_now] = {state_now};
      }
      while(visited_states.size() > 0) {
        for (auto it_v=visited_states.begin(); it_v != visited_states.end(); ++it_v) {
          std::size_t visited_key = it_v->first;
          std::set<std::size_t>& visited_set = it_v->second;
          if (visited_set.count(state_now) != 0) {
            // state has already been visited on this path;
            // MPP for this single visited state can be closed.
            visited_states.erase(state_now);
          } else {
            // update MPP/visiting list for every state already visited
            mpp[visited_key].push_back(state_now);
            visited_states[visited_key].insert(state_now);
          }
        }
        // next iteration: follow path of most probable transitions
        state_now = f[state_now];
      }
    }
  }
  return mpp;
}

bool mpps_equal(const std::vector<std::size_t>& mpp1, const std::vector<std::size_t>& mpp2) {
  // compare size
  if (mpp1.size() != mpp2.size()) {
    return false;
  }
  // compare element by element
  std::priority_queue<std::size_t> q1;
  std::priority_queue<std::size_t> q2;
  for (std::size_t i=0; i < mpp1.size(); ++i) {
    q1.push(mpp1[i]);
    q2.push(mpp2[i]);
  }
  while (q1.size() > 0) {
    if (q1.top() != q2.top()) {
      return false;
    }
    q1.pop();
    q2.pop();
  }
  // everything's equal ...
  return true;
}

FutureMap path_sinks(const MicroStates& ms, Transitions t, MostProbablePaths mpp, ClusterCenters centers) {
  FutureMap sinks;
  PopulationMap pop = state_populations(ms.trajectory);
  std::map<std::size_t, std::set<std::size_t>> common_mpps;
  // first check for common MPPs for different states
  for (auto s1=t.states.begin(); s1 != t.states.end(); ++s1) {
    for (auto s2=t.states.begin(); s2 != t.states.end(); ++s2) {
      if (*s1 != *s2
      &&  mpps_equal(mpp[*s1], mpp[*s2])) {
        common_mpps[*s1].insert(*s2);
        common_mpps[*s2].insert(*s1);
      }
    }
  }
  // now find sinks of MPPs. if some are common, save result for all states,
  // i.e. calculate the sink of the according MPP only once.
  for (auto state=t.states.begin(); state != t.states.end(); ++state) {
    // sink already exists (from equivalent MPP computed before)
    if (sinks.count(*state) != 0) {
      continue;
    } else {
      std::size_t max_pop = 0;
      double max_self_trans_rate = 0.0;
      double min_center_dist = std::numeric_limits<double>::max();
      std::size_t sink = *state;
      for (auto visited=mpp[*state].begin(); visited != mpp[*state].end(); ++visited) {
        if (pop[*visited] >= max_pop) {
          double self_trans_rate = t.transition_matrix[Transition(*visited, *visited)];
          double center_dist = geometric_distance_squared(centers[*state], centers[*visited]);
          // decide on sink via max. population or (if equal) higher self-transition probability
          if (pop[*visited] > max_pop
          ||  self_trans_rate > max_self_trans_rate
          ||  center_dist < min_center_dist) {
            sink = *visited;
            max_pop = pop[sink];
            max_self_trans_rate = self_trans_rate;
            min_center_dist = center_dist;
          }
        }
      }
      sinks[*state] = sink;
    }
  }
  return sinks;
}

} // end local namespace

void run_mpp(ProjectData& project,
             std::size_t n_microstates,
             std::size_t lagtime,
             std::vector<std::size_t> break_points,
             std::size_t n_threads) {

  MicroStates mstates = project.microstates[n_microstates];
  ClusterCenters cluster_centers = convert_to_vectors(mstates.centers, mstates.n_data_cols);


  const int MAX_ITERATIONS = 30;
  const float Q_STEP = 0.01f;

  for (float q_min=0.01f; q_min <= 1.00f; q_min += Q_STEP) {
    int iterations = 0;
    bool converged = false;
    while ( ! converged
       &&   (iterations < MAX_ITERATIONS)) {

      PopulationMap population = state_populations(mstates.trajectory);
      if (population.size() == 1) {
        converged = true;
      } else {
        Transitions trans = transitions(mstates.trajectory, lagtime, break_points);
        FutureMap immediate_future = future_map(trans, cluster_centers, q_min);

        // TODO  finish

        ++iterations;
      }
    }
    if (converged) {
      // TODO  finish
    } else {
      // TODO handle 'not converged within MAX_ITERATIONS'
    }
  }
  

}

