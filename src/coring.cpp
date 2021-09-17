/*
Copyright (c) 2015-2017, Florian Sittel (www.lettis.net)
Copyright (c) 2018-2021, Daniel Nagel
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

#include "coring.hpp"

#include "logger.hpp"
#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <numeric>

#include <omp.h>

namespace Clustering {
namespace Coring {
  WTDMap
  compute_wtd(std::list<std::size_t> streaks) {
    WTDMap wtd;
    if (streaks.size() > 0) {
      streaks.sort(std::greater<std::size_t>());
      std::size_t max_streak = streaks.front();
      for (std::size_t i=0; i <= max_streak; ++i) {
        float n_steps = 0.0f;
        for (auto s: streaks) {
          if (i > s) {
            break;
          }
          n_steps += 1.0f;
        }
        wtd[i] = n_steps / ((float) streaks.size());
      }
    }
    return wtd;
  }

  void
  main(boost::program_options::variables_map args) {
    using namespace Clustering::Tools;
    namespace b_po = boost::program_options;
    // load states
    Clustering::logger(std::cout) << "~~~ reading files\n"
                                  << "    trajectory from: " << args["states"].as<std::string>()
                                  << std::endl;
    std::vector<std::size_t> states = Clustering::Tools::read_clustered_trajectory(args["states"].as<std::string>());
    std::set<std::size_t> state_names(states.begin(), states.end());
    std::size_t n_frames = states.size();
    bool iterative_coring = args["iterative"].as<bool>();
    std::string header_comment = args["header"].as<std::string>();
    std::map<std::string,float> commentsMap = args["commentsMap"].as<std::map<std::string,float>>();
    // read previously used parameters
    read_comments(args["states"].as<std::string>(), commentsMap);
    if (args.count("output") || args.count("distribution") || args.count("cores")) {
      // load concatenation limits to treat concatenated trajectories correctly
      // when performing dynamical corrections
      std::vector<std::size_t> concat_limits;
      if (args.count("concat-limits")) {
        Clustering::logger(std::cout) << "    limits from: "
                                      << args["concat-limits"].as<std::string>() << std::endl;
        concat_limits = Clustering::Tools::read_concat_limits(args["concat-limits"].as<std::string>());
      } else if (args.count("concat-nframes")) {
        std::size_t n_frames_per_subtraj = args["concat-nframes"].as<std::size_t>();
        for (std::size_t i=n_frames_per_subtraj; i <= n_frames; i += n_frames_per_subtraj) {
          concat_limits.push_back(i);
        }
      } else {
        concat_limits = {n_frames};
      }
      // check if concat_limits are well definied
      Clustering::Tools::check_concat_limits(concat_limits, n_frames);
      Clustering::logger(std::cout) << "    interpret data as " << concat_limits.size()
                                    << " trajectories" << std::endl;
      if (commentsMap["limits"] == 0) {
        commentsMap["limits"] = concat_limits.size();
      } else if (std::abs(commentsMap["limits"]-concat_limits.size()) > 0.001) {
        Clustering::logger(std::cout) << "warning: the number of limits are not in agreement\n"
                                      << "         " << commentsMap["limits"] << " vs. "
                                      << concat_limits.size() << std::endl;
      }
      // load window size information
      std::map<std::size_t, std::size_t> coring_windows;
      std::size_t size_for_all = 1;
      try{  // read single integer as coring window
        size_for_all = std::stoi(args["windows"].as<std::string>());
      } catch (...) {  // read window file
        Clustering::logger(std::cout) << "\n~~~ coring windows:\n    from file: "
                                      << args["windows"].as<std::string>() << std::endl;
        std::ifstream ifs(args["windows"].as<std::string>());
        std::string buf;
        std::size_t state, window;
        size_for_all = 1;
        if (ifs.fail()) {
          std::cerr << "error: cannot open file '"
                    << args["windows"].as<std::string>()
                    << "'" << std::endl;
          exit(EXIT_FAILURE);
        } else {
          while (ifs.good()) {
            if (std::isdigit(ifs.peek())) {
              ifs >> state;
              ifs >> window;
              if ( ! ifs.fail()) {
                coring_windows[state] = window;
              } else {
                std::cerr << "error: file not correctly formated." << std::endl;
              }
            } else if (ifs.peek() == '*') {
              ifs >> buf;
              ifs >> window;
              if ( ! ifs.fail()) {
                size_for_all = window;
              } else {
                std::cerr << "error: file not correctly formated." << std::endl;
              }
            } else {
              ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
          }
        }
      }
      // fill remaining, not explicitly defined states with common window size
      std::size_t undefined_windows = 0;
      for (std::size_t name: state_names) {
        if ( ! coring_windows.count(name)){
          coring_windows[name] = size_for_all;
          ++undefined_windows;
        }
      }
      // if only single coring window is used
      if (coring_windows.size() == 0) {
        commentsMap["single_coring_time"] = size_for_all;
      }
      header_comment.append(Clustering::Tools::stringprintf(
        "#\n# coring specific parameters: \n"
        "#    %i state-specific coring windows were read\n"
        "#    %i frames is used for reamining states\n",
      state_names.size() - undefined_windows, size_for_all));
      if (iterative_coring) {
          header_comment.append("# iterative mode active\n");
      }
      if ((state_names.size() - undefined_windows) > 0) {
        Clustering::logger(std::cout) << "    " << state_names.size() - undefined_windows
                                      << " state-specific coring windows were read" << std::endl;
      }
      if (size_for_all > 1) {
        Clustering::logger(std::cout) << "    default window was set to " << size_for_all
                                      << " frames" << std::endl;
      }

      // get smallest coring window to prevent window of size 0
      auto min_window_iter = std::min_element(
          coring_windows.begin(), coring_windows.end(),
          [](const std::pair<std::size_t, std::size_t>& p1, const std::pair<std::size_t, std::size_t>& p2) {
              return p1.second < p2.second;
          }
      );
      std::size_t min_window = min_window_iter->second;
      if (min_window == 0) {
          std::cerr << "error: no window of size 0 is allowed. A window of length 1"
                    << " corresponds to no coring" << std::endl;
          exit(EXIT_FAILURE);
      }

      // core trajectory
      Clustering::logger(std::cout) << "\n~~~ coring trajectory" << std::endl;
      std::vector<std::size_t> cored_traj(n_frames);
      // copy states trajectory
      std::vector<std::size_t> cored_traj_prev(n_frames);
      std::copy(states.begin(), states.end(), cored_traj_prev.begin());
      std::vector<long> cores(n_frames);
      std::size_t changed_frames = 0;

      // get greatest coring window
      auto max_window_iter = std::max_element(
          coring_windows.begin(), coring_windows.end(),
          [](const std::pair<std::size_t, std::size_t>& p1, const std::pair<std::size_t, std::size_t>& p2) {
              return p1.second < p2.second;
          }
      );
      std::size_t max_window = max_window_iter->second;
      Clustering::logger(std::cout) << "    max coring window: " << max_window << std::endl;

      // generate range with iterative slices or single value
      std::vector<std::size_t> max_coring_window;
      if (iterative_coring && max_window > 1) {
          max_coring_window.resize(max_window-1);
          std::iota(max_coring_window.begin(), max_coring_window.end(), 2);
      } else {
          max_coring_window.push_back(max_window);
      }

      for (std::size_t curr_max_window: max_coring_window) {
          std::size_t current_core = cored_traj_prev[0];
          bool is_in_core = true;
          // honour concatenation limits, i.e. treat every concatenated trajectory-part on its own
          std::size_t last_limit = 0;
          for (std::size_t next_limit: concat_limits) {
            // this should never have an impact for correct limits
            std::size_t next_limit_corrected = std::min(next_limit, n_frames);
            // find first core
            current_core = cored_traj_prev[last_limit];  // fallback option
            for (std::size_t i=last_limit; i < next_limit_corrected; ++i) {
              std::size_t cw = std::min(curr_max_window, coring_windows[cored_traj_prev[i]]);
              std::size_t w = std::min(i+cw, next_limit);
              for (std::size_t j=i+1; j < w; ++j) {
                if (cored_traj_prev[j] != cored_traj_prev[i]) {
                  goto try_next_state;
                }
              }
              current_core = cored_traj_prev[i];
              break;

              try_next_state: ;
            }
            // core trajectories
            for (std::size_t i=last_limit; i < next_limit_corrected; ++i) {
              // coring window
              std::size_t cw = std::min(curr_max_window, coring_windows[cored_traj_prev[i]]);
              if (i+cw <= next_limit) {
                // std::size_t w = std::min(i+cw, next_limit_corrected);
                is_in_core = true;
                std::size_t j;
                // for iterative coring only last frame needs to be checked
                if (iterative_coring) {
                    j=i+cw-1;
                } else {
                    j=i+1;
                }
                for (; j < i+cw; ++j) {
                  if (cored_traj_prev[j] != cored_traj_prev[i]) {
                    is_in_core = false;
                    break;
                    }
                }
              } else {  // <-- so last frames can not be in core? should be true
                is_in_core = false;
              }

              if (is_in_core) {
                current_core = cored_traj_prev[i];
              }

              // generate statistic only for last iteration
              if (curr_max_window == max_window) {
                if (is_in_core) {
                  cores[i] = current_core;
                } else {
                  cores[i] = -1;
                }
                if (current_core != states[i]) {
                  ++changed_frames;
                }
              }
              cored_traj[i] = current_core;
            }
            last_limit = next_limit_corrected;
          }
        std::copy(cored_traj.begin(), cored_traj.end(), cored_traj_prev.begin());
      }
      float changed_frames_perc = (float) 100*changed_frames / n_frames;
      Clustering::logger(std::cout) << Clustering::Tools::stringprintf("    %.2f", changed_frames_perc)
                                    << "% of frames were changed\n    " << changed_frames
                                    << " frames in total"
                                    << std::endl;
      // write cored trajectory to file
      std::string header_coring = header_comment
                  + Clustering::Tools::stringprintf("#    %.2f", changed_frames_perc)
                  + "% of frames were changed\n";
      if (args.count("output")) {
        Clustering::logger(std::cout) << "    store result in: " << args["output"].as<std::string>()
                                      << std::endl;
        Clustering::Tools::write_clustered_trajectory(args["output"].as<std::string>(),
                                                      cored_traj,
                                                      header_coring,
                                                      commentsMap);
      }
      // write core information to file
      if (args.count("cores")) {
        append_commentsMap(header_coring, commentsMap);
        Clustering::Tools::write_single_column<long>(args["cores"].as<std::string>(),
                                                     cores,
                                                     header_coring,
                                                     false);
      }
      // compute/save escape time distributions
      if (args.count("distribution")) {
        Clustering::logger(std::cout) << "~~~ generating distribution" << std::endl;
        std::map<std::size_t, std::list<std::size_t>> streaks;
        std::size_t current_state = cored_traj[0];
        long n_counts = 0;
        for (std::size_t state: cored_traj) {
          if (state == current_state) {
            ++n_counts;
          } else {
            streaks[current_state].push_back(n_counts);
            current_state = state;
            n_counts = 1;
          }
        }
        streaks[current_state].push_back(n_counts);

        // append values to header comment
        append_commentsMap(header_comment, commentsMap);

        std::map<std::size_t, WTDMap> etds;
        for (std::size_t state: state_names) {
          etds[state] = compute_wtd(streaks[state]);
        }
        // write WTDs to file
        Clustering::logger(std::cout) << "    storing..." << std::endl;
        for (auto state_etd: etds) {
          std::string fname = Clustering::Tools::stringprintf(args["distribution"].as<std::string>() + "_%d", state_etd.first);
          Clustering::Tools::write_map<std::size_t, float>(fname, state_etd.second, header_comment);
        }
      }
    } else {
      std::cerr << "\n" << "error (coring): nothing to do! please define '--output', '--distribution' or both!" << "\n\n";
      exit(EXIT_FAILURE);
    }
  }
} // end namespace Coring
} // end namespace Clustering
