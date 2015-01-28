
#include "coring.hpp"

#include "logger.hpp"
#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <boost/program_options.hpp>
#include <omp.h>

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

int main(int argc, char* argv[]) {
  using namespace Clustering::Tools;
  namespace b_po = boost::program_options;
  b_po::variables_map args;
  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "compute boundary corrections for clustering results."
    "\n"
    "options"));
  desc.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
        "show this help.")
    // optional
    ("states,s", b_po::value<std::string>()->required(),
        "(required): file with state information (i.e. clustered trajectory")
    ("windows,w", b_po::value<std::string>()->required(), 
        "(required): file with window sizes."
        "format is space-separated lines of\n\n"
        "STATE_ID WINDOW_SIZE\n\n"
        "use * as STATE_ID to match all (other) states.\n"
        "e.g.:\n\n"
        "* 20\n"
        "3 40\n"
        "4 60\n\n"
        "matches 40 frames to state 3, 60 frames to state 4 and 20 frames to all the other states")
    ("output,o", b_po::value<std::string>(),
        "(optional): cored trajectory")
    ("distribution,d", b_po::value<std::string>(),
        "(optional): write waiting time distributions to file.")
    ("cores,c", b_po::value<std::string>(),
        "(optional): write core information to file, i.e. trajectory with state name if in core region or -1 if not in core region")
    ("concat-nframes", b_po::value<std::size_t>(),
      "input (optional parameter): no. of frames per (equally sized) sub-trajectory for concatenated trajectory files.")
    ("concat-limits", b_po::value<std::string>(),
      "input (optional, file): file with frame ids (base 0) of first frames per (not equally sized) sub-trajectory for concatenated trajectory files.")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false),
        "verbose mode: print runtime information to STDOUT.")
  ;
  // parse cmd arguments
  try {
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).run(), args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\n" << e.what() << "\n\n" << std::endl;
    }
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }
  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }
  // setup general flags / options
  Clustering::verbose = args["verbose"].as<bool>();
  // load states
  std::vector<std::size_t> states = Clustering::Tools::read_clustered_trajectory(args["states"].as<std::string>());
  std::set<std::size_t> state_names(states.begin(), states.end());
  std::size_t n_frames = states.size();
  if (args.count("output") || args.count("distribution") || args.count("cores")) {
    // load concatenation limits to treat concatenated trajectories correctly
    // when performing dynamical corrections
//    std::vector<std::size_t> concat_limits;
//    if (args.count("concat-limits")) {
//      concat_limits = Clustering::Tools::read_single_column<std::size_t>(args["concat-limits"].as<std::string>());
//    } else if (args.count("concat-nframes")) {
//      std::size_t n_frames_per_subtraj = args["concat-nframes"].as<std::size_t>();
//      for (std::size_t i=n_frames_per_subtraj; i < traj.size(); i += n_frames_per_subtraj) {
//        concat_limits.push_back(i);
//      }
//    }

//TODO use concat limits


    // load window size information
    std::map<std::size_t, std::size_t> coring_windows;
    {
      std::ifstream ifs(args["windows"].as<std::string>());
      std::string buf1, buf2;
      std::size_t size_for_all = 1;
      while (ifs.good()) {
        ifs >> buf1;
        ifs >> buf2;
        if (ifs.good()) {
          if (buf1 == "*") {
            size_for_all = string_to_num<std::size_t>(buf2);
          } else {
            coring_windows[string_to_num<std::size_t>(buf1)] = string_to_num<std::size_t>(buf2);
          }
        }
      }
      // fill remaining, not explicitly defined states with common window size
      for (std::size_t name: state_names) {
        if ( ! coring_windows.count(name)){
          coring_windows[name] = size_for_all;
        }
      }
    }
    // core trajectory
    std::vector<std::size_t> cored_traj(n_frames);
    std::size_t current_core = states[0];
    std::vector<long> cores(n_frames);
    for (std::size_t i=0; i < states.size(); ++i) {
      std::size_t w = coring_windows[states[i]];
      bool is_in_core = true;
      for (std::size_t j=i+1; j < i+w; ++j) {
        if (states[j] != states[i]) {
          is_in_core = false;
          break;
        }
      }
      if (is_in_core) {
        current_core = states[i];
        cores[i] = current_core;
      } else {
        cores[i] = -1;
      }
      cored_traj[i] = current_core;
    }
    // write cored trajectory to file
    if (args.count("output")) {
      Clustering::Tools::write_clustered_trajectory(args["output"].as<std::string>(), cored_traj);
    }
    // write core information to file
    if (args.count("cores")) {
      Clustering::Tools::write_single_column<long>(args["cores"].as<std::string>(), cores, false);
    }
    // compute/save escape time distributions
    if (args.count("distribution")) {
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

      std::map<std::size_t, WTDMap> etds;
      for (std::size_t state: state_names) {
        etds[state] = compute_wtd(streaks[state]);
      }
      // write WTDs to file
      for (auto state_etd: etds) {
        std::string fname = Clustering::Tools::stringprintf(args["distribution"].as<std::string>() + "_%d", state_etd.first);
        Clustering::Tools::write_map<std::size_t, float>(fname, state_etd.second);
      }
    }
  } else {
    std::cerr << "\n" << "nothing to do! please define '--output', '--distribution' or both!" << "\n\n";
    std::cerr << desc << std::endl;
  }
  return EXIT_SUCCESS;
}

