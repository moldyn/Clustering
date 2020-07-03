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

#include "coords_file/coords_file.hpp"

#include <boost/program_options.hpp>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <queue>
#include <map>
#include <vector>

#include "state_filter.hpp"
#include "tools.hpp"
#include "logger.hpp"


namespace Clustering {
namespace Filter {
  void
  fixedprint(double num, std::size_t prec, std::size_t width){
    std::cout.width(width);
    std::cout << std::setprecision(prec);
    std::cout << num;
    std::cout.width(0);
    return;
  }

  void
  main(boost::program_options::variables_map args) {
    using namespace Clustering::Tools;
    // load states
    Clustering::logger(std::cout) << "~~~ reading files\n"
                                  << "    trajectory from: "
                                  << args["states"].as<std::string>()
                                  << std::endl;
    std::string fname_states = args["states"].as<std::string>();
    std::vector<std::size_t> states = Clustering::Tools::read_clustered_trajectory(args["states"].as<std::string>());
    std::size_t n_frames = states.size();
    if (args["list"].as<bool>()) { // mode stats with verbose true
      std::map<std::string,float> commentsMap = args["commentsMap"].as<std::map<std::string,float>>();
      // read previously used parameters
      read_comments(args["states"].as<std::string>(), commentsMap);
      std::priority_queue<std::pair<std::size_t, std::size_t>> pops;
      // list states with pops
      std::set<std::size_t> state_ids(states.begin(), states.end());
      for (std::size_t id: state_ids) {
        std::size_t pop = std::count(states.begin(), states.end(), id);
        pops.push({pop, id});
      }

      // load concat limits
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

      // get number of entering each state
      std::map<std::size_t, std::size_t> entered;
      std::map<std::size_t, std::size_t> left;
      //entered[states[0]] = 1;
//      std::size_t diff(n_frames);
      std::size_t last_limit = 0;
      for (std::size_t next_limit: concat_limits) {
        std::size_t next_limit_corrected = std::min(next_limit, n_frames);
        for (std::size_t i=last_limit; i < next_limit_corrected-1; ++i) {
          if (states[i+1] != states[i]){
            std::map<std::size_t, std::size_t>::iterator it(entered.find(states[i+1]));
            if (it != entered.end()){
              it->second++;
            } else {
              entered[states[i+1]] = 1;
            }
            std::map<std::size_t, std::size_t>::iterator it_l(left.find(states[i]));
            if (it_l != left.end()){
              it_l->second++;
            } else {
              left[states[i]] = 1;
            }

          }
        }
        last_limit = next_limit_corrected;
      }
      std::cout << "~~~ state stats\n"
                << "    state  population  pop [%]  tot [%]  entered     left" << std::endl;
      std::cout << std::fixed;
      double total_pop = 0.;
      std::size_t total_entered = 0;
      std::size_t total_left = 0;
      while ( ! pops.empty()) {
        auto pop_id = pops.top(); // get top element
        pops.pop(); // remove top element
        std::cout << "    ";
        // state id
        fixedprint(pop_id.second, 0, 5);
        // absolute pop
        fixedprint(pop_id.first, 0, 12);
        // relative pop
        fixedprint(100.*pop_id.first/(float)n_frames, 3, 9);
        // total pop
        total_pop += 100.*pop_id.first/(float)n_frames;
        fixedprint(total_pop, 3, 9);
        // entered
        std::map<std::size_t, std::size_t>::iterator it(entered.find(pop_id.second));
        if (it != entered.end()){
          total_entered += it->second;
          fixedprint(it->second, 0, 9);
        } else { // the following should never be the case
          fixedprint(0, 0, 9);
        }
        std::map<std::size_t, std::size_t>::iterator it_l(left.find(pop_id.second));
        if (it_l != left.end()){
          total_left += it_l->second;
          fixedprint(it_l->second, 0, 9);
        } else { // the following should never be the case
          fixedprint(0, 0, 9);
        }
        std::cout << std::endl;
      }
      std::cout << "\n~~~ total number of microstates: " << entered.size()
                << "\n                    transitions: " << total_entered
                << std::endl;
    } else {
      // filter data
      std::string coords_name = args["coords"].as<std::string>();
      Clustering::logger(std::cout) << "        coords from: "
                                    << coords_name
                                    << std::endl;
      // find selected (or all) states
      std::vector<std::size_t> selected_states;
      if (args.count("selected-states")) {
        selected_states = args["selected-states"].as<std::vector<std::size_t>>();
      } else { // use all states
        std::vector<std::size_t> unique_states(states);
        std::sort(unique_states.begin(), unique_states.end());
        auto last = std::unique(unique_states.begin(), unique_states.end());
        unique_states.erase(last, unique_states.end());
        selected_states = unique_states;
      }
      // get file extension of coords
      std::string file_extension = "";
      if (coords_name.compare(coords_name.size()-4, 1, ".") == 0){
        file_extension = coords_name.substr(coords_name.size()-4, 4);
      }
      std::string output_basename;
      if (args.count("output")) {
        output_basename = args["output"].as<std::string>();
      } else {
        if (file_extension.size()==0) {
          output_basename = coords_name;
        } else {
          output_basename = coords_name.substr(0,coords_name.size()-4);
        }
      }
      // filter data
      Clustering::logger(std::cout) << "\n~~~ filter states:" << std::endl;
      std::size_t every_nth = args["every-nth"].as<std::size_t>();
      if (every_nth > 1) {
        Clustering::logger(std::cout) << "    use only every " << every_nth
                                      << "th frame" << std::endl;
      }
      if (args.count("nRandom") and (every_nth > 1)) {
        std::cerr << "\nerror parsing arguments:\n\n"
                  << "Use either 'every-nth' or 'nRandom'"
                  << "\n\n" << std::endl;
        exit(EXIT_FAILURE);
      }
      std::size_t nRandom = 0;
      if (args.count("nRandom")) {
        nRandom = args["nRandom"].as<std::size_t>();
      }
      // initialize random number generator
      std::random_device rd;
      std::mt19937 g(rd());
      for (std::size_t selected_state: selected_states){
        std::vector<std::size_t> selected_state_idx;
        if (nRandom > 0) {
          // get all indices corresponding to selected state
          for(std::size_t idx=0; idx<states.size(); ++idx){
              if (states[idx] == selected_state) {
                selected_state_idx.push_back(idx);
              }
          }

          // catch if state has less frames than requested
          std::size_t nRandomState = std::min(nRandom, selected_state_idx.size());
          // shuffle indices
          std::shuffle(selected_state_idx.begin(), selected_state_idx.end(), g);
          // extract first nRandom one
          selected_state_idx = {selected_state_idx.begin(), selected_state_idx.begin() + nRandomState};
        }
         // open coords
        CoordsFile::FilePointer coords_in = CoordsFile::open(coords_name, "r");
        // get output name
        std::string basename = output_basename + ".state%i" + file_extension;
        std::string output_name = Clustering::Tools::stringprintf(basename,
                                                                  selected_state);
        CoordsFile::FilePointer coords_out = CoordsFile::open(output_name, "w");

        Clustering::logger(std::cout) << "    " << selected_state
                                      << " : "
                                      << output_name
                                      << std::endl;
        std::size_t nth = 0;  // counting frames
        for (std::size_t idx=0; idx< n_frames; ++idx) {
          if (states[idx] == selected_state) {
            if (nRandom > 0) {
              if (std::count(selected_state_idx.begin(), selected_state_idx.end(), idx)) {
                coords_out->write(coords_in->next());
              } else {
                coords_in->next();
              }
            } else {
              if ((nth % every_nth) == 0) {
                coords_out->write(coords_in->next());
              } else {
                coords_in->next();
              }
              ++nth;
            }
          } else {
            coords_in->next();
          }
        }
      }
    }
  }
} // end namespace Filter
} // end namespace Clustering
