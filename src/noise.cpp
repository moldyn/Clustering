/*
Copyright (c) 2015-2019, Florian Sittel (www.lettis.net) and Daniel Nagel
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

#include "noise.hpp"

#include "logger.hpp"
#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <boost/filesystem.hpp>
#include <omp.h>

namespace Clustering {
namespace Noise {
  void
  main(boost::program_options::variables_map args) {
    using namespace Clustering::Tools;
    namespace b_fs = boost::filesystem;
    namespace b_po = boost::program_options;
    // load states
    Clustering::logger(std::cout) << "~~~ reading files\n"
                                  << "    trajectory from: " << args["states"].as<std::string>()
                                  << std::endl;
    std::vector<std::size_t> states = Clustering::Tools::read_clustered_trajectory(args["states"].as<std::string>());
    std::vector<std::size_t> states_without_noise;
    std::copy(states.begin(), states.end(), std::back_inserter(states_without_noise));
    std::set<std::size_t> state_names(states.begin(), states.end());
    std::size_t n_frames = states.size();
    // shift cmin fomr [0,100] -> [0,1]
    float cmin = 0.01*args["cmin"].as<float>();
    std::string basename = args["basename"].as<std::string>();
    basename.append(".");
    Clustering::verbose = args["verbose"].as<bool>();
    // generate header comment
    std::string header_comment = args["header"].as<std::string>();
    std::map<std::string,float> commentsMap = args["commentsMap"].as<std::map<std::string,float>>();
    // read previously used parameters
    read_comments(args["states"].as<std::string>(), commentsMap);
    commentsMap["cmin"] = cmin;
    // noise state is 1 lower than lowest
    auto lowestState = std::min_element(states.begin(), states.end());
    std::size_t noiseState = *lowestState-1;
    
    if (args.count("output") || args.count("cores")) {
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
      // findest highest clust file
      b_fs::path cwd(b_fs::current_path());
      b_fs::path wd_basename(basename);

      // if relative or absolute path is given, use this instead of cwd
      if (wd_basename.parent_path() != "") {
        cwd = wd_basename.parent_path();
      }

      typedef std::vector<b_fs::path> vec;             // store paths,
      vec v;                                           // so we can sort them later

      std::copy(b_fs::directory_iterator(cwd), b_fs::directory_iterator(), std::back_inserter(v));

      std::sort(v.begin(), v.end());             // sort, since directory iteration
                                                 // is not ordered on some file systems
//      std::cout << " basename: " << basename << "\n";
//      std::cout << " v.end(): " << v.back().string() << "\n";      
      // iterate reverse over all files in directory
      std::string clust_filename;
      std::size_t found = std::string::npos;
      for (vec::reverse_iterator it(v.rbegin()), it_end(v.rend()); it != it_end; ++it)
      {
        std::ostringstream oss;
        oss << *it;
        std::string file_str = oss.str();

        found = file_str.rfind(basename);
        if (found!=std::string::npos) {
          // if commentsMap is provided, check if found fe > fe_max
          if (commentsMap.count("screening_to") && commentsMap.count("screening_step")) {
            float fe_max;
            try{
              fe_max = std::stof(file_str.substr(found+basename.length(),file_str.length()-found-basename.length()-1));
              if ((fe_max > commentsMap["screening_to"] + commentsMap["screening_step"]) ||
                 (fe_max < commentsMap["screening_to"])){
                continue;
              }
            } catch (...) {
                continue;
            }
          }
          clust_filename = file_str.substr(found,file_str.length()-found-1);
          break;
        }
      }
      // catch if file not found
      if (found == std::string::npos) {
        std::cerr << "\n" << "error (noise): cluster file of type " << basename << " not found\n\n";
        exit(EXIT_FAILURE);
      }
      
      // open highest clust file
      header_comment.append(Clustering::Tools::stringprintf("#\n# Execution remarks:\n"
                            "# used for highest cluster file: %s\n", clust_filename.c_str()));
      Clustering::logger(std::cout) << "    highest cluster: " << clust_filename  << std::endl;
      std::vector<std::size_t> clust = Clustering::Tools::read_clustered_trajectory(clust_filename);
      // check if parameters are identical to states ones
      read_comments(clust_filename, commentsMap);
      if (n_frames != clust.size()) {
        std::cerr << "\n" << "error (noise): clust file is not of same length as state trajectory." << "\n\n";
        exit(EXIT_FAILURE);
      }
      // generate counts of each cluster id
      CounterClustMap counts;
      for (std::size_t i=0; i < n_frames; ++i) {
        CounterClustMap::iterator it(counts.find(clust[i]));
        if (it != counts.end()){
          it->second++;
        } else {
          counts[clust[i]] = 1;
        }
      }
      Clustering::logger(std::cout) << "~~~ noise assignment" << std::endl;
      std::size_t noise_frames = 0;
      // define noise frames as state 0
      for (std::size_t i=0; i < n_frames; ++i) {
        if (counts[clust[i]] < cmin*n_frames){
          states[i] = noiseState;
          ++noise_frames;
        }      
      }
      float noise_frames_per = 100.*noise_frames / n_frames;
      Clustering::logger(std::cout) << Clustering::Tools::stringprintf("    %.2f", noise_frames_per)
                                    << "% of frames were identified as noise" << std::endl;
      header_comment.append(Clustering::Tools::stringprintf("# %.2f", noise_frames_per) +
                            "% of frames were identified as noise\n");
      // TODO: remove following line. Should we keep with argument?
      // Clustering::Tools::write_clustered_trajectory("microstatesNoiseDef", states);
      // noise core trajectory
      std::vector<std::size_t> noise_traj(n_frames);
      std::size_t current_core = states[0];
      std::vector<long> cores(n_frames);
      std::size_t changed_frames = 0;
      // honour concatenation limits, i.e. treat every concatenated trajectory-part on its own
      std::size_t last_limit = 0;
      for (std::size_t next_limit: concat_limits) {
        std::size_t next_limit_corrected = std::min(next_limit, n_frames);
        for (std::size_t i=last_limit; i < next_limit_corrected; ++i) {
          if (states[i] != noiseState) {
            current_core = states[i];
            break;      
          }
        }
        for (std::size_t i=last_limit; i < next_limit_corrected; ++i) {
          // coring window
          if (states[i] != noiseState) {
            current_core = states[i];
            cores[i] = current_core;
          } else {
            cores[i] = -1;
          }
          if (current_core != states_without_noise[i]) {
            ++changed_frames;
          }
          noise_traj[i] = current_core;
        }
        last_limit = next_limit_corrected;
      }
      float changed_frames_per = 100.*changed_frames / n_frames;
      Clustering::logger(std::cout) << Clustering::Tools::stringprintf("    %.2f", changed_frames_per)
                                    << "% of frames were reassigned\n"
                                    << "    store result in: " << args["output"].as<std::string>()
                                    << std::endl;
      header_comment.append(Clustering::Tools::stringprintf("# %.2f", changed_frames_per) +
                            "% of frames were reassigned\n");
      // write cored trajectory to file
      if (args.count("output")) {
        Clustering::Tools::write_clustered_trajectory(args["output"].as<std::string>(),
                                                      noise_traj,
                                                      header_comment,
                                                      commentsMap);
      }
      // write core information to file
      if (args.count("cores")) {
        append_commentsMap(header_comment, commentsMap);
        Clustering::Tools::write_single_column<long>(args["cores"].as<std::string>(),
                                                     cores,
                                                     header_comment,
                                                     false);
      }
    } else {
      std::cerr << "\n" << "error (noise): nothing to do! please define '--output' or '--cores'" << "\n\n";
      exit(EXIT_FAILURE);
    }
  }
} // end namespace Noise
} // end namespace Clustering

