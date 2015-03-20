
#include "coords_file/coords_file.hpp"

#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <queue>

#include "state_filter.hpp"

namespace {
  int io_error(std::string fname) {
    std::cerr << "error: cannot open file " << fname << "." << std::endl;
    return EXIT_FAILURE;
  }
} // end local namespace


namespace Clustering {
namespace Filter {
  void
  main(boost::program_options::variables_map args) {
    // load states
    std::string fname_states = args["states"].as<std::string>();
    std::vector<std::size_t> states;
    {
      std::ifstream ifs(fname_states);
      if (ifs.fail()) {
        exit(io_error(fname_states));
      } else {
        while (ifs.good()) {
          std::size_t buf;
          ifs >> buf;
          if (ifs.good()) {
            states.push_back(buf);
          }
        }
      }
    }
    if (args["list"].as<bool>()) {
      std::priority_queue<std::pair<std::size_t, std::size_t>> pops;
      // list states with pops
      std::set<std::size_t> state_ids(states.begin(), states.end());
      for (std::size_t id: state_ids) {
        std::size_t pop = std::count(states.begin(), states.end(), id);
        pops.push({pop, id});
      }
      while ( ! pops.empty()) {
        auto pop_id = pops.top(); // get top element
        pops.pop(); // remove top element
        std::cout << pop_id.second << " " << pop_id.first << "\n";
      }
    } else {
      // filter data
      std::size_t selected_state = args["state"].as<std::size_t>();
      CoordsFile::FilePointer coords_in = CoordsFile::open(args["coords"].as<std::string>(), "r");
      CoordsFile::FilePointer coords_out = CoordsFile::open(args["output"].as<std::string>(), "w");
      for (std::size_t s: states) {
        if (s == selected_state) {
          coords_out->write(coords_in->next());
        } else {
          coords_in->next();
        }
      }
    }
  }
} // end namespace Filter
} // end namespace Clustering

