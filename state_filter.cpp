
#include "state_filter.hpp"
#include "tools.hpp"

#include <iostream>
#include <fstream>

namespace Clustering {
namespace Filter {

  void
  main(boost::program_options::variables_map args) {
    // filter data
    std::string fname_states = args["states"].as<std::string>();
    std::string fname_coords = args["phase-space"].as<std::string>();
    std::size_t selected_state = args["selected-state"].as<std::size_t>();
    std::ifstream ifs_states(fname_states);
    std::ofstream ofs;
    if (args.count("output")) {
      std::string fname_out = args["output"].as<std::string>();
      ofs.open(fname_out);
      if (ofs.fail()) {
        std::cerr << "error: cannot open file '" << fname_out << "' for writing." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    auto out = [&ofs]() -> std::ostream& {
      return ofs.is_open() ? ofs : std::cout;
    };
    if (ifs_states.fail()) {
      std::cerr << "error: cannot open file '" << fname_states << "' for reading." << std::endl;
      exit(EXIT_FAILURE);
    }
    std::ifstream ifs_coords(fname_coords);
    if (ifs_coords.fail()) {
      std::cerr << "error: cannot open file '" << fname_coords << "' for reading." << std::endl;
      exit(EXIT_FAILURE);
    }
    while (ifs_states.good() && ifs_coords.good()) {
      std::size_t buf_state;
      std::string buf_coords;
      ifs_states >> buf_state;
      std::getline(ifs_coords, buf_coords);
      if (ifs_states.good() && ifs_coords.good()) {
        if (buf_state == selected_state) {
          out() << buf_coords << "\n";
        }
      }
    }
  }
} // end namespace Filter
} // end namespace Clustering

