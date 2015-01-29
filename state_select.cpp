
#include "coords_file/coords_file.hpp"

#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <queue>

namespace {
  int io_error(std::string fname) {
    std::cerr << "error: cannot open file " << fname << "." << std::endl;
    return EXIT_FAILURE;
  }
} // end local namespace


namespace Clustering {
namespace Filter {

  int main(int argc, char* argv[]) {
    namespace b_po = boost::program_options;
    b_po::variables_map args;
    b_po::options_description desc (
      "filter phase space (e.g. dihedral angles, cartesian coords, etc.) for given state."
      "\n"
      "options");
    desc.add_options()
      ("help,h", b_po::bool_switch()->default_value(false),
            "show this help.")
      ("states,s", b_po::value<std::string>()->required(),
            "(required): file with state information (i.e. clustered trajectory).")
      ("coords,c", b_po::value<std::string>(),
            "file with coordinates (either plain ASCII or GROMACS' xtc).")
      ("output,o", b_po::value<std::string>(),
            "filtered data.")
      ("state,S", b_po::value<std::size_t>(),
            "state id of selected state.")
              
      ("list", b_po::bool_switch()->default_value(false),
            "list states and their populations")
    ;
    // parse cmd arguments
    auto print_help = [&desc]() -> void {
      std::cout << "usage: state_select [options]" << std::endl;
      std::cout << desc << std::endl;
    };
    try {
      b_po::store(
        b_po::command_line_parser(argc, argv)
          .options(desc)
          .run(),
        args);
      b_po::notify(args);
    } catch (b_po::error& e) {
      if ( ! args["help"].as<bool>()) {
        print_help();
        std::cout << "\n" << e.what() << "\n\n" << std::endl;
      } else {
        print_help();
      }
      exit(EXIT_FAILURE);
    }
    if (args["help"].as<bool>()) {
      print_help();
      exit(EXIT_SUCCESS);
    }
    // load states
    std::string fname_states = args["states"].as<std::string>();
    std::vector<std::size_t> states;
    {
      std::ifstream ifs(fname_states);
      if (ifs.fail()) {
        return io_error(fname_states);
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

