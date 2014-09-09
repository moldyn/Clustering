
#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>
#include <string>

#include <boost/program_options.hpp>

namespace {
  bool verbose = false;
  std::ostream devnull(0);
  std::ostream& logger(std::ostream& s) {
    if (verbose) {
      return s;
    } else {
      return devnull; 
    }
  }
} // end local namespace


std::vector<uint>
read_clustered_trajectory(std::string filename) {
  std::vector<uint> traj;
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    while (ifs.good()) {
      uint buf;
      ifs >> buf;
      if ( ! ifs.fail()) {
        traj.push_back(buf);
      }
    }
  }
  return traj;
}


void
write_clustered_trajectory(std::string filename, std::vector<uint> traj) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    for (uint c: traj) {
      ofs << c << "\n";
    }
  }
}


////////

int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  b_po::variables_map args;
  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "build network information from density based clustering."
    "\n"
    "options"));
  desc.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false), "verbose mode: print runtime information to STDOUT.")
  ;
  // parse cmd arguments
  try {
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).run(), args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cout << "\n" << e.what() << "\n\n" << std::endl;
    }
    std::cout << desc << std::endl;
    return 2;
  }
  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return 1;
  }
  // setup general flags / options
  verbose = args["verbose"].as<bool>();

  std::vector<uint> cl = read_clustered_trajectory(input_filename);





  return 0;
}

