
#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>

#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  b_po::variables_map args;
  b_po::options_description desc (
    "filter phase space (e.g. dihedral angles) for given state."
    "\n"
    "options");
  desc.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    // optional
    ("states,s", b_po::value<std::string>()->required(),
          "(required): file with state information (i.e. clustered trajectory).")
    ("phase-space,p", b_po::value<std::string>()->required(),
          "(required): file with phase space data.")
    ("output,o", b_po::value<std::string>(), "(optional): filtered data. will write to STDOUT if not given.")
    ("selected-state", b_po::value<std::size_t>()->required(),
          "(required): state id for selected state.")
  ;
  b_po::positional_options_description pos_opts;
  pos_opts.add("selected-state", 1);
  // parse cmd arguments
  auto print_help = [&desc]() -> void {
    std::cout << "usage: state_filter [options] SELECTED_STATE" << std::endl;
    std::cout << desc << std::endl;
  };
  try {
    b_po::store(
      b_po::command_line_parser(argc, argv)
        .options(desc)
        .positional(pos_opts)
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
    return EXIT_FAILURE;
  }
  if (args["help"].as<bool>()) {
    print_help();
    return EXIT_SUCCESS;
  }
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
      return EXIT_FAILURE;
    }
  }
  auto out = [&ofs]() -> std::ostream& {
    return ofs.is_open() ? ofs : std::cout;
  };
  if (ifs_states.fail()) {
    std::cerr << "error: cannot open file '" << fname_states << "' for reading." << std::endl;
    return EXIT_FAILURE;
  }
  std::ifstream ifs_coords(fname_coords);
  if (ifs_coords.fail()) {
    std::cerr << "error: cannot open file '" << fname_coords << "' for reading." << std::endl;
    return EXIT_FAILURE;
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
  return EXIT_SUCCESS;
}

