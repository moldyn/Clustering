
#include "clustering.hpp"

#include <omp.h>

#include "tools.hpp"
#include "density_clustering.hpp"
#include "mpp.hpp"
#include "logger.hpp"

template <typename NUM>
std::vector<NUM>
read_single_column(std::string filename) {
  std::vector<NUM> dat;
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    while (ifs.good()) {
      NUM buf;
      ifs >> buf;
      if ( ! ifs.fail()) {
        dat.push_back(buf);
      }
    }
  }
  return dat;
}

template <typename NUM>
void
write_single_column(std::string filename, std::vector<NUM> dat) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  for (NUM i: dat) {
    ofs << i << "\n";
  }
}

std::vector<std::size_t>
read_clustered_trajectory(std::string filename) {
  return read_single_column<std::size_t>(filename);
}

std::vector<float>
read_free_energies(std::string filename) {
  return read_single_column<float>(filename);
}

template <typename KEY, typename VAL>
void
write_map(std::string filename, std::map<KEY, VAL> mapping) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  for (auto key_val: mapping) {
    ofs << key_val.first << " " << key_val.second << "\n";
  }
}

std::map<std::size_t, std::size_t>
microstate_populations(std::vector<std::size_t> traj) {
  std::map<std::size_t, std::size_t> populations;
  for (std::size_t state: traj) {
    if (populations.count(state) == 0) {
      populations[state] = 1;
    } else {
      ++populations[state];
    }
  }
  return populations;
}

void density_main(boost::program_options::variables_map args) {
  using namespace Clustering::Density;
  const std::string input_file = args["file"].as<std::string>();
  const float radius = args["radius"].as<float>();
  // setup coords
  float* coords;
  std::size_t n_rows;
  std::size_t n_cols;
  Clustering::logger(std::cout) << "reading coords" << std::endl;
  std::tie(coords, n_rows, n_cols) = read_coords<float>(input_file);
  #ifdef DC_USE_OPENCL
  DC_OpenCL::setup(coords, n_rows, n_cols);
  #endif
  //// free energies
  std::vector<float> free_energies;
  if (args.count("free-energy-input")) {
    Clustering::logger(std::cout) << "re-using free energy data." << std::endl;
    free_energies = read_free_energies(args["free-energy-input"].as<std::string>());
  } else if (args.count("free-energy") || args.count("population") || args.count("output")) {
    Clustering::logger(std::cout) << "calculating populations" << std::endl;
    #ifdef DC_USE_OPENCL
      std::vector<std::size_t> pops = DC_OpenCL::calculate_poputions(radius);
    #else
      std::vector<std::size_t> pops = calculate_populations(coords, n_rows, n_cols, radius);
    #endif
    if (args.count("population")) {
      std::ofstream ofs(args["population"].as<std::string>());
      for (std::size_t p: pops) {
        ofs << p << "\n";
      }
    }
    Clustering::logger(std::cout) << "calculating free energies" << std::endl;
    free_energies = calculate_free_energies(pops);
    if (args.count("free-energy")) {
      std::ofstream ofs(args["free-energy"].as<std::string>());
      ofs << std::scientific;
      for (float f: free_energies) {
        ofs << f << "\n";
      }
    }
  }
  //// nearest neighbors
  Neighborhood nh;
  Neighborhood nh_high_dens;
  if (args.count("nearest-neighbors-input")) {
    Clustering::logger(std::cout) << "re-using nearest neighbor data." << std::endl;
    std::ifstream ifs(args["nearest-neighbors-input"].as<std::string>());
    if (ifs.fail()) {
      std::cerr << "error: cannot open file '" << args["nearest-neighbors-input"].as<std::string>() << "'" << std::endl;
      exit(EXIT_FAILURE);
    } else {
      std::size_t i=0;
      while (ifs.good()) {
        std::size_t buf1;
        float buf2;
        std::size_t buf3;
        float buf4;
        ifs >> buf1;
        ifs >> buf2;
        ifs >> buf3;
        ifs >> buf4;
        if ( ! ifs.fail()) {
          nh[i] = std::pair<std::size_t, float>(buf1, buf2);
          nh_high_dens[i] = std::pair<std::size_t, float>(buf3, buf4);
          ++i;
        }
      }
    }
  } else if (args.count("nearest-neighbors") || args.count("output")) {
    Clustering::logger(std::cout) << "calculating nearest neighbors" << std::endl;
    auto nh_tuple = nearest_neighbors(coords, n_rows, n_cols, free_energies);
    nh = std::get<0>(nh_tuple);
    nh_high_dens = std::get<1>(nh_tuple);
    if (args.count("nearest-neighbors")) {
      std::ofstream ofs(args["nearest-neighbors"].as<std::string>());
      auto p = nh.begin();
      auto p_hd = nh_high_dens.begin();
      while (p != nh.end() && p_hd != nh_high_dens.end()) {
        // first: key (not used)
        // second: neighbor
        // second.first: id; second.second: squared dist
        ofs << p->second.first    << " " << p->second.second    << " "
            << p_hd->second.first << " " << p_hd->second.second << "\n";
        ++p;
        ++p_hd;
      }
    }
  }
  //// clustering
  if (args.count("output")) {
    const std::string output_file = args["output"].as<std::string>();
    std::vector<std::size_t> clustering;
    if (args.count("input")) {
      Clustering::logger(std::cout) << "reading initial clusters from file." << std::endl;
      clustering = read_clustered_trajectory(args["input"].as<std::string>());
    } else {
      Clustering::logger(std::cout) << "calculating initial clusters" << std::endl;
      if (args.count("threshold") == 0) {
        std::cerr << "error: need threshold value for initial clustering" << std::endl;
        exit(EXIT_FAILURE);
      }
      float threshold = args["threshold"].as<float>();
      clustering = initial_density_clustering(free_energies, nh, threshold, coords, n_rows, n_cols);
    }
    if ( ! args["only-initial"].as<bool>()) {
      Clustering::logger(std::cout) << "assigning low density states to initial clusters" << std::endl;
      clustering = assign_low_density_frames(clustering, nh_high_dens, free_energies);
    }
    Clustering::logger(std::cout) << "writing clusters to file " << output_file << std::endl;
    write_single_column<std::size_t>(output_file, clustering);
  }
  Clustering::logger(std::cout) << "freeing coords" << std::endl;
  free_coords(coords);
}

void mpp_main(boost::program_options::variables_map args) {
  using namespace Clustering::MPP;
  std::string basename = args["basename"].as<std::string>();
  // load initial trajectory
  std::map<std::size_t, std::size_t> transitions;
  std::map<std::size_t, std::size_t> max_pop;
  std::map<std::size_t, float> max_qmin;
  Clustering::logger(std::cout) << "loading microstates" << std::endl;
  std::vector<std::size_t> traj = read_clustered_trajectory(args["input"].as<std::string>());
  Clustering::logger(std::cout) << "loading free energies" << std::endl;
  std::vector<float> free_energy = read_free_energies(args["input"].as<std::string>());
  float q_min_from = args["qmin-from"].as<float>();
  float q_min_to = args["qmin-to"].as<float>();
  float q_min_step = args["qmin-step"].as<float>();
  int lagtime = args["lagtime"].as<int>();
  Clustering::logger(std::cout) << "beginning q_min loop" << std::endl;
  for (float q_min=q_min_from; q_min <= q_min_to; q_min += q_min_step) {
    auto traj_sinks = fixed_metastability_clustering(traj, q_min, lagtime, free_energy);
    // write trajectory at current Qmin level to file
    traj = std::get<0>(traj_sinks);
    write_single_column(stringprintf("%s_traj_%0.2f.dat", basename.c_str(), q_min), traj);
    // save transitions (i.e. lumping of states)
    std::map<std::size_t, std::size_t> sinks = std::get<1>(traj_sinks);
    transitions.insert(sinks.begin(), sinks.end());
    // write microstate populations to file
    std::map<std::size_t, std::size_t> pops = microstate_populations(traj);
    write_map<std::size_t, std::size_t>(stringprintf("%s_pop_%0.2f.dat", basename.c_str(), q_min), pops);
    // collect max. pops + max. q_min per microstate
    for (std::size_t id: std::set<std::size_t>(traj.begin(), traj.end())) {
      max_pop[id] = pops[id];
      max_qmin[id] = q_min;
    }
  }
  write_map<std::size_t, std::size_t>(stringprintf("%s_transitions.dat", basename.c_str()), transitions);
  write_map<std::size_t, std::size_t>(stringprintf("%s_max_pop.dat", basename.c_str()), max_pop);
  write_map<std::size_t, float>(stringprintf("%s_max_qmin.dat", basename.c_str()), max_qmin);
}

int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  std::string general_help = 
    "clustering - a classification framework for MD data\n"
    "\n"
    "modes:\n"
    "  density: run density clustering\n"
    "  mpp:     run MPP (Most Probable Path) clustering\n"
    "           (based on density-results)\n"
    "\n"
    "usage:\n"
    "  clustering MODE --option1 --option2 ...\n"
    "\n"
    "for a list of available options per mode, run with '-h' option, e.g.\n"
    "  clustering density -h\n"
  ;
  enum {DENSITY, MPP} mode;
  if (argc <= 2) {
    std::cerr << general_help;
    return EXIT_FAILURE;
  } else {
    std::string str_mode(argv[1]);
    if (str_mode.compare("density") == 0) {
      mode = DENSITY;
    } else if (str_mode.compare("mpp") == 0) {
      mode = MPP;
    } else {
      std::cerr << "\nerror: unrecognized mode '" << str_mode << "'\n\n";
      std::cerr << general_help;
      return EXIT_FAILURE;
    }
  }
  b_po::variables_map args;
  // density options
  b_po::options_description desc_dens (std::string(argv[1]).append(
    "\n\n"
    "perform clustering of MD data based on phase space densities.\n"
    "densities are approximated by counting neighboring frames inside\n"
    "a n-dimensional hypersphere of specified radius.\n"
    "distances are measured with n-dim P2-norm.\n"
    "\n"
    "options"));
  desc_dens.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    ("file,f", b_po::value<std::string>()->required(), "input (required): phase space coordinates (space separated ASCII).")
    ("radius,r", b_po::value<float>()->required(), "parameter (required): hypersphere radius.")
    // optional
    ("threshold,t", b_po::value<float>(), "parameter: Free Energy threshold for clustering (FEL is normalized to zero).")
    ("output,o", b_po::value<std::string>(), "output (optional): clustering information.")
    ("input,i", b_po::value<std::string>(), "input (optional): initial state definition.")
    ("population,p", b_po::value<std::string>(), "output (optional): population per frame.")
    ("free-energy,d", b_po::value<std::string>(), "output (optional): free energies per frame.")
    ("free-energy-input,D", b_po::value<std::string>(), "input (optional): reuse free energy info.")
    ("nearest-neighbors,b", b_po::value<std::string>(), "output (optional): nearest neighbor info.")
    ("nearest-neighbors-input,B", b_po::value<std::string>(), "input (optional): reuse nearest neighbor info.")
    // defaults
    ("only-initial,I", b_po::bool_switch()->default_value(false),
                      "only assign initial (i.e. low free energy / high density) frames to clusters. "
                      "leave unclustered frames as state '0'.")
    ("nthreads,n", b_po::value<int>()->default_value(0),
                      "number of OpenMP threads. default: 0; i.e. use OMP_NUM_THREADS env-variable.")
    ("verbose,v", b_po::bool_switch()->default_value(false), "verbose mode: print runtime information to STDOUT.")
  ;
  // MPP options
  b_po::options_description desc_mpp (std::string(argv[1]).append(
    "\n\n"
    "TODO: description for MPP"
    "\n"
    "options"));
  desc_mpp.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    ("input,i", b_po::value<std::string>()->required(), "input (required): initial state definition.")
    ("free-energy-input,D", b_po::value<std::string>()->required(), "input (required): reuse free energy info.")
    ("lagtime,l", b_po::value<int>()->required(), "input (required): lagtime in units of frame numbers.")
    ("qmin-from", b_po::value<float>()->default_value(0.01, "0.01"), "initial Qmin value (default: 0.01).")
    ("qmin-to", b_po::value<float>()->default_value(1.0, "1.00"), "final Qmin value (default: 1.00).")
    ("qmin-step", b_po::value<float>()->default_value(0.01, "0.01"), "Qmin stepping (default: 0.01).")
    // defaults
    ("basename", b_po::value<std::string>()->default_value("mpp"), "basename for output files (default: 'mpp').")
    ("nthreads,n", b_po::value<int>()->default_value(0),
                      "number of OpenMP threads. default: 0; i.e. use OMP_NUM_THREADS env-variable.")
    ("verbose,v", b_po::bool_switch()->default_value(false), "verbose mode: print runtime information to STDOUT.")
  ;
  // parse cmd arguments
  b_po::options_description desc;
  switch(mode){
    case DENSITY:
      desc.add(desc_dens);
      break;
    case MPP:
      desc.add(desc_mpp);
      break;
    default:
      std::cerr << "error: unknown mode. this should never happen." << std::endl;
      return EXIT_FAILURE;
  }
  try {
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).run(), args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cerr << "\nerror parsing arguments:\n\n" << e.what() << "\n\n" << std::endl;
    }
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }
  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }
  // setup defaults
  Clustering::verbose = args["verbose"].as<bool>();
  // setup OpenMP
  const int n_threads = args["nthreads"].as<int>();
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  // run clustering subroutines
  switch(mode) {
    case DENSITY:
      density_main(args);
      break;
    case MPP:
      mpp_main(args);
      break;
    default:
      std::cerr << "error: unknown mode. this should never happen." << std::endl;
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

