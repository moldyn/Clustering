
#include <boost/program_options.hpp>
#include <omp.h>

#include "tools.hpp"
#include "density_clustering.hpp"
#include "mpp.hpp"
#include "logger.hpp"

int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  using namespace Clustering::Density;
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
    ("file,f", b_po::value<std::string>(), "input (required): phase space coordinates (space separated ASCII).")
    ("radius,r", b_po::value<float>(), "parameter (required): hypersphere radius.")
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
    ("qmin-from", b_po::value<float>()->default_value(0.01, "0.01"), "initial Qmin value (default: 0.01).")
    ("qmin-to", b_po::value<float>()->default_value(1.0, "1.00"), "final Qmin value (default: 1.00).")
    ("qmin-step", b_po::value<float>()->default_value(0.01, "0.01"), "Qmin stepping (default: 0.01).")
    ("lagtime", b_po::value<std::size_t>(), "lagtime (in units of frame numbers).")
  ;
  // parse cmd arguments
  b_po::options_description desc;
  if (mode == DENSITY) {
    desc.add(desc_dens);
  } else {
    desc.add(desc_mpp);
  }
  try {
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).run(), args);
    b_po::notify(args);
  } catch (b_po::error& e) {
    if ( ! args["help"].as<bool>()) {
      std::cout << "\n" << e.what() << "\n\n" << std::endl;
    }
    std::cout << desc << std::endl;
    return EXIT_FAILURE;
  }
  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }
  // setup general flags / options
  verbose = args["verbose"].as<bool>();
  const std::string input_file = args["file"].as<std::string>();
  const float radius = args["radius"].as<float>();
  // setup OpenMP
  const int n_threads = args["nthreads"].as<int>();
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  // setup coords
  float* coords;
  std::size_t n_rows;
  std::size_t n_cols;
  logger(std::cout) << "reading coords" << std::endl;
  std::tie(coords, n_rows, n_cols) = read_coords<float>(input_file);
  #ifdef DC_USE_OPENCL
  DC_OpenCL::setup(coords, n_rows, n_cols);
  #endif
  //// free energies
  std::vector<float> free_energies;
  if (args.count("free-energy-input")) {
    logger(std::cout) << "re-using free energy data." << std::endl;
    std::ifstream ifs(args["free-energy-input"].as<std::string>());
    if (ifs.fail()) {
      std::cerr << "error: cannot open file '" << args["free-energy-input"].as<std::string>() << "'" << std::endl;
      exit(EXIT_FAILURE);
    } else {
      while(ifs.good()) {
        float buf;
        ifs >> buf;
        if ( ! ifs.fail()) {
          free_energies.push_back(buf);
        }
      }
    }
  } else if (args.count("free-energy") || args.count("output")) {
    logger(std::cout) << "calculating free energies" << std::endl;
    #ifdef DC_USE_OPENCL
    free_energies = calculate_free_energies(
                      DC_OpenCL::calculate_populations(radius));
    #else
    free_energies = calculate_free_energies(
                      calculate_populations(coords, n_rows, n_cols, radius));
    #endif
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
    logger(std::cout) << "re-using nearest neighbor data." << std::endl;
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
    logger(std::cout) << "calculating nearest neighbors" << std::endl;
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
      logger(std::cout) << "reading initial clusters from file." << std::endl;
      std::ifstream ifs(args["input"].as<std::string>());
      if (ifs.fail()) {
        std::cerr << "error: cannot open file '" << args["input"].as<std::string>() << "'" << std::endl;
        exit(EXIT_FAILURE);
      } else {
        while (ifs.good()) {
          std::size_t buf;
          ifs >> buf;
          if ( ! ifs.fail()) {
            clustering.push_back(buf);
          }
        }
      }
    } else {
      logger(std::cout) << "calculating initial clusters" << std::endl;
      if (args.count("threshold") == 0) {
        std::cerr << "error: need threshold value for initial clustering" << std::endl;
        exit(EXIT_FAILURE);
      }
      float threshold = args["threshold"].as<float>();
      clustering = initial_density_clustering(free_energies, nh, threshold, coords, n_rows, n_cols);
    }
    if ( ! args["only-initial"].as<bool>()) {
      logger(std::cout) << "assigning low density states to initial clusters" << std::endl;
      clustering = assign_low_density_frames(clustering, nh_high_dens, free_energies);
    }
    // write clusters to file
    {
      logger(std::cout) << "writing clusters to file " << output_file << std::endl;
      std::ofstream ofs(output_file);
      if (ofs.fail()) {
        std::cerr << "error: cannot open file '" << output_file << "' for writing." << std::endl;
        exit(EXIT_FAILURE);
      }
      for (std::size_t i: clustering) {
        ofs << i << "\n";
      }
    }
  }
  logger(std::cout) << "freeing coords" << std::endl;
  free_coords(coords);
  return EXIT_SUCCESS;
}

