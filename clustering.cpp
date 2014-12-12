
#include "config.hpp"

#include "density_clustering.hpp"
#ifdef DC_USE_MPI
  #include "density_clustering_mpi.hpp"
#endif

#include "mpp.hpp"
#include "logger.hpp"
#include "tools.hpp"

#include <omp.h>
#include <boost/program_options.hpp>

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
    ("radius,r", b_po::value<float>(), "parameter: hypersphere radius.")
    // optional
    ("threshold,t", b_po::value<float>(), "parameter: Free Energy threshold for clustering (FEL is normalized to zero).")
    ("output,o", b_po::value<std::string>(), "output (optional): clustering information.")
    ("input,i", b_po::value<std::string>(), "input (optional): initial state definition.")
    ("radii,R", b_po::value<std::vector<float>>()->multitoken(), "parameter: list of radii for population/free energy calculations "
                                                                 "(i.e. compute populations/free energies for several radii in one go).")
    ("population,p", b_po::value<std::string>(), "output (optional): population per frame (if -R is set: this defines only the basename).")
    ("free-energy,d", b_po::value<std::string>(), "output (optional): free energies per frame (if -R is set: this defines only the basename).")
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
    ("concat-nframes", b_po::value<std::size_t>(),
      "input (parameter): no. of frames per (equally sized) sub-trajectory for concatenated trajectory files.")
    ("concat-limits", b_po::value<std::string>(),
      "input (file): file with frame ids (base 0) of first frames per (not equally sized) sub-trajectory for concatenated trajectory files.")
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
      #ifdef DC_USE_MPI
        Clustering::Density::MPI::main(args);
      #else
        Clustering::Density::main(args);
      #endif
      break;
    case MPP:
      Clustering::MPP::main(args);
      break;
    default:
      std::cerr << "error: unknown mode. this should never happen." << std::endl;
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

