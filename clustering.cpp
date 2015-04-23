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
#include "config.hpp"
// sub-modules
#include "density_clustering.hpp"
#ifdef DC_USE_MPI
  #include "density_clustering_mpi.hpp"
#endif
#include "mpp.hpp"
#include "network_builder.hpp"
#include "state_filter.hpp"
#include "coring.hpp"
// toolset
#include "logger.hpp"
#include "tools.hpp"

#include <omp.h>
#include <boost/program_options.hpp>

/*! 
 * The main function of 'clustering' is essentially a wrapper around the different sub-modules.
 * These include:
 *   - density: for density-based clustering on the given geometric space
 *   - network: for the network/microstate generation from density-based clustering results
 *   - mpp: for Most Probable Path clustering of microstates
 *   - coring: for boundary corrections of clustered state trajectories
 *   - filter: for fast filtering of coordinates, order parameters, etc. based on\n
 *             a given state trajectory (i.e. clustering result)
 */
int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  std::string general_help = 
    "clustering - a classification framework for MD data\n"
    "\n"
    "modes:\n"
    "  density: run density clustering\n"
    "  network: build network from density clustering results\n"
    "  mpp:     run MPP (Most Probable Path) clustering\n"
    "           (based on density-results)\n"
    "  coring:  boundary corrections for clustering results.\n"
    "  filter:  filter phase space (e.g. dihedrals) for given state\n"
    "\n"
    "usage:\n"
    "  clustering MODE --option1 --option2 ...\n"
    "\n"
    "for a list of available options per mode, run with '-h' option, e.g.\n"
    "  clustering density -h\n"
  ;

  enum {DENSITY, MPP, NETWORK, FILTER, CORING} mode;

  if (argc <= 2) {
    std::cerr << general_help;
    return EXIT_FAILURE;

  } else {
    std::string str_mode(argv[1]);
    if (str_mode.compare("density") == 0) {
      mode = DENSITY;
    } else if (str_mode.compare("mpp") == 0) {
      mode = MPP;
    } else if (str_mode.compare("network") == 0) {
      mode = NETWORK;
    } else if (str_mode.compare("filter") == 0) {
      mode = FILTER;
    } else if (str_mode.compare("coring") == 0) {
      mode = CORING;
    } else {
      std::cerr << "\nerror: unrecognized mode '" << str_mode << "'\n\n";
      std::cerr << general_help;
      return EXIT_FAILURE;
    }
  }
  b_po::variables_map args;
  b_po::positional_options_description pos_opts;
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
    ("threshold-screening,T", b_po::value<std::vector<float>>()->multitoken(),
                                          "parameters: screening of free energy landscape. format: FROM STEP TO; e.g.: '-T 0.1 0.1 11.1'.\n"
                                          "for threshold-screening, --output denotes the basename only. output files will have the"
                                          " current threshold limit appended to the given filename.")
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
    ("nearest-neighbor-input,B", b_po::value<std::string>()->required(), "input (required): nearest neighbors.")
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
  // network options
  b_po::options_description desc_network (std::string(argv[1]).append(
    "\n\n"
    "TODO: description for network builder"
    "\n"
    "options"));
  desc_network.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    // optional
    ("basename,b", b_po::value<std::string>()->default_value("clust.\%0.1f"),
          "(optional): basename of input files (default: clust.\%0.1f).")
    ("min", b_po::value<float>()->default_value(0.1f, "0.1"), "(optional): minimum free energy (default: 0.1).")
    ("max", b_po::value<float>()->default_value(8.0f, "8.0"), "(optional): maximum free energy (default: 8.0).")
    ("step", b_po::value<float>()->default_value(0.1f, "0.1"), "(optional): minimum free energy (default: 0.1).")
    ("minpop,p", b_po::value<std::size_t>()->default_value(1),
          "(optional): minimum population of node to be considered for network (default: 1).")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false), "verbose mode: print runtime information to STDOUT.")
  ;
  // filter options
  b_po::options_description desc_filter (std::string(argv[1]).append(
    "\n\n"
    "filter phase space (e.g. dihedral angles, cartesian coords, etc.) for given state."
    "\n"
    "options"));
  desc_filter.add_options()
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
  // coring options
  b_po::options_description desc_coring (std::string(argv[1]).append(
    "\n\n"
    "compute boundary corrections for clustering results."
    "\n"
    "options"));
  desc_coring.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
        "show this help.")
    // optional
    ("states,s", b_po::value<std::string>()->required(),
        "(required): file with state information (i.e. clustered trajectory")
    ("windows,w", b_po::value<std::string>()->required(), 
        "(required): file with window sizes."
        "format is space-separated lines of\n\n"
        "STATE_ID WINDOW_SIZE\n\n"
        "use * as STATE_ID to match all (other) states.\n"
        "e.g.:\n\n"
        "* 20\n"
        "3 40\n"
        "4 60\n\n"
        "matches 40 frames to state 3, 60 frames to state 4 and 20 frames to all the other states")
    ("output,o", b_po::value<std::string>(),
        "(optional): cored trajectory")
    ("distribution,d", b_po::value<std::string>(),
        "(optional): write waiting time distributions to file.")
    ("cores,c", b_po::value<std::string>(),
        "(optional): write core information to file, i.e. trajectory with state name if in core region or -1 if not in core region")
    ("concat-nframes", b_po::value<std::size_t>(),
      "input (optional parameter): no. of frames per (equally sized) sub-trajectory for concatenated trajectory files.")
    ("concat-limits", b_po::value<std::string>(),
      "input (optional, file): file with frame ids (base 0) of first frames per (not equally sized) sub-trajectory for concatenated trajectory files.")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false),
        "verbose mode: print runtime information to STDOUT.")
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
    case NETWORK:
      desc.add(desc_network);
      break;
    case FILTER:
      desc.add(desc_filter);
      break;
    case CORING:
      desc.add(desc_coring);
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
  if (args.count("verbose")) {
    Clustering::verbose = args["verbose"].as<bool>();
  }
  // setup OpenMP
  int n_threads = 0;
  if (args.count("nthreads")) {
    n_threads = args["nthreads"].as<int>();
  }
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  // run selected subroutine
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
    case NETWORK:
      Clustering::NetworkBuilder::main(args);
      break;
    case FILTER:
      Clustering::Filter::main(args);
      break;
    case CORING:
      Clustering::Coring::main(args);
      break;
    default:
      std::cerr << "error: unknown mode. this should never happen." << std::endl;
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

