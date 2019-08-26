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

/*!\file
 * \brief Wrapper for *clustering* package
 *
 * The main function of *clustering* is essentially being a wrapper around the different sub-modules:
 * density, network, mpp, coring, noise and filter.
 * \sa \link main()
 */

#include "config.hpp"
// sub-modules
#include "density_clustering.hpp"
#ifdef USE_CUDA
  #include "density_clustering_cuda.hpp"
#endif
#ifdef DC_USE_MPI
  #include "density_clustering_mpi.hpp"
#endif
#include "mpp.hpp"
#include "network_builder.hpp"
#include "state_filter.hpp"
#include "coring.hpp"
#include "noise.hpp"
// toolset
#include "logger.hpp"
#include "tools.hpp"

#include <omp.h>
#include <time.h>
#include <boost/program_options.hpp>
/*! \brief Parses option and execute corresponding sub-module
 *
 * This method parses the arguments and calls the corresponding sub-modules.
 * \param density for density-based clustering on the given geometric space
 * \param network for the network/microstate generation from density-based clustering results
 * \param mpp for Most Probable Path clustering of microstates
 * \param coring for boundary corrections of clustered state trajectories
 * \param noise for defining and dynamically reassigning noise
 * \param filter for fast filtering of coordinates, order parameters, etc. based on\n
 *               a given state trajectory (i.e. clustering result)
 */
int main(int argc, char* argv[]) {
  namespace b_po = boost::program_options;
  std::string version_number = "v1.1.1";
  // generate header string
  std::string leading_whitespace(25 - (19 + version_number.size())/2, ' ');
  std::ostringstream header_ostring;
  header_ostring << "\n" << leading_whitespace
                 << "~~~ clustering " + version_number + " ~~~\n";
  if (argc > 2){
    std::string leading_whitespace2nd(25 - (4 + strlen(argv[1]))/2, ' ');
    header_ostring << leading_whitespace2nd << "~ " << argv[1] << " ~\n";
  }
  std::string clustering_copyright =
    header_ostring.str() + ""
    "\nclustering " + version_number + ": a classification framework for MD data\n"
    "Copyright (c) 2015-2019, Florian Sittel and Daniel Nagel\n\n";
  std::string general_help =
    clustering_copyright +
    "modes:\n"
    "  density: run density clustering\n"
    "  network: build network from density clustering results\n"
    "  mpp:     run MPP (Most Probable Path) clustering\n"
    "           (based on density-results)\n"
    "  coring:  boundary corrections for clustering results.\n"
    "  noise:   defining and dynamically reassigning noise.\n"
    "  filter:  filter phase space (e.g. dihedrals) for given state\n"
    "  stats:   give statistics of state trajectory\n"
    "\n"
    "usage:\n"
    "  clustering MODE --option1 --option2 ...\n"
    "\n"
    "for a list of available options per mode, run with '-h' option, e.g.\n"
    "  clustering density -h\n\n"
  ;
  #ifdef USE_CUDA
        general_help += "this binary is parallized with cuda\n\n";
  #else
        general_help += "this binary is parallized with openmp\n\n";
  #endif

  enum {DENSITY, MPP, NETWORK, FILTER, STATS, CORING, NOISE} mode;

#ifdef USE_CUDA
  // check for CUDA-enabled GPUs (will fail if none found)
  Clustering::Density::CUDA::get_num_gpus();
#endif

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
    } else if (str_mode.compare("stats") == 0) {
      mode = STATS;
    } else if (str_mode.compare("coring") == 0) {
      mode = CORING;
    } else if (str_mode.compare("noise") == 0) {
      mode = NOISE;
    } else {
      std::cerr << "\nerror: unrecognized mode '" << str_mode << "'\n\n";
      std::cerr << general_help;
      return EXIT_FAILURE;
    }
  }
  b_po::variables_map args;
  b_po::positional_options_description pos_opts;
  // density options
  b_po::options_description desc_dens (
    clustering_copyright + std::string(argv[1]) + ": \n"
    "perform clustering of MD data based on phase space densities.\n"
    "densities are approximated by counting neighboring frames inside\n"
    "a n-dimensional hypersphere of specified radius.\n"
    "distances are measured with n-dim P2-norm.\n"
    "\n"
    "options");
  desc_dens.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    ("file,f", b_po::value<std::string>()->required(),
          "input (required): phase space coordinates (space separated ASCII).")
    // optional
    ("radius,r", b_po::value<float>(),
          "parameter: hypersphere radius. If not used, the lumping radius "
          "will be used instead.")
    ("threshold-screening,T", b_po::value<std::vector<float>>()->multitoken(),
          "parameters: screening of free energy landscape. "
          "format: FROM STEP TO; e.g.: '-T 0.1 0.1 11.1'.\n"
          "set -T -1 for default values: FROM=0.1, STEP=0.1, TO=MAX_FE.\n"
          "parameters may be given partially, e.g.: -T 0.2 0.4 to start "
          "at 0.2 and go to MAX_FE at steps 0.4.\n"
          "for threshold-screening, --output denotes the basename only. "
          "output files will have the "
          "current threshold limit appended to the given filename.")
    ("output,o", b_po::value<std::string>(),
          "output (optional): clustering information.")
    ("input,i", b_po::value<std::string>(),
          "input (optional): initial state definition.")
    ("radii,R", b_po::value<std::vector<float>>()->multitoken(),
          "parameter: list of radii for population/free energy calculations "
          "(i.e. compute populations/free energies for several radii in one go).")
    ("population,p", b_po::value<std::string>(),
          "output (optional): population per frame (if -R is set: "
          "this defines only the basename).")
    ("free-energy,d", b_po::value<std::string>(),
          "output (optional): free energies per frame "
          "(if -R is set: this defines only the basename).")
    ("free-energy-input,D", b_po::value<std::string>(),
          "input (optional): reuse free energy info.")
    ("nearest-neighbors,b", b_po::value<std::string>(),
          "output (optional): nearest neighbor info.")
    ("nearest-neighbors-input,B", b_po::value<std::string>(),
          "input (optional): reuse nearest neighbor info.")
    // defaults
    ("nthreads,n", b_po::value<int>()->default_value(0),
          "number of OpenMP threads. default: 0; i.e. use OMP_NUM_THREADS env-variable.")
    ("verbose,v", b_po::bool_switch()->default_value(false),
          "verbose mode: print runtime information to STDOUT.")
  ;
  // MPP options
  b_po::options_description desc_mpp (
    clustering_copyright + std::string(argv[1]) + ": \n"
    "performs a most probable path (MPP) clustering based on the given lag time."
    "\n"
    "options");
  desc_mpp.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    ("states,s", b_po::value<std::string>()->required(),
          "(required): file with state information (i.e. clustered trajectory")
    ("free-energy-input,D", b_po::value<std::string>()->required(),
          "input (required): reuse free energy info.")
    ("lagtime,l", b_po::value<int>()->required(),
          "input (required): lagtime in units of frame numbers. Note: Lagtime "
          "should be greater than the coring time/ smallest timescale. ")
    ("qmin-from", b_po::value<float>()->default_value(0.01, "0.01"),
          "initial Qmin value (default: 0.01).")
    ("qmin-to", b_po::value<float>()->default_value(1.0, "1.00"),
          "final Qmin value (default: 1.00).")
    ("qmin-step", b_po::value<float>()->default_value(0.01, "0.01"),
          "Qmin stepping (default: 0.01).")
    ("concat-nframes", b_po::value<std::size_t>(),
          "input (parameter): no. of frames per (equally sized) sub-trajectory"
          " for concatenated trajectory files.")
    ("concat-limits", b_po::value<std::string>(),
          "input (file): file with sizes of individual (not equally sized)"
          " sub-trajectories for concatenated trajectory files. e.g.: for a"
          " concatenated trajectory of three chunks of sizes 100, 50 and 300 "
          "frames: '100 50 300'")
    ("tprob", b_po::value<std::string>(),
          "input (file): initial transition probability matrix. "
          "-l still needs to be given, but will be ignored.\n"
          "Format:three space-separated columns 'state_from' 'state_to' 'probability'")
    // defaults
    ("output,o", b_po::value<std::string>()->default_value("mpp"),
          "output (optional): basename for output files (default: 'mpp').")
    ("nthreads,n", b_po::value<int>()->default_value(0),
          "number of OpenMP threads. default: 0; i.e. use OMP_NUM_THREADS env-variable.")
    ("verbose,v", b_po::bool_switch()->default_value(false),
          "verbose mode: print runtime information to STDOUT.")
  ;
  // network options
  b_po::options_description desc_network (
    clustering_copyright + std::string(argv[1]) + ": \n"
    "create a network from screening data."
    "\n"
    "options");
  desc_network.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    ("minpop,p", b_po::value<std::size_t>()->required(),
          "(required): minimum population of node to be considered for network.")
    // optional
    ("basename,b", b_po::value<std::string>()->default_value("clust"),
          "(optional): basename of input files (default: clust).")
    ("output,o", b_po::value<std::string>()->default_value("network"),
          "(optional): basename of output files (default: network).")
    ("min", b_po::value<float>()->default_value(0.1f, "0.10"),
          "(optional): minimum free energy (default:  0.10).")
    ("max", b_po::value<float>()->default_value(0.0f, "0"),
          "(optional): maximum free energy (default:  0; i.e. max. available).")
    ("step", b_po::value<float>()->default_value(0.1f, "0.10"),
          "(optional): free energy stepping (default: 0.10).")
    ("network-html", b_po::bool_switch()->default_value(false),
          "Generate html visualization of fe tree.")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false),
          "verbose mode: print runtime information to STDOUT.")
  ;
  // filter options
  b_po::options_description desc_filter (
    clustering_copyright + std::string(argv[1]) + ": \n"
    "filter phase space (e.g. dihedral angles, cartesian coords, etc.) for given state."
    "\n"
    "options");
  desc_filter.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
          "show this help.")
    ("states,s", b_po::value<std::string>()->required(),
          "(required): file with state information (i.e. clustered trajectory).")
    ("coords,c", b_po::value<std::string>()->required(),
          "(required): file with coordinates (either plain ASCII or GROMACS' xtc).")
    ("output,o", b_po::value<std::string>(),
          "basename of filtered data output (extended by e.g. "
          "basename.state5 for state 5) keeping file extension of input. If not "
          "specified, the inmput name will be used.")
    ("selected-states,S", b_po::value<std::vector<std::size_t>>()->multitoken(),
          "state ids of selected states. Default all states.")
    ("every-nth", b_po::value<std::size_t>()->default_value(1),
          "Take only every nth frame. Default all frames.")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false),
          "verbose mode: print runtime information to STDOUT.")
  ;
  // stats options
  b_po::options_description desc_stats (
    clustering_copyright + std::string(argv[1]) + ": \n"
    "list statistics and population of state trajectory."
    "\n"
    "options");
  desc_stats.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
          "show this help.")
    ("states,s", b_po::value<std::string>()->required(),
          "(required): file with state information (i.e. clustered trajectory).")
    ("concat-nframes", b_po::value<std::size_t>(),
          "input (optional parameter): no. of frames per (equally sized) "
          "sub-trajectory for concatenated trajectory files.")
    ("concat-limits", b_po::value<std::string>(),
          "input (file): file with sizes of individual (not equally sized)"
          " sub-trajectories for concatenated trajectory files. e.g.: for a"
          " concatenated trajectory of three chunks of sizes 100, 50 and 300 "
          "frames: '100 50 300'")
  ;
  // coring options
  b_po::options_description desc_coring (
    clustering_copyright + std::string(argv[1]) + ": \n"
    "compute boundary corrections for clustering results."
    "\n"
    "options");
  desc_coring.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
          "show this help.")
    ("states,s", b_po::value<std::string>()->required(),
          "(required): file with state information (i.e. clustered trajectory)")
    ("windows,w", b_po::value<std::string>()->required(), 
          "(required): file with window sizes."
          "format is space-separated lines of\n\n"
          "STATE_ID WINDOW_SIZE\n\n"
          "use * as STATE_ID to match all (other) states.\n"
          "e.g.:\n\n"
          "* 20\n"
          "3 40\n"
          "4 60\n\n"
          "matches 40 frames to state 3, 60 frames to state 4 and 20 frames "
          "to all the other states.")
    // optional
    ("output,o", b_po::value<std::string>(),
          "(optional): cored trajectory")
    ("distribution,d", b_po::value<std::string>(),
          "(optional): write waiting time distributions to file.")
    ("cores", b_po::value<std::string>(),
          "(optional): write core information to file, i.e. trajectory with "
          "state name if in core region or -1 if not in core region")
    ("concat-nframes", b_po::value<std::size_t>(),
          "input (optional parameter): no. of frames per (equally sized) "
          "sub-trajectory for concatenated trajectory files.")
    ("concat-limits", b_po::value<std::string>(),
          "input (file): file with sizes of individual (not equally sized)"
          " sub-trajectories for concatenated trajectory files. e.g.: for a"
          " concatenated trajectory of three chunks of sizes 100, 50 and 300 "
          "frames: '100 50 300'")
    // defaults
    ("verbose,v", b_po::bool_switch()->default_value(false),
          "verbose mode: print runtime information to STDOUT.")
  ;
  // noise options
  b_po::options_description desc_noise (
    clustering_copyright + std::string(argv[1]) + ": \n"
    "defining and dynamically reassigning noise for clustering results."
    "\n"
    "options");
  desc_noise.add_options()
    ("help,h", b_po::bool_switch()->default_value(false),
          "show this help.")
    ("states,s", b_po::value<std::string>()->required(),
          "(required): file with state information (i.e. clustered trajectory)")
    ("output,o", b_po::value<std::string>()->required(),
          "(required): noise-reassigned trajectory")
    // optional
    ("basename,b", b_po::value<std::string>()->default_value("clust"),
          "(optional): basename of input files (default: clust) used to "
          "determine isolated clusters")
    ("cmin,c", b_po::value<float>()->default_value(0.1f, "0.10"),
          "(optional): population (in percent) threshold below which an "
          "isolated cluster is assigned as noise.(default: 0.1).")
    ("cores", b_po::value<std::string>(),
          "(optional): write core information to file, i.e. trajectory with "
          "state name if in core region or -1 if not in core region")
    ("concat-nframes", b_po::value<std::size_t>(),
          "input (optional parameter): no. of frames per (equally sized) "
          "sub-trajectory for concatenated trajectory files.")
    ("concat-limits", b_po::value<std::string>(),
          "input (file): file with sizes of individual (not equally sized)"
          " sub-trajectories for concatenated trajectory files. e.g.: for a"
          " concatenated trajectory of three chunks of sizes 100, 50 and 300"
          " frames: '100 50 300'")
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
    case STATS:
      desc.add(desc_stats);
      break;
    case CORING:
      desc.add(desc_coring);
      break;
    case NOISE:
      desc.add(desc_noise);
      break;
    default:
      std::cerr << "error: unknown mode. this should never happen." << std::endl;
      return EXIT_FAILURE;
  }
  try {
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).run(), args);
    if (args["help"].as<bool>()) {
      std::cout << desc << std::endl;
      return EXIT_SUCCESS;
    }
    b_po::notify(args);
  } catch (b_po::error& e) {
    std::cerr << "\nerror parsing arguments:\n\n" << e.what() << "\n\n" << std::endl;
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }
  if (args["help"].as<bool>()) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }
  // activate verbose mode for stats and change
  if (mode == STATS) {
    args.insert(std::make_pair("verbose", b_po::variable_value(true, false)));
  }
  // setup defaults
  if (args.count("verbose")) {
    Clustering::verbose = args["verbose"].as<bool>();
  }
  // print head
  Clustering::logger(std::cout) << "\n" << header_ostring.str() << std::endl;
  if (mode == DENSITY) {
    Clustering::logger(std::cout) << "~~~ using for parallization: ";
#ifdef USE_CUDA
      Clustering::logger(std::cout) << "CUDA" << std::endl;
#else
      Clustering::logger(std::cout) << "cpu" << std::endl;
#endif
  }
  // setup OpenMP
  int n_threads = 0;
  if (args.count("nthreads")) {
    n_threads = args["nthreads"].as<int>();
  }
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  // separate between stats and filter
  if (mode == STATS) {
    args.insert(std::make_pair("list", b_po::variable_value(true, false)));
  } else if (mode == FILTER) {
    args.insert(std::make_pair("list", b_po::variable_value(false, false)));
  }
  // generate header comment
  std::ostringstream header;
  time_t rawtime;
  time(&rawtime);
  struct tm * timeinfo = localtime(&rawtime);
  header << "# clustering " + version_number + " - " << argv[1] << "\n"
         << "#\n"
         << "# Created " << asctime(timeinfo)
         << "# by following command:\n#\n# ";
  std::vector<std::string> arguments(argv, argv + argc);
  for (std::string& arg_string : arguments){
      header << arg_string << " ";
  }
  header << "\n#\n# Copyright (c) 2015-2019 Florian Sittel and Daniel Nagel\n"
         << "# please cite the corresponding paper, "
         << "see https://github.com/moldyn/clustering\n";
  args.insert(std::make_pair("header", b_po::variable_value(header.str(), false)));
  // add parameters which should be read from comments
  std::map<std::string,float> commentsMap = {{"clustering_radius", 0.},
                                             {"lumping_radius", 0.},
                                             {"screening_from", 0.},
                                             {"screening_to", 0.},
                                             {"screening_step", 0.},
                                             {"minimal_population", 0.},
                                             {"cmin", 0.},
                                             {"single_coring_time", 0.}};
  args.insert(std::make_pair("commentsMap", b_po::variable_value(commentsMap, false)));
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
    case STATS:
      Clustering::Filter::main(args);
      break;
    case CORING:
      Clustering::Coring::main(args);
      break;
    case NOISE:
      Clustering::Noise::main(args);
      break;
    default:
      std::cerr << "error: unknown mode. this should never happen." << std::endl;
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

