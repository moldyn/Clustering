
#include <iostream>

#include <omp.h>
#include <boost/program_options.hpp>
#include <boost/format.hpp>


#include "project_data.h"
#include "kmeans.h"
#include "mpp.h"

namespace {

namespace b_po = boost::program_options;

int n_microstates_defined(const b_po::variables_map& var_map) {
  int n_microstates = -1;
  if ( ! var_map.count("microstates")) {
    std::cout << "ERROR: please specify number of microstates!" << std::endl;
  } else {
    n_microstates = var_map["microstates"].as<std::size_t>();
  }
  return n_microstates;
}

bool check_microstates_exist(const ProjectData& project, const std::size_t n_microstates) {
  if (project.microstates.count(n_microstates) == 0) {
    std::cerr << "ERROR: no microsates with " << n_microstates << " clusters available.\n"
              << "       Run k-means first!" << std::endl;
    return false;
  }
  return true;
}

} // end local namespace



int main(int argc, char* argv[]) {
  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "calculate geometric/dynamical clustering from time series."
    "\n"
    "options"));

  // running modes
  bool mpp=false;
  bool kmeans=false;
  // information retrieval
  bool get_microstates=false;
  bool get_cluster_centers=false;

  desc.add_options()
    ("help,h", "show this help")
    ("project,p", b_po::value<std::string>()->default_value("clustering.dat"),
                  "project file with clustering information (default: clustering.dat)")
    ("coords,c", b_po::value<std::string>(),
                 "input file with coordinates (whitespace-separated geometric information, one line per timestep).")
    ("mpp", b_po::value(&mpp)->zero_tokens(),
            "run most-probable-path clustering")
    ("kmeans", b_po::value(&kmeans)->zero_tokens(),
               "run k-means clustering")
    ("microstates,m", b_po::value<std::size_t>(),
                      "number of microstates")
    ("lagtime,l", b_po::value<std::size_t>()->default_value(1),
                  "lagtime (default: 1)")
    ("iter", b_po::value<std::size_t>()->default_value(10),
             "kmeans iterations (default: 10)")
    ("nthreads,n", b_po::value<std::size_t>()->default_value(0),
                   "number of OpenMP threads to use. if unset, will use value of $OMP_NUM_THREADS")
    // retrieving the information
    ("get-microstates", b_po::value(&get_microstates)->zero_tokens(),
                        "print microstate trajectory to stdout")
    ("get-cluster-centers", b_po::value(&get_cluster_centers)->zero_tokens(),
                            "print cluster centers to stdout");

  b_po::variables_map var_map;
  try {
    b_po::store(b_po::parse_command_line(argc, argv, desc), var_map);
  } catch (b_po::error e) {
    std::cout << e.what() << std::endl;
  }
  b_po::notify(var_map);

  if (var_map.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }
 
  //// input checks

  ProjectData project;

  if ( ! var_map.count("project")) {
    std::cerr << "ERROR: please specify a project file" << std::endl;
    exit(1);
  } else {
    std::string project_fname = var_map["project"].as<std::string>();
    bool project_file_exists;
    {
      project_file_exists = std::ifstream(project_fname).good();
    }
    if (project_file_exists) {
      project = load_project(project_fname);
    }
  }

  if (var_map.count("coords")) {
    project.coord_file = var_map["coords"].as<std::string>();
  }
  
  //// running calculations

  if (kmeans) {
    // check input
    if (project.coord_file.empty()) {
      std::cerr << "ERROR: please specify a file with coordinates for k-means" << std::endl;
      exit(2);
    }
    if ( ! var_map.count("microstates")) {
      std::cerr << "ERROR: please specify number of microstates for k-means" << std::endl;
      exit(3);
    }
    ////
    run_kmeans(project,
               var_map["microstates"].as<std::size_t>(),
               var_map["iter"].as<std::size_t>(),
               var_map["nthreads"].as<std::size_t>());
  }

  if (mpp) {
    int n_microstates = n_microstates_defined(var_map);
    if ( n_microstates == -1
    ||   ! check_microstates_exist(project, n_microstates)) {
      exit(1);
    }
    run_mpp(project,
            n_microstates,
            var_map["lagtime"].as<std::size_t>(),
            std::vector<std::size_t>(),  // TODO break_points
            var_map["nthreads"].as<std::size_t>());
  }

  //// information retrieval
  if (get_microstates || get_cluster_centers) {
    int n_microstates = n_microstates_defined(var_map);
    if (n_microstates == -1) {
      exit(1);
    }
    if (project.microstates.count(n_microstates) == 0) {
      std::cout << "ERROR: number of microstates defined, but no according data available.\n"
                << "       please run k-means clustering first."
                << std::endl;
      exit(1);
    }
    if (get_microstates) {
      if ( ! check_microstates_exist(project, n_microstates)) {
        exit(1);
      }
      for (std::size_t i=0; i < project.microstates[n_microstates].trajectory.size(); ++i) {
        std::cout << i
                  << " "
                  << project.microstates[n_microstates].trajectory[i]
                  << "\n";
      }
    }
    if (get_cluster_centers) {
      std::size_t n_cols = project.microstates[n_microstates].n_data_cols;
      for (std::size_t i=0; i < (std::size_t) n_microstates; ++i) {
        std::cout << boost::format("%.6f") % project.microstates[n_microstates].centers[i*n_cols];
        for (std::size_t j=1; j < n_cols; ++j) {
          std::cout << " " << boost::format("%.6f") % project.microstates[n_microstates].centers[i*n_cols+j];
        }
        std::cout << "\n";
      }
    }
  }

  save_project(var_map["project"].as<std::string>(), project);
  return 0;
}

