
#include "density_clustering.hpp"
#include "tools.hpp"

#include <sstream>
#include <fstream>
#include <iterator>
#include <utility>
#include <functional>
#include <algorithm>
#include <limits>

#include <time.h>

#include <omp.h>
#include <boost/program_options.hpp>

namespace b_po = boost::program_options;

void
log(std::string msg) {
  std::cout << msg << std::endl;
}

void
log(std::string name, float value) {
  std::cout << name << ": " << value << std::endl;
}



std::vector<std::size_t>
calculate_populations(const CoordsPointer<float>& coords_pointer,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const float radius) {

  // provide easy access to coordinates data
  #if defined(__INTEL_COMPILER)
  float* coords = coords_pointer.get();
  __assume_aligned(coords, MEM_ALIGNMENT);
  #else // assume gnu compiler
  float* coords = (float*) __builtin_assume_aligned(coords_pointer.get(), MEM_ALIGNMENT);
  #endif

  std::vector<std::size_t> pops(n_rows, 1);
  const float rad2 = radius * radius;
  std::size_t i, j, k;
  float dist, c;
  #pragma omp parallel for default(shared) private(i,j,k,c,dist) schedule(dynamic)
  for (i=0; i < n_rows; ++i) {
    for (j=i+1; j < n_rows; ++j) {
      dist = 0.0f;
      for (k=0; k < n_cols; ++k) {
        c = coords[i*n_cols+k] - coords[j*n_cols+k];
        dist += c*c;
      }
      if (dist < rad2) {
        pops[i] += 1;
      }
    }
  }
  return pops;
}

std::vector<float>
calculate_densities(const std::vector<std::size_t>& pops) {
  std::size_t i;
  const std::size_t n_frames = pops.size();
  std::vector<float> dens(n_frames);
  float max_pop = (float) ( * std::max_element(pops.begin(), pops.end()));
  #pragma omp parallel for default(shared) private(i)
  for (i=0; i < n_frames; ++i) {
    dens[i] = (float) pops[i] / max_pop;
  }
  return dens;
}


const std::pair<std::size_t, float>
nearest_neighbor(const CoordsPointer<float>& coords_pointer,
                 const std::vector<Density>& sorted_density,
                 std::size_t n_cols,
                 std::size_t frame_id,
                 std::pair<std::size_t, std::size_t> search_range) {

  #if defined(__INTEL_COMPILER)
  float* coords = coords_pointer.get();
  __assume_aligned(coords, MEM_ALIGNMENT);
  #else // assume gnu compiler
  float* coords = (float*) __builtin_assume_aligned(coords_pointer.get(), MEM_ALIGNMENT);
  #endif
  std::size_t c,j;
  float d, dist;
  std::vector<float> distances(search_range.second-search_range.first);

  #pragma omp parallel for default(shared) private(dist,j,c,d) firstprivate(n_cols)
  for (j=search_range.first; j < search_range.second; ++j) {
    dist = 0.0f;
    for (c=0; c < n_cols; ++c) {
      d = coords[sorted_density[frame_id].first*n_cols+c] - coords[sorted_density[j].first*n_cols+c];
      dist += d*d;
    }
    distances[j-search_range.first] = dist;
  }

  if (distances.size() == 0) {
    return {0, 0.0f};
  } else {
    std::size_t min_ndx = std::min_element(distances.begin(), distances.end()) - distances.begin();
    return {min_ndx+search_range.first, distances[min_ndx]};
  }
}


std::vector<std::size_t>
density_clustering(const std::vector<float>& dens,
                   const float density_threshold,
                   const float density_radius,
                   const CoordsPointer<float>& coords_pointer,
                   const std::size_t n_rows,
                   const std::size_t n_cols) {

  std::vector<Density> density_sorted;
  for (std::size_t i=0; i < dens.size(); ++i) {
    density_sorted.push_back({i, dens[i]});
  }

  log("sort densities");

  // sort for density: highest to lowest
  std::sort(density_sorted.begin(),
            density_sorted.end(),
            [] (const Density& d1, const Density& d2) -> bool {return d1.second > d2.second;});

  std::vector<std::size_t> clustering(n_rows);
  std::size_t n_clusters = 0;

  auto lb = std::lower_bound(density_sorted.begin(),
                             density_sorted.end(),
                             Density(0, density_threshold), 
                             [](const Density& d1, const Density& d2) -> bool {return d1.second < d2.second;});

  std::size_t last_frame_below_threshold = (lb - density_sorted.begin());

  log("find initial clusters");

  for (std::size_t i=0; i < last_frame_below_threshold; ++i) {
    if (i % 1000 == 0) {
      std::cout << "   frame: " << i << " / " << last_frame_below_threshold << "\n";
    }
    auto nn_pair = nearest_neighbor(coords_pointer, density_sorted, n_cols, i, {0,i});
    if (nn_pair.second < density_radius) {
      // add to existing cluster of frame with 'min_dist'
      clustering[density_sorted[i].first] = clustering[density_sorted[nn_pair.first].first];
    } else {
      // create new cluster
      ++n_clusters;
      clustering[density_sorted[i].first] = n_clusters;
    }
  }

  log("find nearest neigbors for unassigned frames in clusters");

  // find nearest neighbors for all unassigned frames
  std::map<std::size_t, std::size_t> nearest_neighbors;
  for (std::size_t i=last_frame_below_threshold; i < density_sorted.size(); ++i) {
    if (i % 1000 == 0) {
      std::cout << "   frame: " << i << " / " << density_sorted.size() << "\n";
    }
    auto nn_pair = nearest_neighbor(coords_pointer, density_sorted, n_cols, i, {0, density_sorted.size()});
    nearest_neighbors[i] = nn_pair.first;
  }

  log("assign frames to clusters");

  // assign clusters to unassigned frames via neighbor-info
  bool nothing_happened = true;
  while (nearest_neighbors.size() > 0 && ( ! nothing_happened)) {
    nothing_happened = true;
    for (auto it=nearest_neighbors.begin(); it != nearest_neighbors.end(); ++it) {
      if (clustering[it->second] != 0) {
        clustering[it->first] = it->second;
        nearest_neighbors.erase(it);
        nothing_happened = false;
      }
    }
  }
  return clustering;
}

////////

int main(int argc, char* argv[]) {

  b_po::variables_map args;
  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "perform clustering of MD data based on phase space densities.\n"
    "densities are approximated by counting neighboring frames inside\n"
    "a n-dimensional hypersphere of specified radius.\n"
    "distances are measured with n-dim P2-norm.\n"
    "\n"
    "options"));
  desc.add_options()
    ("help,h", b_po::bool_switch()->default_value(false), "show this help.")
    // required
    ("file,f", b_po::value<std::string>()->required(), "input (required): phase space coordinates (space separated ASCII).")
    ("output,o", b_po::value<std::string>()->required(), "output (required): clustering information.")
    ("radius,r", b_po::value<float>()->required(), "parameter (required): hypersphere radius.")
    ("threshold,t", b_po::value<float>()->required(), "parameter (required): density threshold for clustering.")
    // optional
    ("population,p", b_po::value<std::string>(), "output (optional): population per frame.")
    ("density,d", b_po::value<std::string>(), "output (optional): density per frame.")
    ("density-input,D", b_po::value<std::string>(), "input (optional): reuse density info.")
    // defaults
    ("nthreads,n", b_po::value<int>()->default_value(0), "number of OpenMP threads. default: 0; i.e. use OMP_NUM_THREADS env-variable.")
    ("verbose,v", b_po::bool_switch()->default_value(false), "verbose mode: print max density, runtime information, etc. to STDOUT.")
  ;

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

  ////

  const std::string input_file = args["file"].as<std::string>();
  const std::string output_file = args["output"].as<std::string>();

  const float radius = args["radius"].as<float>();
  const float threshold = args["threshold"].as<float>();

  // setup OpenMP
  const int n_threads = args["nthreads"].as<int>();
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }

  bool verbose = args["verbose"].as<bool>();

  CoordsPointer<float> coords_pointer;
  std::size_t n_rows;
  std::size_t n_cols;
  std::tie(coords_pointer, n_rows, n_cols) = read_coords<float>(input_file);

  std::vector<float> densities;

  if (args.count("density-input")) {
    if (verbose) {
      log("re-using density data.");
    }
    // reuse density info
    std::ifstream ifs(args["density-input"].as<std::string>());
    if (ifs.fail()) {
      std::cerr << "error: cannot open file '" << args["density-input"].as<std::string>() << "'" << std::endl;
      return 3;
    } else {
      while(ifs.good()) {
        float buf;
        ifs >> buf;
        densities.push_back(buf);
      }
    }
  } else {
    if (verbose) {
      log("calculating densities");
    }
    densities = calculate_densities(
                  calculate_populations(coords_pointer, n_rows, n_cols, radius));
    if (args.count("density")) {
      std::ofstream ofs(args["density"].as<std::string>());
      ofs << std::scientific;
      for (float d: densities) {
        ofs << d << "\n";
      }
    }
  }

  if (verbose) {
    log("calculating clusters");
  }
  std::vector<std::size_t> clustering = density_clustering(densities, threshold, radius, coords_pointer, n_rows, n_cols);
  // write clusters to file
  {
    std::ofstream ofs(output_file);
    if (ofs.fail()) {
      std::cerr << "error: cannot open file '" << output_file << "' for writing." << std::endl;
      return 3;
    }
    for (std::size_t i: clustering) {
      ofs << i << "\n";
    }
  }
  return 0;
}

