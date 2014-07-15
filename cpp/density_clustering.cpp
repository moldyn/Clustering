
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

bool verbose = false;

std::ostream& log(std::ostream& s) {
  return s;
}

void
log(std::string msg) {
  log(std::cout) << msg << std::endl;
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
  // and give hint to compiler that it is readily
  // aligned for vectorization
  //
  // DC_MEM_ALIGNMENT is defined during cmake and
  // set depending on usage of SSE2, SSE4_1, AVX or Xeon Phi
  #if defined(__INTEL_COMPILER)
  float* coords = coords_pointer.get();
  __assume_aligned(coords, DC_MEM_ALIGNMENT);
  #else
  // assume gnu compiler
  float* coords = (float*) __builtin_assume_aligned(coords_pointer.get(), DC_MEM_ALIGNMENT);
  #endif

  std::vector<std::size_t> pops(n_rows, 1);
  const float rad2 = radius * radius;
  std::size_t i, j, k;
  float dist, c;
  #pragma omp parallel for default(shared) private(i,j,k,c,dist) firstprivate(n_rows,n_cols,rad2) schedule(dynamic)
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
  const float max_pop = (float) ( * std::max_element(pops.begin(), pops.end()));
  std::vector<float> dens(n_frames);
  #pragma omp parallel for default(shared) private(i) firstprivate(max_pop, n_frames)
  for (i=0; i < n_frames; ++i) {
    dens[i] = (float) pops[i] / max_pop;
  }
  return dens;
}


/*
 * returns the smallest squared distance between two clusters defined
 * by ids 'ndx_cluster1' and 'ndx_cluster2' based on the
 * given coordinate set.
 */
float
cluster_mindist2(const CoordsPointer<float>& coords_pointer,
                 const std::size_t n_cols,
                 const std::vector<std::size_t>& clustering,
                 const std::size_t ndx_cluster1,
                 const std::size_t ndx_cluster2) {
  #if defined(__INTEL_COMPILER)
  float* coords = coords_pointer.get();
  __assume_aligned(coords, DC_MEM_ALIGNMENT);
  #else // assume gnu compiler
  float* coords = (float*) __builtin_assume_aligned(coords_pointer.get(), DC_MEM_ALIGNMENT);
  #endif
  std::size_t i,j,c;
  float dist,d;
  float min_dist = std::numeric_limits<float>::max();
  // select frames of the two clusters
  std::vector<std::size_t> frames1;
  std::vector<std::size_t> frames2;
  for (std::size_t i: clustering) {
    if (i == ndx_cluster1) {
      frames1.push_back(i);
    } else if (i == ndx_cluster2) { 
      frames2.push_back(i);
    }
  }
  const std::size_t n_frames1 = frames1.size();
  const std::size_t n_frames2 = frames2.size();
  //TODO: isn't collapse(2) possible?
  #pragma omp parallel for \
    default(shared) \
    private(i,j,c,d,dist) \
    firstprivate(n_frames1,n_frames2,n_cols) \
    reduction(min: min_dist)
  for (i=0; i < n_frames1; ++i) {
    for (j=0; j < n_frames2; ++j) {
      dist = 0.0f;
      for (c=0; c < n_cols; ++c) {
        d = coords[frames1[i]*n_cols+c] - coords[frames2[j]*n_cols+c];
        dist += d*d;
      }
      if (dist < min_dist) {
        min_dist = dist;
      }
    }
  }
  return min_dist;
}


const std::pair<std::size_t, float>
nearest_neighbor(const CoordsPointer<float>& coords_pointer,
                 const std::vector<Density>& sorted_density,
                 const std::size_t n_cols,
                 const std::size_t frame_id,
                 const std::pair<std::size_t, std::size_t> search_range) {

  #if defined(__INTEL_COMPILER)
  float* coords = coords_pointer.get();
  __assume_aligned(coords, DC_MEM_ALIGNMENT);
  #else // assume gnu compiler
  float* coords = (float*) __builtin_assume_aligned(coords_pointer.get(), DC_MEM_ALIGNMENT);
  #endif
  std::size_t c,j;
  const std::size_t real_id = sorted_density[frame_id].first;
  float d, dist;
  std::vector<float> distances(search_range.second-search_range.first);
  #pragma omp parallel for default(shared) private(dist,j,c,d) firstprivate(n_cols,real_id)
  for (j=search_range.first; j < search_range.second; ++j) {
    if (frame_id == j) {
      distances[j-search_range.first] = std::numeric_limits<float>::max();
    } else {
      dist = 0.0f;
      for (c=0; c < n_cols; ++c) {
        d = coords[real_id*n_cols+c] - coords[sorted_density[j].first*n_cols+c];
        dist += d*d;
      }
      distances[j-search_range.first] = dist;
    }
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
                   const CoordsPointer<float>& coords_pointer,
                   const std::size_t n_rows,
                   const std::size_t n_cols) {

  std::vector<Density> density_sorted;
  for (std::size_t i=0; i < dens.size(); ++i) {
    density_sorted.push_back({i, dens[i]});
  }
  // sort for density: highest to lowest
  std::sort(density_sorted.begin(),
            density_sorted.end(),
            [] (const Density& d1, const Density& d2) -> bool {return d1.second > d2.second;});
  auto lb = std::lower_bound(density_sorted.begin(),
                             density_sorted.end(),
                             Density(0, density_threshold), 
                             [](const Density& d1, const Density& d2) -> bool {return d1.second > d2.second;});
  std::size_t last_frame_below_threshold = (lb - density_sorted.begin());
  // find initial clusters
  std::vector<std::size_t> clustering(n_rows);
  std::map<std::size_t, std::size_t> nearest_neighbors;
  // compute sigma as deviation of nearest-neighbor distances
  // (beware: actually, sigma2 is  E[x^2] > Var(x) = E[x^2] - E[x]^2,
  //  with x being the distances between nearest neighbors)
  double sigma2 = 0.0;
  for (std::size_t i=0; i < n_rows; ++i) {
    auto nn_pair = nearest_neighbor(coords_pointer, density_sorted, n_cols, i, {0,n_rows});
    nearest_neighbors[i] = nn_pair.first;
    sigma2 += nn_pair.second;
  }
  sigma2 /= n_rows;
//TODO remove this test result
//  double sigma2 = 0.0549172;
  // initialize with highest density frame
  std::size_t n_clusters = 1;
  clustering[0] = n_clusters;
  // find frames of same cluster (if geometrically close enough) or add them as new clusters
  for (std::size_t i=1; i < last_frame_below_threshold; ++i) {
    // nn_pair:  first := index in traj,  second := distance to reference (i)
    auto nn_pair = nearest_neighbor(coords_pointer, density_sorted, n_cols, i, {0,i});
    if (nn_pair.second < sigma2) {
      // add to existing cluster
      clustering[density_sorted[i].first] = clustering[density_sorted[nn_pair.first].first];
    } else {
      // create new cluster
      ++n_clusters;
      clustering[density_sorted[i].first] = n_clusters;
    }
  }
  // join clusters if they are close enough to each other
  std::vector<std::set<std::size_t>> cluster_joining;
  for (std::size_t i=1; i <= n_clusters; ++i) {
    for (std::size_t j=1; j < i; ++j) {
      if (cluster_mindist2(coords_pointer, n_cols, clustering, i, j) < sigma2) {
        // both clusters have each (at least one) element(s) that are
        // closer than sigma2 to an element of the other cluster.
        // -> join them!
        for (auto& join: cluster_joining) {
          if (join.count(i) != 0) {
            // already joining-info on cluster i: add cluster j to this set
            join.insert(j);
            break;
          }
        }
        // cluster i has no joining-info yet: create new set
        cluster_joining.push_back({i,j});
      }
    }
  }
  std::map<std::size_t, std::size_t> old_to_new_names;
  for (std::size_t new_name=1; new_name <= cluster_joining.size(); ++new_name) {
    for (std::size_t old_name: cluster_joining[new_name]) {
      old_to_new_names[old_name] = new_name;
    }
  }
  old_to_new_names[0] = 0;
  for (std::size_t i=0; i < n_rows; ++i) {
    clustering[i] = old_to_new_names[clustering[i]];
  }
  // assign unassigned frames to clusters via neighbor-info
  bool nothing_happened = true;
  while (nearest_neighbors.size() > 0 && ( ! nothing_happened)) {
    nothing_happened = true;
    // it:  first := index in traj,  second := index of nearest neighbor (i)
    for (auto it=nearest_neighbors.begin(); it != nearest_neighbors.end(); ++it) {
      if (clustering[it->second] != 0) {
        clustering[it->first] = clustering[it->second];
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
    ("threshold,t", b_po::value<float>()->required(), "parameter (required, elem. of [0.0, 1.0]): density threshold for clustering.")
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
  std::vector<std::size_t> clustering = density_clustering(densities, threshold, coords_pointer, n_rows, n_cols);
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

