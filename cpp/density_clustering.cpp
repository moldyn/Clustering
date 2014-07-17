
#include "density_clustering.hpp"
#include "tools.hpp"

#include <sstream>
#include <fstream>
#include <iterator>
#include <list>
#include <utility>
#include <functional>
#include <algorithm>
#include <limits>
#include <numeric>

#include <time.h>

#include <omp.h>
#include <boost/program_options.hpp>

namespace b_po = boost::program_options;

namespace {

bool verbose = false;
std::ostream devnull(0);
std::ostream& log(std::ostream& s) {
  if (verbose) {
    return s;
  } else {
    return devnull; 
  }
}

} // end local namespace


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
  std::size_t sr_first = search_range.first;
  std::size_t sr_second = search_range.second;
  std::vector<float> distances(sr_second - sr_first);
  #pragma omp parallel for default(shared) private(dist,j,c,d) firstprivate(n_cols,real_id,sr_first)
  for (j=sr_first; j < sr_second; ++j) {
    if (frame_id == j) {
      distances[j-sr_first] = std::numeric_limits<float>::max();
    } else {
      dist = 0.0f;
      for (c=0; c < n_cols; ++c) {
        d = coords[real_id*n_cols+c] - coords[sorted_density[j].first*n_cols+c];
        dist += d*d;
      }
      distances[j-sr_first] = dist;
    }
  }
  for (std::size_t i=0; i < (sr_second-sr_first); ++i) {
    distances[i] = distances[i];
  }
  if (distances.size() == 0) {
    return {0, 0.0f};
  } else {
    std::size_t min_ndx = std::min_element(distances.begin(), distances.end()) - distances.begin();
    return {min_ndx+sr_first, distances[min_ndx]};
  }
}

std::vector<Density>
sorted_densities(const std::vector<float>& dens) {
  std::vector<Density> density_sorted;
  for (std::size_t i=0; i < dens.size(); ++i) {
    density_sorted.push_back(Density(i, dens[i]));
  }
  // sort for density: highest to lowest
  std::sort(density_sorted.begin(),
            density_sorted.end(),
            [] (const Density& d1, const Density& d2) -> bool {return d1.second > d2.second;});
  return density_sorted;
}

// returns vector of neighborhood sets.
// all ids (vector-id and set-ids) are
// in sorted density order.
std::vector<std::set<std::size_t>>
high_density_neighborhood(const CoordsPointer<float>& coords_pointer,
                          const std::size_t n_cols,
                          const std::vector<Density>& sorted_density,
                          const std::size_t limit,
                          const float max_dist) {
  #if defined(__INTEL_COMPILER)
  float* coords = coords_pointer.get();
  __assume_aligned(coords, DC_MEM_ALIGNMENT);
  #else // assume gnu compiler
  float* coords = (float*) __builtin_assume_aligned(coords_pointer.get(), DC_MEM_ALIGNMENT);
  #endif
  std::vector<std::set<std::size_t>> nh(limit);
  std::size_t i,j,c;
  float d,dist2;
  #pragma omp parallel for default(shared) private(i,j,c,d,dist2) firstprivate(limit,max_dist)
  for (i=0; i < limit; ++i) {
    for (j=0; j < limit; ++j) {
      if (i != j) {
        dist2 = 0.0f;
        for (c=0; c < n_cols; ++c) {
          d = coords[sorted_density[i].first*n_cols+c] - coords[sorted_density[j].first*n_cols+c];
          dist2 += d*d;
        }
        if (dist2 < max_dist) {
          nh[i].insert(j);
        }
      } else {
        nh[i].insert(i);
      }
    }
  }
  return nh;
}


Neighborhood
nearest_neighbors(const CoordsPointer<float>& coords_pointer,
                  const std::size_t n_rows,
                  const std::size_t n_cols,
                  const std::vector<float>& densities,
                  int i_limit=-1) {
  if (i_limit == -1) {
    i_limit = (int) n_rows;
  }
  Neighborhood nh;
  std::vector<Density> densities_sorted = sorted_densities(densities);
  for (int i=0; i < i_limit; ++i) {
    nh[densities_sorted[i].first] = nearest_neighbor(coords_pointer, densities_sorted, n_cols, i, SizePair(0,i_limit));
  }
  return nh;
}

double
compute_sigma2(const Neighborhood& nh) {
  double sigma2 = 0.0;
  for (auto match: nh) {
    // first second: nearest neighbor info
    // second second: squared dist to nearest neighbor
    sigma2 += match.second.second;
  }
  return (sigma2 / nh.size());
}


std::vector<std::size_t>
density_clustering(const std::vector<float>& dens,
                   Neighborhood nh,
                   const float density_threshold,
                   const CoordsPointer<float>& coords_pointer,
                   const std::size_t n_rows,
                   const std::size_t n_cols) {

  std::vector<Density> density_sorted = sorted_densities(dens);
  auto lb = std::lower_bound(density_sorted.begin(),
                             density_sorted.end(),
                             Density(0, density_threshold), 
                             [](const Density& d1, const Density& d2) -> bool {return d1.second > d2.second;});
  std::size_t last_frame_below_threshold = (lb - density_sorted.begin());
  // find initial clusters
  std::vector<std::size_t> clustering(n_rows);
  // compute sigma as deviation of nearest-neighbor distances
  // (beware: actually, sigma2 is  E[x^2] > Var(x) = E[x^2] - E[x]^2,
  //  with x being the distances between nearest neighbors)
  double sigma2 = compute_sigma2(nh);
  log(std::cout) << "sigma2: " << sigma2 << std::endl;
  log(std::cout) << last_frame_below_threshold << " frames with high density" << std::endl;
  // compute a neighborhood with distance 4*sigma2 only on high density frames
  std::vector<std::set<std::size_t>> high_dens_nh = high_density_neighborhood(coords_pointer,
                                                                              n_cols,
                                                                              density_sorted,
                                                                              last_frame_below_threshold,
                                                                              4*sigma2);
  std::list<std::set<std::size_t>> clusters;
  for (auto cluster: high_dens_nh) {
    clusters.push_back(cluster);
  }
  log(std::cout) << "merging initial clusters" << std::endl;
  bool there_was_a_merge = true;
  while (there_was_a_merge) {
    there_was_a_merge = false;
    for (auto it_i=clusters.begin(); it_i != clusters.end(); ++it_i) {
      for (auto it_j=clusters.begin(); it_j != it_i; ++it_j) {
        bool disjoint = true;
        // check if clusters are disjoint or not
        for (std::size_t elem_i: *it_i) {
          for (std::size_t elem_j: *it_j) {
            if (elem_i == elem_j) {
              // (at least) two elements are equal: not disjoint
              disjoint = false;
              break;
            }
          }
          if ( ! disjoint) {
            break;
          }
        }
        if ( ! disjoint) {
          // not disjoint: merge!
          std::set<std::size_t> new_cluster = *it_i;
          clusters.erase(it_i);
          for (std::size_t elem: *it_j) {
            new_cluster.insert(elem);
          }
          clusters.erase(it_j);
          clusters.push_back(new_cluster);
          there_was_a_merge = true;
          break; // ... the j-loop
        }
      }
      if (there_was_a_merge) {
        break;  // ... the i-loop
      }
    }
  }
  log(std::cout) << "assigning cluster-ids to high density frames" << std::endl;
  std::size_t cluster_name = 0;
  for (auto cluster: clusters) {
    for (auto i_frame: cluster) {
      clustering[density_sorted[i_frame].first] = cluster_name+1;
    }
    ++cluster_name;
  }


//  for (std::size_t i=1; i <= n_clusters; ++i) {
//    std::set<std::size_t> temp_set;
//    temp_set.insert(i);
//    cluster_joining.push_back(temp_set);
//  }
//  bool join_happened = true;
//  while (join_happened) {
//    join_happened = false;
//    for (std::size_t i=0; i < cluster_joining.size(); ++i) {
//      for (std::size_t j=0; j < i; ++j) {
//        std::set<std::size_t> set1 = cluster_joining[i];
//        std::set<std::size_t> set2 = cluster_joining[j];
//        if (cluster_set_joinable(coords_pointer, n_cols, clustering, set1, set2, 4*sigma2)) {
//          log(std::cout) << "join happened, #clusters left: " << cluster_joining.size()-1 << std::endl;
//          join_happened = true;
//          // join sets
//          set1.insert(set2.begin(), set2.end());
//          // delete old sets (highest index always first!)
//          cluster_joining.erase(cluster_joining.begin() + i);
//          cluster_joining.erase(cluster_joining.begin() + j);
//          // add new (joined) set
//          cluster_joining.push_back(set1);
//          break;
//        }
//      }
//      // start from beginning, since 'cluster_joining' is now
//      // a different data structure
//      if (join_happened) {
//        break;
//      }
//    }
//  }
//  log(std::cout) << "joining clusters to " << cluster_joining.size() << " new clusters" << std::endl;
//  std::map<std::size_t, std::size_t> old_to_new_names;
//  for (std::size_t new_name=0; new_name < cluster_joining.size(); ++new_name) {
//    for (std::size_t old_name: cluster_joining[new_name]) {
//      // let new names begin with 1, 2, ... to keep 0 for non-assigned frames
//      old_to_new_names[old_name] = new_name+1;
//    }
//  }
//  old_to_new_names[0] = 0;
//  for (std::size_t i=0; i < n_rows; ++i) {
//    clustering[i] = old_to_new_names[clustering[i]];
//  }



//TODO: must be done in descending density order

  // assign unassigned frames to clusters via neighbor-info
//  bool nothing_happened = false;
//  while (nh.size() > 0 && ( ! nothing_happened)) {
//    nothing_happened = true;
//    // it: first := index of frame, second := neighbor pair(index, dist)
//    for (auto it=nh.begin(); it != nh.end(); ++it) {
//      if (clustering[it->first] == 0 && clustering[it->second.first] != 0) {
//        // frame itself is unassigned, while neighbor is assigned
//        //  -> assign to neighbor's cluster
//        clustering[it->first] = clustering[it->second.first];
//        nh.erase(it);
//        nothing_happened = false;
//      }
//    }
//  }
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
    ("nearest-neighbors,b", b_po::value<std::string>(), "output (optional): nearest neighbor info.")
    ("nearest-neighbors-input,B", b_po::value<std::string>(), "input (optional): reuse nearest neighbor info.")
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
  verbose = args["verbose"].as<bool>();

  const std::string input_file = args["file"].as<std::string>();
  const std::string output_file = args["output"].as<std::string>();

  const float radius = args["radius"].as<float>();
  const float threshold = args["threshold"].as<float>();

  // setup OpenMP
  const int n_threads = args["nthreads"].as<int>();
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }

  CoordsPointer<float> coords_pointer;
  std::size_t n_rows;
  std::size_t n_cols;
  std::tie(coords_pointer, n_rows, n_cols) = read_coords<float>(input_file);

  //// densities
  std::vector<float> densities;
  if (args.count("density-input")) {
    log(std::cout) << "re-using density data." << std::endl;
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
    log(std::cout) << "calculating densities" << std::endl;
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
  //// nearest neighbors
  Neighborhood nh;
  if (args.count("nearest-neighbors-input")) {
    log(std::cout) << "re-using nearest neighbor data." << std::endl;
    
    std::ifstream ifs(args["nearest-neighbors-input"].as<std::string>());
    if (ifs.fail()) {
      std::cerr << "error: cannot open file '" << args["nearest-neighbors-input"].as<std::string>() << "'" << std::endl;
      return 3;
    } else {
      std::size_t i=0;
      while (ifs.good()) {
        std::size_t buf1;
        float buf2;
        ifs >> buf1;
        ifs >> buf2;
        nh[i] = std::pair<std::size_t, float>(buf1, buf2);
        ++i;
      }
    }
  } else {
    log(std::cout) << "calculating nearest neighbors" << std::endl;
    nh = nearest_neighbors(coords_pointer, n_rows, n_cols, densities);
    if (args.count("nearest-neighbors")) {
      std::ofstream ofs(args["nearest-neighbors"].as<std::string>());
      for (auto p: nh) {
        // second: neighbor
        // second.first: id; second.second: squared dist
        ofs << p.second.first << " " << p.second.second << "\n";
      }
    }
  }
  //// clustering
  log(std::cout) << "calculating clusters" << std::endl;
  std::vector<std::size_t> clustering = density_clustering(densities, nh, threshold, coords_pointer, n_rows, n_cols);
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

