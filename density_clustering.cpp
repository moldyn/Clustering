
#include "density_clustering.hpp"
#include "tools.hpp"

#include <sstream>
#include <fstream>
#include <iterator>
#include <list>
#include <utility>
#include <functional>
#include <algorithm>
#include <queue>
#include <limits>
#include <numeric>

#include <time.h>

#include <omp.h>
#include <boost/program_options.hpp>

// definitions for offloading
#define ALLOC  alloc_if(1)
#define FREE   free_if(1)
#define RETAIN free_if(0)
#define REUSE  alloc_if(0)

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
calculate_populations(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const float radius) {
  std::vector<std::size_t> pops(n_rows, 1);
  const float rad2 = radius * radius;
  std::size_t i, j, k;
  float dist, c;
  std::cout << "  start: " << __TIMESTAMP__ << std::endl;
  ASSUME_ALIGNED(coords);
  #pragma omp parallel for private(i,j,k,c,dist) \
                           firstprivate(n_rows,n_cols,rad2) \
                           shared(coords,pops) \
                           schedule(dynamic,1024)
  for (i=0; i < n_rows; ++i) {
    for (j=i+1; j < n_rows; ++j) {
      dist = 0.0f;
      #pragma simd reduction(+:dist)
      for (k=0; k < n_cols; ++k) {
        c = coords[i*n_cols+k] - coords[j*n_cols+k];
        dist += c*c;
      }
      if (dist < rad2) {
        #pragma omp atomic
        pops[i] += 1;
        #pragma omp atomic
        pops[j] += 1;
      }
    }
  }
  std::cout << "  finished: " << __TIMESTAMP__ << std::endl;
  return pops;
}


std::vector<float>
calculate_free_energies(const std::vector<std::size_t>& pops) {
  std::size_t i;
  const std::size_t n_frames = pops.size();
  const float max_pop = (float) ( * std::max_element(pops.begin(), pops.end()));
  std::vector<float> free_energy(n_frames);
  float* p_fe = free_energy.data();
  const std::size_t* p_pops = pops.data();
  #pragma omp parallel for private(i) firstprivate(max_pop, n_frames) shared(p_fe, p_pops)
  for (i=0; i < n_frames; ++i) {
    p_fe[i] = (float) -1 * log(p_pops[i]/max_pop);
  }
  return free_energy;
}

std::vector<FreeEnergy>
sorted_free_energies(const std::vector<float>& fe) {
  std::vector<FreeEnergy> fe_sorted;
  for (std::size_t i=0; i < fe.size(); ++i) {
    fe_sorted.push_back(FreeEnergy(i, fe[i]));
  }
  // sort for free energy: lowest to highest (low free energy = high density)
  std::sort(fe_sorted.begin(),
            fe_sorted.end(),
            [] (const FreeEnergy& d1, const FreeEnergy& d2) -> bool {return d1.second < d2.second;});
  return fe_sorted;
}

const std::pair<std::size_t, float>
nearest_neighbor(const float* coords,
                 const std::vector<FreeEnergy>& sorted_free_energies,
                 const std::size_t n_cols,
                 const std::size_t frame_id,
                 const std::pair<std::size_t, std::size_t> search_range) {
  std::size_t c,j;
  const std::size_t real_id = sorted_free_energies[frame_id].first;
  float d, dist;
  std::size_t sr_first = search_range.first;
  std::size_t sr_second = search_range.second;
  std::vector<float> distances(sr_second - sr_first);
  std::vector<std::size_t> sorted_ids;
  for (FreeEnergy fe: sorted_free_energies) {
    sorted_ids.push_back(fe.first);
  }
  ASSUME_ALIGNED(coords);
  #pragma omp parallel for default(shared) private(dist,j,c,d) firstprivate(n_cols,real_id,sr_first)
  for (j=sr_first; j < sr_second; ++j) {
    if (frame_id == j) {
      distances[j-sr_first] = std::numeric_limits<float>::max();
    } else {
      dist = 0.0f;
      #pragma simd reduction(+:dist)
      for (c=0; c < n_cols; ++c) {
        //d = coords[real_id*n_cols+c] - coords[sorted_free_energies[j].first*n_cols+c];
        d = coords[real_id*n_cols+c] - coords[sorted_ids[j]*n_cols+c];
        dist += d*d;
      }
      distances[j-sr_first] = dist;
    }
  }
  if (distances.size() == 0) {
    return {0, 0.0f};
  } else {
    std::size_t min_ndx = std::min_element(distances.begin(), distances.end()) - distances.begin();
    return {min_ndx+sr_first, distances[min_ndx]};
  }
}

Neighborhood
nearest_neighbors(const float* coords,
                  const std::size_t n_rows,
                  const std::size_t n_cols,
                  const std::vector<float>& free_energies,
                  int i_limit=-1) {
  if (i_limit == -1) {
    i_limit = (int) n_rows;
  }
  Neighborhood nh;
  std::vector<FreeEnergy> fe_sorted = sorted_free_energies(free_energies);
  for (int i=0; i < i_limit; ++i) {
    nh[fe_sorted[i].first] = nearest_neighbor(coords, fe_sorted, n_cols, i, SizePair(0,i_limit));
  }
  return nh;
}



Neighborhood
new_nearest_neighbors(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols) {
  Neighborhood nh;
  // initialize neighborhood
  for (std::size_t i=0; i < n_rows; ++i) {
    nh[i] = Neighbor(n_rows+1, std::numeric_limits<float>::max());
  }
  // calculate nearest neighbors with distances
  std::size_t i, j, c, min_j;
  float dist, d, mindist;
  std::cout << "  start: " << __TIMESTAMP__ << std::endl;
  ASSUME_ALIGNED(coords);
  #pragma omp parallel for private(i,j,c,dist,d,mindist,min_j) \
                           firstprivate(n_rows,n_cols) \
                           shared(coords,nh) \
                           schedule(dynamic,1024)
  for (i=0; i < n_rows; ++i) {
    mindist = std::numeric_limits<float>::max();
    min_j = n_rows+1;
    for (j=1; j < n_rows; ++j) {
      if (i != j) {
        dist = 0.0f;
        #pragma simd reduction(+:dist)
        for (c=0; c < n_cols; ++c) {
          d = coords[i*n_cols+c] - coords[j*n_cols+c];
          dist += d*d;
        }
        if (dist < mindist) {
          mindist = dist;
          min_j = j;
        }
      }
    }
    nh[i] = Neighbor(min_j, mindist);
  }
  std::cout << "  finished: " << __TIMESTAMP__ << std::endl;
  return nh;
}


// returns neighborhood set of single frame.
// all ids are sorted in free energy.
std::set<std::size_t>
high_density_neighborhood(const float* coords,
                          const std::size_t n_cols,
                          const std::vector<FreeEnergy>& sorted_fe,
                          const std::size_t i_frame,
                          const std::size_t limit,
                          const float max_dist) {
  std::set<std::size_t> nh;
  std::size_t j,c;
  float d,dist2;
  ASSUME_ALIGNED(coords);
  #pragma omp parallel for default(shared) private(j,c,d,dist2) firstprivate(i_frame,limit,max_dist)
  for (j=0; j < limit; ++j) {
    if (i_frame != j) {
      dist2 = 0.0f;
      for (c=0; c < n_cols; ++c) {
        d = coords[sorted_fe[i_frame].first*n_cols+c] - coords[sorted_fe[j].first*n_cols+c];
        dist2 += d*d;
      }
      if (dist2 < max_dist) {
        #pragma omp critical
        {
          nh.insert(j);
        }
      }
    } else {
      #pragma omp critical
      {
        nh.insert(i_frame);
      }
    }
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
density_clustering(const std::vector<float>& free_energy,
                   const Neighborhood& nh,
                   const float free_energy_threshold,
                   const float* coords,
                   const std::size_t n_rows,
                   const std::size_t n_cols,
                   bool only_initial_frames) {
  std::vector<std::size_t> clustering(n_rows);
  // sort lowest to highest (low free energy = high density)
  std::vector<FreeEnergy> fe_sorted = sorted_free_energies(free_energy);
  // find last frame below free energy threshold
  auto lb = std::upper_bound(fe_sorted.begin(),
                             fe_sorted.end(),
                             FreeEnergy(0, free_energy_threshold), 
                             [](const FreeEnergy& d1, const FreeEnergy& d2) -> bool {return d1.second < d2.second;});
  std::size_t last_frame_below_threshold = (lb - fe_sorted.begin());
  // compute sigma as deviation of nearest-neighbor distances
  // (beware: actually, sigma2 is  E[x^2] > Var(x) = E[x^2] - E[x]^2,
  //  with x being the distances between nearest neighbors)
  double sigma2 = compute_sigma2(nh);
  log(std::cout) << "sigma2: " << sigma2 << std::endl;
  log(std::cout) << last_frame_below_threshold << " frames with low free energy / high density" << std::endl;
  log(std::cout) << "last frame below threshold has free energy: " << fe_sorted[last_frame_below_threshold].second << std::endl;
  // compute a neighborhood with distance 4*sigma2 only on high density frames
  log(std::cout) << "merging initial clusters" << std::endl;
  std::size_t distinct_name = 0;
  bool clusters_merged = false;
  while ( ! clusters_merged) {
    std::set<std::size_t> visited_frames = {};
    clusters_merged = true;
    log(std::cout) << "initial merge iteration" << std::endl;
    for (std::size_t i=0; i < last_frame_below_threshold; ++i) {
      if (visited_frames.count(i) == 0) {
        visited_frames.insert(i);
        // all frames in local neighborhood should be clustered
        std::set<std::size_t> local_nh = high_density_neighborhood(coords,
                                                                   n_cols,
                                                                   fe_sorted,
                                                                   i,
                                                                   last_frame_below_threshold,
                                                                   4*sigma2);
        // let's see if at least some of them already have a
        // designated cluster assignment
        std::set<std::size_t> cluster_names;
        for (auto j: local_nh) {
          cluster_names.insert(clustering[fe_sorted[j].first]);
          visited_frames.insert(j);
        }
        if ( ! (cluster_names.size() == 1 && cluster_names.count(0) != 1)) {
          clusters_merged = false;
          // remove the 'zero' state, i.e. state of unassigned frames
          if (cluster_names.count(0) == 1) {
            cluster_names.erase(0);
          }
          std::size_t common_name;
          if (cluster_names.size() > 0) {
            // indeed, there are already cluster assignments.
            // these should now be merged under a common name.
            // (which will be the id with smallest numerical value,
            //  due to the properties of STL-sets).
            common_name = (*cluster_names.begin());
          } else {
            // no clustering of these frames yet.
            // choose a distinct name.
            common_name = ++distinct_name;
          }
          for (auto j: local_nh) {
            clustering[fe_sorted[j].first] = common_name;
          }

          std::size_t j;
          #pragma omp parallel for private(j)\
                                   firstprivate(common_name,last_frame_below_threshold,cluster_names) \
                                   shared(clustering,fe_sorted)
          for (j=0; j < last_frame_below_threshold; ++j) {
            if (cluster_names.count(clustering[fe_sorted[j].first]) == 1) {
              clustering[fe_sorted[j].first] = common_name;
            }
          }
        }
      }
    }
  }
  // normalize names
  std::set<std::size_t> final_names;
  for (std::size_t i=0; i < last_frame_below_threshold; ++i) {
    final_names.insert(clustering[fe_sorted[i].first]);
  }
  std::map<std::size_t, std::size_t> old_to_new;
  old_to_new[0] = 0;
  std::size_t new_name=0;
  for (auto name: final_names) {
    old_to_new[name] = ++new_name;
  }
  for(auto& elem: clustering) {
    elem = old_to_new[elem];
  }

  // assignment of low-density states
  if ( ! only_initial_frames) {
    log(std::cout) << "assigning remaining frames to " << final_names.size() << " clusters" << std::endl;
//    if ( ! geometry_based_assignment) { // density based assignment
      // assign unassigned frames to clusters via neighbor-info (in descending density order)
    for (std::size_t i=last_frame_below_threshold; i < n_rows; ++i) {
      auto nn = nearest_neighbor(coords, fe_sorted, n_cols, i, SizePair(0,i));
      clustering[fe_sorted[i].first] = clustering[fe_sorted[nn.first].first];
    }
//    } else { // geometry based assignment
//      // always assign the frame next, that has minimal distance to any cluster
//      auto candidate_assigned_id = [](Candidate c) {return std::get<0>(c);};
//      auto candidate_unassigned_id = [](Candidate c) {return std::get<1>(c);};
//      auto candidate_distance = [](Candidate c) {return std::get<2>(c);};
//      auto candidate_comp = [&candidate_distance] (Candidate c1, Candidate c2) -> bool {return candidate_distance(c1) > candidate_distance(c2);};
//      std::priority_queue<Candidate, std::vector<Candidate>, decltype(candidate_comp)> candidates(candidate_comp);
//      // fill candidate queue
//      log(std::cout) << "filling candidate queue" << std::endl;
//      std::size_t i;
//      #pragma omp parallel for default(shared) private(i) firstprivate(n_rows,n_cols) schedule(dynamic)
//      for (i=0; i < n_rows; ++i) {
//        if (clustering[i] != 0) {
//          Candidate c = get_candidate_for_frame(coords, n_rows, n_cols, clustering, i);
//          if (candidate_assigned_id(c) != candidate_unassigned_id(c)) {
//            #pragma omp critical
//            {
//              candidates.push(c);
//            }
//          }
//        }
//      }
//      // assign frames
//      while ( ! candidates.empty()) {
//        Candidate champ = candidates.top();
//        candidates.pop();
//        if (clustering[candidate_unassigned_id(champ)] == 0) {
//          // it's still unassigned, assign to its next cluster
//          clustering[candidate_unassigned_id(champ)] = clustering[candidate_assigned_id(champ)];
//          // push new candidate from freshly assigned frame
//          Candidate rookie = get_candidate_for_frame(coords, n_rows, n_cols, clustering, candidate_unassigned_id(champ));
//          if (candidate_unassigned_id(rookie) != candidate_assigned_id(rookie)) {
//            candidates.push(rookie);
//          }
//        }
//      }
//    }
  }
  log(std::cout) << "clustering finished" << std::endl;
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
    ("threshold,t", b_po::value<float>()->required(), "parameter (required): Free Energy threshold for clustering (FEL is normalized to zero).")
    // optional
    ("population,p", b_po::value<std::string>(), "output (optional): population per frame.")
    ("free-energy,d", b_po::value<std::string>(), "output (optional): free energies per frame.")
    ("free-energy-input,D", b_po::value<std::string>(), "input (optional): reuse free energy info.")
    ("nearest-neighbors,b", b_po::value<std::string>(), "output (optional): nearest neighbor info.")
    ("nearest-neighbors-input,B", b_po::value<std::string>(), "input (optional): reuse nearest neighbor info.")
    // defaults
    ("only-initial,I", b_po::bool_switch()->default_value(false), "only assign initial (i.e. low free energy / high density) frames to clusters.")
    ("geometry-based-assignment,G", b_po::bool_switch()->default_value(false), "use geometry-based assignment of unassigned frames to clusters.")
    ("nthreads,n", b_po::value<int>()->default_value(0), "number of OpenMP threads. default: 0; i.e. use OMP_NUM_THREADS env-variable.")
    ("verbose,v", b_po::bool_switch()->default_value(false), "verbose mode: print runtime information to STDOUT.")
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
  //kmp_set_defaults("KMP_AFFINITY=compact");

  float* coords;
  std::size_t n_rows;
  std::size_t n_cols;
  std::tie(coords, n_rows, n_cols) = read_coords<float>(input_file);

  //// free energies
  std::vector<float> free_energies;
  if (args.count("free-energy-input")) {
    log(std::cout) << "re-using free energy data." << std::endl;
    std::ifstream ifs(args["free-energy-input"].as<std::string>());
    if (ifs.fail()) {
      std::cerr << "error: cannot open file '" << args["free-energy-input"].as<std::string>() << "'" << std::endl;
      return 3;
    } else {
      while(ifs.good()) {
        float buf;
        ifs >> buf;
        free_energies.push_back(buf);
      }
    }
  } else {
    log(std::cout) << "calculating free energies" << std::endl;
    free_energies = calculate_free_energies(
                      calculate_populations(coords, n_rows, n_cols, radius));
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
    //nh = nearest_neighbors(coords, n_rows, n_cols, free_energies);
    nh = new_nearest_neighbors(coords, n_rows, n_cols);
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
  std::vector<std::size_t> clustering = density_clustering(free_energies,
                                                           nh,
                                                           threshold,
                                                           coords,
                                                           n_rows,
                                                           n_cols,
                                                           args["only-initial"].as<bool>());
                                                       //    args["geometry-based-assignment"].as<bool>());
  log(std::cout) << "freeing coords" << std::endl;
  free_coords(coords);
  log(std::cout) << "writing clusters to file " << output_file << std::endl;
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

