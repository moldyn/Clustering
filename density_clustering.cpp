
#include "tools.hpp"
#include "logger.hpp"
#include "density_clustering.hpp"
#ifdef DC_USE_OPENCL
  #include "density_clustering_opencl.hpp"
#endif

#include <algorithm>

namespace Clustering {
  namespace Density {
    std::vector<std::size_t>
    calculate_populations(const float* coords,
                          const std::size_t n_rows,
                          const std::size_t n_cols,
                          const float radius) {
      std::vector<std::size_t> pops(n_rows, 1);
      const float rad2 = radius * radius;
      std::size_t i, j, k;
      float dist, c;
      ASSUME_ALIGNED(coords);
      #pragma omp parallel for default(none) private(i,j,k,c,dist) \
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
      return pops;
    }
  
    std::vector<float>
    calculate_free_energies(const std::vector<std::size_t>& pops) {
      std::size_t i;
      const std::size_t n_frames = pops.size();
      const float max_pop = (float) ( * std::max_element(pops.begin(), pops.end()));
      std::vector<float> fe(n_frames);
      #pragma omp parallel for default(none) private(i) firstprivate(max_pop, n_frames) shared(fe, pops)
      for (i=0; i < n_frames; ++i) {
        fe[i] = (float) -1 * log(pops[i]/max_pop);
      }
      return fe;
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
  
    std::tuple<Neighborhood, Neighborhood>
    nearest_neighbors(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const std::vector<float>& free_energy) {
      Neighborhood nh;
      Neighborhood nh_high_dens;
      // initialize neighborhood
      for (std::size_t i=0; i < n_rows; ++i) {
        nh[i] = Neighbor(n_rows+1, std::numeric_limits<float>::max());
        nh_high_dens[i] = Neighbor(n_rows+1, std::numeric_limits<float>::max());
      }
      // calculate nearest neighbors with distances
      std::size_t i, j, c, min_j, min_j_high_dens;
      float dist, d, mindist, mindist_high_dens;
      ASSUME_ALIGNED(coords);
      #pragma omp parallel for default(none) \
                               private(i,j,c,dist,d,mindist,mindist_high_dens,min_j,min_j_high_dens) \
                               firstprivate(n_rows,n_cols) \
                               shared(coords,nh,nh_high_dens,free_energy) \
                               schedule(dynamic, 2048)
      for (i=0; i < n_rows; ++i) {
        mindist = std::numeric_limits<float>::max();
        mindist_high_dens = std::numeric_limits<float>::max();
        min_j = n_rows+1;
        min_j_high_dens = n_rows+1;
        for (j=1; j < n_rows; ++j) {
          if (i != j) {
            dist = 0.0f;
            #pragma simd reduction(+:dist)
            for (c=0; c < n_cols; ++c) {
              d = coords[i*n_cols+c] - coords[j*n_cols+c];
              dist += d*d;
            }
            // direct neighbor
            if (dist < mindist) {
              mindist = dist;
              min_j = j;
            }
            // next neighbor with higher density / lower free energy
            if (free_energy[j] < free_energy[i] && dist < mindist_high_dens) {
              mindist_high_dens = dist;
              min_j_high_dens = j;
            }
          }
        }
        nh[i] = Neighbor(min_j, mindist);
        nh_high_dens[i] = Neighbor(min_j_high_dens, mindist_high_dens);
      }
      return std::make_tuple(nh, nh_high_dens);
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
      // buffer to hold information whether frame i is
      // in neighborhood (-> assign 1) or not (-> keep 0)
      std::vector<int> frame_in_nh(limit, 0);
      std::set<std::size_t> nh;
      std::size_t j,c;
      const std::size_t i_frame_sorted = sorted_fe[i_frame].first * n_cols;
      float d,dist2;
      ASSUME_ALIGNED(coords);
      #pragma omp parallel for default(none) private(j,c,d,dist2) \
                               firstprivate(i_frame,i_frame_sorted,limit,max_dist,n_cols) \
                               shared(coords,sorted_fe,frame_in_nh)
      for (j=0; j < limit; ++j) {
        if (i_frame != j) {
          dist2 = 0.0f;
          #pragma simd reduction(+:dist2)
          for (c=0; c < n_cols; ++c) {
            d = coords[i_frame_sorted+c] - coords[sorted_fe[j].first*n_cols+c];
            dist2 += d*d;
          }
          if (dist2 < max_dist) {
            frame_in_nh[j] = 1;
          }
        }
      }
      // reduce buffer data to real neighborhood structure
      for (j=0; j < limit; ++j) {
        if (frame_in_nh[j] > 0) {
          nh.insert(j);
        }
      }
      nh.insert(i_frame);
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
    initial_density_clustering(const std::vector<float>& free_energy,
                               const Neighborhood& nh,
                               const float free_energy_threshold,
                               const float* coords,
                               const std::size_t n_rows,
                               const std::size_t n_cols) {
      std::vector<std::size_t> clustering(n_rows);
      // sort lowest to highest (low free energy = high density)
      std::vector<FreeEnergy> fe_sorted = sorted_free_energies(free_energy);
      // find last frame below free energy threshold
      auto lb = std::upper_bound(fe_sorted.begin(),
                                 fe_sorted.end(),
                                 FreeEnergy(0, free_energy_threshold), 
                                 [](const FreeEnergy& d1, const FreeEnergy& d2) -> bool {return d1.second < d2.second;});
      std::size_t first_frame_above_threshold = (lb - fe_sorted.begin());
      // compute sigma as deviation of nearest-neighbor distances
      // (beware: actually, sigma2 is  E[x^2] > Var(x) = E[x^2] - E[x]^2,
      //  with x being the distances between nearest neighbors).
      // then compute a neighborhood with distance 4*sigma2 only on high density frames.
      double sigma2 = compute_sigma2(nh);
      logger(std::cout) << "sigma2: " << sigma2 << std::endl
                        << first_frame_above_threshold << " frames with low free energy / high density" << std::endl
                        << "first frame above threshold has free energy: "
                        << fe_sorted[first_frame_above_threshold].second << std::endl
                        << "merging initial clusters" << std::endl;
      std::size_t distinct_name = 0;
      bool clusters_merged = false;
      while ( ! clusters_merged) {
        std::set<std::size_t> visited_frames = {};
        clusters_merged = true;
        logger(std::cout) << "initial merge iteration" << std::endl;
        for (std::size_t i=0; i < first_frame_above_threshold; ++i) {
          if (visited_frames.count(i) == 0) {
            visited_frames.insert(i);
            // all frames in local neighborhood should be clustered
            std::set<std::size_t> local_nh = high_density_neighborhood(coords,
                                                                       n_cols,
                                                                       fe_sorted,
                                                                       i,
                                                                       first_frame_above_threshold,
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
              #pragma omp parallel for default(none) private(j)\
                                       firstprivate(common_name,first_frame_above_threshold,cluster_names) \
                                       shared(clustering,fe_sorted)
              for (j=0; j < first_frame_above_threshold; ++j) {
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
      for (std::size_t i=0; i < first_frame_above_threshold; ++i) {
        final_names.insert(clustering[fe_sorted[i].first]);
      }
      std::map<std::size_t, std::size_t> old_to_new;
      old_to_new[0] = 0;
      std::size_t new_name=0;
      for (auto name: final_names) {
        old_to_new[name] = ++new_name;
      }
      // write clustered trajectory
      for(auto& elem: clustering) {
        elem = old_to_new[elem];
      }
      return clustering;
    }
  
    std::vector<std::size_t>
    assign_low_density_frames(const std::vector<std::size_t>& initial_clustering,
                              const Neighborhood& nh_high_dens,
                              const std::vector<float>& free_energy) {
      std::vector<FreeEnergy> fe_sorted = sorted_free_energies(free_energy);
      std::vector<std::size_t> clustering(initial_clustering);
      for (const auto& fe: fe_sorted) {
        std::size_t id = fe.first;
        if (clustering[id] == 0) {
          std::size_t neighbor_id = nh_high_dens.find(id)->second.first;
          // assign cluster of nearest neighbor with higher density
          clustering[id] = clustering[neighbor_id];
        }
      }
      return clustering;
    }
  } // end namespace Density
} // end namespace Clustering

