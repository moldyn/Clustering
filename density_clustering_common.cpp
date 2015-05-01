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

#include "logger.hpp"
#include "density_clustering_common.hpp"

#ifdef DC_USE_MPI
  #include "density_clustering_mpi.hpp"
#endif

namespace Clustering {
namespace Density {

  std::vector<std::size_t>
  initial_density_clustering(const std::vector<float>& free_energy
                           , const Neighborhood& nh
                           , const float free_energy_threshold
                           , const float* coords
                           , const std::size_t n_rows
                           , const std::size_t n_cols
                           , const std::vector<std::size_t> initial_clusters
#ifdef DC_USE_MPI
                           , const int mpi_n_nodes
                           , const int mpi_node_id
#endif
                            ) {
#ifdef DC_USE_MPI
    using namespace Clustering::Density::MPI;
#endif
    std::vector<std::size_t> clustering;
    bool have_initial_clusters = (initial_clusters.size() == n_rows);
    if (have_initial_clusters) {
      clustering = initial_clusters;
    } else {
      clustering = std::vector<std::size_t>(n_rows);
    }
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
#ifdef DC_USE_MPI
    if (mpi_node_id == MAIN_PROCESS) {
#endif
      logger(std::cout) << "sigma2: " << sigma2 << std::endl
                        << first_frame_above_threshold << " frames with low free energy / high density" << std::endl
                        << "first frame above threshold has free energy: "
                        << fe_sorted[first_frame_above_threshold].second << std::endl
                        << "merging initial clusters" << std::endl;
#ifdef DC_USE_MPI
    }
#endif
    // initialize distinct name from initial clustering
    std::size_t distinct_name = *std::max_element(clustering.begin(), clustering.end());
    bool neighboring_clusters_merged = false;
    std::set<std::size_t> visited_frames = {};
    if (have_initial_clusters) {
      // initialize visited_frames from initial clustering
      // (with indices in order of sorted free energies)
      for (std::size_t i=0; i < first_frame_above_threshold; ++i) {
        std::size_t i_original = fe_sorted[i].first;
        if (initial_clusters[i_original] != 0) {
          visited_frames.insert(i);
        }
      }
      logger(std::cout) << "# already visited: " << visited_frames.size() << std::endl;
    }
    // indices inside this loop are in order of sorted(!) free energies
    while ( ! neighboring_clusters_merged) {
      neighboring_clusters_merged = true;
#ifdef DC_USE_MPI
      if (mpi_node_id == MAIN_PROCESS) {
#endif
        logger(std::cout) << "initial merge iteration" << std::endl;
#ifdef DC_USE_MPI
      }
#endif
      for (std::size_t i=0; i < first_frame_above_threshold; ++i) {
        if (visited_frames.count(i) == 0) {
          visited_frames.insert(i);
          // all frames/clusters in local neighborhood should be merged ...
#ifdef DC_USE_MPI
          std::set<std::size_t> local_nh = Clustering::Density::MPI::high_density_neighborhood(coords,
                                                                                               n_cols,
                                                                                               fe_sorted,
                                                                                               i,
                                                                                               first_frame_above_threshold,
                                                                                               4*sigma2,
                                                                                               mpi_n_nodes,
                                                                                               mpi_node_id);
#else
          std::set<std::size_t> local_nh = high_density_neighborhood(coords,
                                                                     n_cols,
                                                                     fe_sorted,
                                                                     i,
                                                                     first_frame_above_threshold,
                                                                     4*sigma2);
#endif
          // ... let's see if at least some of them already have a
          // designated cluster assignment
          std::set<std::size_t> cluster_names;
          for (auto j: local_nh) {
            cluster_names.insert(clustering[fe_sorted[j].first]);
            if ( ! have_initial_clusters) {
              visited_frames.insert(j);
            }
          }
          if ( ! (cluster_names.size() == 1 && cluster_names.count(0) != 1)) {
            neighboring_clusters_merged = false;
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
      } // end for
    } // end while
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

} // end namespace Density
} // end namespace Clustering

