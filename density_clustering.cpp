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

#include "tools.hpp"
#include "logger.hpp"
#include "density_clustering.hpp"

#ifdef USE_CUDA
  #include "density_clustering_cuda.hpp"
#else
  #include "density_clustering_common.hpp"
#endif

#include <algorithm>

namespace Clustering {
  namespace Density {
    BoxGrid
    compute_box_grid(const float* coords,
                     const std::size_t n_rows,
                     const std::size_t n_cols,
                     const float radius) {
      // use first and second coordinates, since these usually
      // correspond to first and second PCs, having highest variance.
      // if clustering is only in 1 dimension, boxes of higher dimensions
      // will be kept empty.
      const int BOX_DIM_1 = 0;
      const int BOX_DIM_2 = 1;
      BoxGrid grid;
      ASSUME_ALIGNED(coords);
      // find min/max values for first and second dimension
      float min_x1=0.0f, max_x1=0.0f, min_x2=0.0f, max_x2=0.0f;
      min_x1=coords[0*n_cols+BOX_DIM_1];
      max_x1=coords[0*n_cols+BOX_DIM_1];
      if (n_cols > 1) {
        min_x2=coords[0*n_cols+BOX_DIM_2];
        max_x2=coords[0*n_cols+BOX_DIM_2];
      }
      Clustering::logger(std::cout) << "setting up boxes for fast NN search" << std::endl;
      for (std::size_t i=1; i < n_rows; ++i) {
        min_x1 = std::min(min_x1, coords[i*n_cols+BOX_DIM_1]);
        max_x1 = std::max(max_x1, coords[i*n_cols+BOX_DIM_1]);
        if (n_cols > 1) {
          min_x2 = std::min(min_x2, coords[i*n_cols+BOX_DIM_2]);
          max_x2 = std::max(max_x2, coords[i*n_cols+BOX_DIM_2]);
        }
      }
      // build 2D grid with boxes for efficient nearest neighbor search
      grid.n_boxes.push_back((max_x1 - min_x1) / radius + 1);
      if (n_cols > 1) {
        grid.n_boxes.push_back((max_x2 - min_x2) / radius + 1);
      } else {
        grid.n_boxes.push_back(1);
      }
      grid.assigned_box.resize(n_rows);
      for (std::size_t i=0; i < n_rows; ++i) {
        int i_box_1=0, i_box_2=0;
        i_box_1 = (coords[i*n_cols+BOX_DIM_1] - min_x1) / radius;
        if (n_cols > 1) {
          i_box_2 = (coords[i*n_cols+BOX_DIM_2] - min_x2) / radius;
        }
        grid.assigned_box[i] = {i_box_1, i_box_2};
        grid.boxes[grid.assigned_box[i]].push_back(i);
      }
      return grid;
    }

    constexpr Box
    neighbor_box(const Box center, const int i_neighbor) {
      return {center[0] + BOX_DIFF[i_neighbor][0]
            , center[1] + BOX_DIFF[i_neighbor][1]};
    }

    bool
    is_valid_box(const Box box, const BoxGrid& grid) {
      int i1 = box[0];
      int i2 = box[1];
      return ((i1 >= 0)
           && (i1 < grid.n_boxes[0])
           && (i2 >= 0)
           && (i2 < grid.n_boxes[1]));
    }

    std::vector<std::size_t>
    calculate_populations(const float* coords,
                          const std::size_t n_rows,
                          const std::size_t n_cols,
                          const float radius) {
      std::vector<float> radii = {radius};
#ifdef USE_CUDA
      std::map<float, std::vector<std::size_t>> pop_map =
        Clustering::Density::CUDA::calculate_populations(coords
                                                       , n_rows
                                                       , n_cols
                                                       , radii);
#else
      std::map<float, std::vector<std::size_t>> pop_map =
        calculate_populations(coords, n_rows, n_cols, radii);
#endif
      return pop_map[radius];
    }

    std::map<float, std::vector<std::size_t>>
    calculate_populations(const float* coords,
                          const std::size_t n_rows,
                          const std::size_t n_cols,
                          std::vector<float> radii) {
      std::map<float, std::vector<std::size_t>> pops;
      for (float rad: radii) {
        pops[rad].resize(n_rows, 1);
      }
      std::sort(radii.begin(), radii.end(), std::greater<float>());
      std::size_t n_radii = radii.size();
      std::vector<float> rad2(n_radii);
      for (std::size_t i=0; i < n_radii; ++i) {
        rad2[i] = radii[i]*radii[i];
      }
      ASSUME_ALIGNED(coords);
      std::size_t i, j, k, l, ib;
      BoxGrid grid = compute_box_grid(coords, n_rows, n_cols, radii[0]);
      Clustering::logger(std::cout) << " box grid: "
                                    << grid.n_boxes[0]
                                    << " x "
                                    << grid.n_boxes[1]
                                    << std::endl;
      Clustering::logger(std::cout) << "computing pops" << std::endl;
      float dist, c;
      Box box;
      Box center;
      int i_neighbor;
      std::vector<int> box_buffer;
      #pragma omp parallel for default(none)\
        private(i,box,box_buffer,center,i_neighbor,ib,dist,j,k,l,c)\
        firstprivate(n_rows,n_cols,n_radii,radii,rad2,N_NEIGHBOR_BOXES)\
        shared(coords,pops,grid)\
        schedule(dynamic,1024)
      for (i=0; i < n_rows; ++i) {
        center = grid.assigned_box[i];
        // loop over surrounding boxes to find neighbor candidates
        for (i_neighbor=0; i_neighbor < N_NEIGHBOR_BOXES; ++i_neighbor) {
          box = neighbor_box(center, i_neighbor);
          if (is_valid_box(box, grid)) {
            box_buffer = grid.boxes[box];
            // loop over frames inside surrounding box
            for (ib=0; ib < box_buffer.size(); ++ib) {
              j = box_buffer[ib];
              if (i < j) {
                dist = 0.0f;
                #pragma simd reduction(+:dist)
                for (k=0; k < n_cols; ++k) {
                  c = coords[i*n_cols+k] - coords[j*n_cols+k];
                  dist += c*c;
                }
                for (l=0; l < n_radii; ++l) {
                  if (dist < rad2[l]) {
                    #pragma omp atomic
                    pops[radii[l]][i] += 1;
                    #pragma omp atomic
                    pops[radii[l]][j] += 1;
                  } else {
                    // if it's not in the bigger radius,
                    // it won't be in the smaller ones.
                    break;
                  }
                }
              }
            }
          }
        }
      }
      return pops;
    }
  
    std::vector<float>
    calculate_free_energies(const std::vector<std::size_t>& pops) {
      std::size_t i;
      const std::size_t n_frames = pops.size();
      const float max_pop = (float) (*std::max_element(pops.begin()
                                                     , pops.end()));
      std::vector<float> fe(n_frames);
      #pragma omp parallel for default(none)\
                               private(i)\
                               firstprivate(max_pop, n_frames)\
                               shared(fe, pops)
      for (i=0; i < n_frames; ++i) {
        fe[i] = (float) -1.0f * log(pops[i]/max_pop);
      }
      return fe;
    }
  
    std::vector<FreeEnergy>
    sorted_free_energies(const std::vector<float>& fe) {
      std::vector<FreeEnergy> fe_sorted;
      for (std::size_t i=0; i < fe.size(); ++i) {
        fe_sorted.push_back(FreeEnergy(i, fe[i]));
      }
      // sort for free energy: lowest to highest
      // (low free energy = high density)
      std::sort(fe_sorted.begin(),
                fe_sorted.end(),
                [] (const FreeEnergy& d1, const FreeEnergy& d2) -> bool {
                  return d1.second < d2.second;
                });
      return fe_sorted;
    }
 
    std::tuple<Neighborhood, Neighborhood>
    nearest_neighbors(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const std::vector<float>& free_energy) {
//TODO: there is a small error somewhere that misclassifies frames as
//      nearest neighbors. compare to results with CUDA-driven code
//      (whose output was manually checked for correctness)
      Neighborhood nh;
      Neighborhood nh_high_dens;
      // initialize neighborhood
      for (std::size_t i=0; i < n_rows; ++i) {
        nh[i] = Neighbor(n_rows+1
                       , std::numeric_limits<float>::max());
        nh_high_dens[i] = Neighbor(n_rows+1
                                 , std::numeric_limits<float>::max());
      }
      // calculate nearest neighbors with distances
      std::size_t i, j, c, min_j, min_j_high_dens;
      float dist, d, mindist, mindist_high_dens;
      ASSUME_ALIGNED(coords);
      #pragma omp parallel for default(none) \
        private(i,j,c,dist,d,mindist,mindist_high_dens,min_j,min_j_high_dens)\
        firstprivate(n_rows,n_cols) \
        shared(coords,nh,nh_high_dens,free_energy) \
        schedule(dynamic, 2048)
      for (i=0; i < n_rows; ++i) {
        mindist = std::numeric_limits<float>::max();
        mindist_high_dens = std::numeric_limits<float>::max();
        min_j = n_rows+1;
        min_j_high_dens = n_rows+1;
        for (j=0; j < n_rows; ++j) {
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
            if (free_energy[j] < free_energy[i]
             && dist < mindist_high_dens) {
              mindist_high_dens = dist;
              min_j_high_dens = j;
            }
          }
        }
        nh[i] = Neighbor(min_j
                       , mindist);
        nh_high_dens[i] = Neighbor(min_j_high_dens
                                 , mindist_high_dens);
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
      #pragma omp parallel for default(none)\
        private(j,c,d,dist2)\
        firstprivate(i_frame,i_frame_sorted,limit,max_dist,n_cols)\
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
        // 'first second': nearest neighbor info
        // 'second second': squared dist to nearest neighbor
        sigma2 += match.second.second;
      }
      return (sigma2 / nh.size());
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

    void
    screening_log(const double sigma2
                , const std::size_t first_frame_above_threshold
                , const std::vector<FreeEnergy>& fe_sorted) {
      logger(std::cout) << "sigma2: "
                        << sigma2
                        << std::endl
                        << first_frame_above_threshold
                        << " frames with low free energy / high density"
                        << std::endl
                        << "first frame above threshold has free energy: "
                        << fe_sorted[first_frame_above_threshold].second
                        << std::endl
                        << "merging initial clusters"
                        << std::endl
                        << std::endl
      ;
    }


    std::tuple<std::vector<std::size_t>
             , std::size_t
             , double
             , std::vector<FreeEnergy>
             , std::set<std::size_t>
             , std::size_t>
    prepare_initial_clustering(const std::vector<float>& free_energy
                             , const Neighborhood& nh
                             , const float free_energy_threshold
                             , const std::size_t n_rows
                             , const std::vector<std::size_t> initial_clusters) {
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
      auto lb = std::upper_bound(fe_sorted.begin()
                               , fe_sorted.end()
                               , FreeEnergy(0, free_energy_threshold)
                               , [](const FreeEnergy& d1
                                  , const FreeEnergy& d2) -> bool {
                                   return d1.second < d2.second;
                                 });
      std::size_t first_frame_above_threshold = (lb - fe_sorted.begin());
      // compute sigma as deviation of nearest-neighbor distances
      // (beware: actually, sigma2 is  E[x^2] > Var(x) = E[x^2] - E[x]^2,
      //  with x being the distances between nearest neighbors).
      // then compute a neighborhood with distance 4*sigma2 only on high density frames.
      double sigma2 = compute_sigma2(nh);
      // initialize distinct name from initial clustering
      std::size_t distinct_name = *std::max_element(clustering.begin(), clustering.end());
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
      }
      return std::make_tuple(clustering
                           , first_frame_above_threshold
                           , sigma2
                           , fe_sorted
                           , visited_frames
                           , distinct_name);
    }

    std::vector<std::size_t>
    normalized_cluster_names(std::size_t first_frame_above_threshold
                           , std::vector<std::size_t> clustering
                           , std::vector<FreeEnergy>& fe_sorted) {
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
      // rewrite clustered trajectory with new names
      for(auto& elem: clustering) {
        elem = old_to_new[elem];
      }
      return clustering;
    }

    std::vector<std::size_t>
    sorted_cluster_names(std::vector<std::size_t> clustering) {

      std::size_t n_frames = clustering.size();

      // generate counting maps
      typedef std::map<std::size_t,std::size_t> CounterClustMap;
      CounterClustMap counts;
      for (std::size_t i=0; i < n_frames; ++i) {
        CounterClustMap::iterator it(counts.find(clustering[i]));
        if (it != counts.end()){
          it->second++;
        } else {
          counts[clustering[i]] = 1;
        }
      }

      // convert to vector for sorting
      std::vector<std::pair<std::size_t,std::size_t>> counts_vec;
      std::copy(counts.begin(), counts.end(), std::back_inserter(counts_vec));

      std::sort(counts_vec.begin(), counts_vec.end(), compare2DVector);

      // generate cluster name conversion map
      CounterClustMap MapNames;
      for(std::size_t i = 0; i < counts_vec.size(); ++i) {
        MapNames[counts_vec[i].first] = counts_vec.size() - i;
      }

      // remap cluster names
      std::vector<std::size_t> remapped_clustering(n_frames);
      for (std::size_t i = 0; i < n_frames; ++i) {
        remapped_clustering[i] = MapNames[clustering[i]];
      }
      return remapped_clustering;
    }

    bool
    compare2DVector(const std::pair<std::size_t,std::size_t>  &p1, const std::pair<std::size_t,std::size_t> &p2) {
      return p1.second < p2.second;
    }

    bool
    lump_initial_clusters(const std::set<std::size_t>& local_nh
                        , std::size_t& distinct_name
                        , std::vector<std::size_t>& clustering
                        , const std::vector<FreeEnergy>& fe_sorted
                        , std::size_t first_frame_above_threshold) {
      bool neighboring_clusters_merged = true;
      // ... let's see if at least some of them already have a
      // designated cluster assignment
      std::set<std::size_t> cluster_names;
      for (auto j: local_nh) {
        cluster_names.insert(clustering[fe_sorted[j].first]);
      }
      if ( ! (cluster_names.size() == 1
           && cluster_names.count(0) != 1)) {
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
        std::size_t j,ndx;
        #pragma omp parallel for default(none) private(j,ndx)\
                                 firstprivate(common_name,\
                                              first_frame_above_threshold,\
                                              cluster_names)\
                                 shared(clustering,fe_sorted)
        for (j=0; j < first_frame_above_threshold; ++j) {
          ndx = fe_sorted[j].first;
          if (cluster_names.count(clustering[ndx]) == 1) {
            clustering[ndx] = common_name;
          }
        }
      }
      return neighboring_clusters_merged;
    }


#ifndef DC_USE_MPI
    void
    main(boost::program_options::variables_map args) {

//TODO check if files have been read correctly

      using namespace Clustering::Tools;
      const std::string input_file = args["file"].as<std::string>();
      // setup coords
      float* coords;
      std::size_t n_rows;
      std::size_t n_cols;
      Clustering::logger(std::cout) << "reading coords" << std::endl;
      std::tie(coords, n_rows, n_cols) = read_coords<float>(input_file);
      //// free energies
      std::vector<float> free_energies;
      if (args.count("free-energy-input")) {
        Clustering::logger(std::cout) << "re-using free energy data." << std::endl;
        free_energies = read_free_energies(args["free-energy-input"].as<std::string>());
      } else if (args.count("free-energy") || args.count("population") || args.count("output")) {
        if (args.count("radii")) {
          // compute populations & free energies for different radii in one go
          if (args.count("output")) {
            std::cerr << "error: clustering cannot be done with several radii (-R is set)." << std::endl;
            exit(EXIT_FAILURE);
          }
          if ( ! (args.count("population") || args.count("free-energy"))) {
            std::cerr << "error: no output defined for populations or free energies. why did you define -R ?" << std::endl;
            exit(EXIT_FAILURE);
          }
          std::vector<float> radii = args["radii"].as<std::vector<float>>();
#ifdef USE_CUDA
          Pops pops = Clustering::Density::CUDA::calculate_populations(coords
                                                                     , n_rows
                                                                     , n_cols
                                                                     , radii);
#else
          Pops pops = calculate_populations(coords
                                          , n_rows
                                          , n_cols
                                          , radii);
#endif
          for (auto radius_pops: pops) {
            if (args.count("population")) {
              std::string basename_pop = args["population"].as<std::string>() + "_%f";
              write_pops(Clustering::Tools::stringprintf(basename_pop, radius_pops.first), radius_pops.second);
            }
            if (args.count("free-energy")) {
              std::string basename_fe = args["free-energy"].as<std::string>() + "_%f";
              write_fes(Clustering::Tools::stringprintf(basename_fe, radius_pops.first), calculate_free_energies(radius_pops.second));
            }
          }
        } else {
          if ( ! args.count("radius")) {
            std::cerr << "error: radius (-r) is required!" << std::endl;
          }
          const float radius = args["radius"].as<float>();
          // compute populations & free energies for clustering and/or saving
          Clustering::logger(std::cout) << "calculating populations" << std::endl;
          std::vector<std::size_t> pops = calculate_populations(coords, n_rows, n_cols, radius);
          if (args.count("population")) {
            write_pops(args["population"].as<std::string>(), pops);
          }
          Clustering::logger(std::cout) << "calculating free energies" << std::endl;
          free_energies = calculate_free_energies(pops);
          if (args.count("free-energy")) {
            write_fes(args["free-energy"].as<std::string>(), free_energies);
          }
        }
      }
      //// nearest neighbors
      Neighborhood nh;
      Neighborhood nh_high_dens;
      if (args.count("nearest-neighbors-input")) {
        Clustering::logger(std::cout) << "re-using nearest neighbor data." << std::endl;
        auto nh_pair = read_neighborhood(args["nearest-neighbors-input"].as<std::string>());
        nh = nh_pair.first;
        nh_high_dens = nh_pair.second;
      } else if (args.count("nearest-neighbors") || args.count("output")) {
        Clustering::logger(std::cout) << "calculating nearest neighbors" << std::endl;
        if ( ! args.count("radius")) {
          std::cerr << "error: radius (-r) is required!" << std::endl;
        }
#ifdef USE_CUDA
        auto nh_tuple = Clustering::Density::CUDA::nearest_neighbors(coords
                                                                   , n_rows
                                                                   , n_cols
                                                                   , free_energies);
#else
        auto nh_tuple = nearest_neighbors(coords, n_rows, n_cols, free_energies);
#endif
        nh = std::get<0>(nh_tuple);
        nh_high_dens = std::get<1>(nh_tuple);
        if (args.count("nearest-neighbors")) {
          Clustering::Tools::write_neighborhood(args["nearest-neighbors"].as<std::string>(), nh, nh_high_dens);
        }
      }
      //// clustering
      if (args.count("output")) {
#ifdef USE_CUDA
        using Clustering::Density::CUDA::screening;
#else
        using Clustering::Density::screening;
#endif
        const std::string output_file = args["output"].as<std::string>();
        std::vector<std::size_t> clustering;
        if (args.count("input")) {
          Clustering::logger(std::cout) << "reading initial clusters from file." << std::endl;
          clustering = read_clustered_trajectory(args["input"].as<std::string>());
        }
        if (args.count("threshold-screening")) {
          std::vector<float> threshold_params = args["threshold-screening"].as<std::vector<float>>();
          if (threshold_params.size() > 3) {
            std::cerr << "error: option -T expects at most three floating point arguments: FROM STEP TO." << std::endl;
            exit(EXIT_FAILURE);
          }
          Clustering::logger(std::cout) << "running free energy landscape screening" << std::endl;
          float t_from = 0.1;
          float t_step = 0.1;
          float t_to = *std::max_element(free_energies.begin(), free_energies.end());
          if (threshold_params.size() >= 1 && threshold_params[0] >= 0.0f) {
            t_from = threshold_params[0];
          }
          if (threshold_params.size() >= 2) {
            t_step = threshold_params[1];
          }
          if (threshold_params.size() == 3) {
            t_to = threshold_params[2];
          }
          // upper limit extended to a 10th of the stepsize to
          // circumvent rounding errors when comparing on equality
          float t_to_low = t_to - t_step/10.0f + t_step;
          float t_to_high = t_to + t_step/10.0f + t_step;
          for (float t=t_from; (t < t_to_low) && !(t_to_high < t); t += t_step) {
            // compute clusters, re-using old results from previous step
            clustering = screening(free_energies
                                 , nh
                                 , t
                                 , coords
                                 , n_rows
                                 , n_cols
                                 , clustering);
            write_single_column(Clustering::Tools::stringprintf(output_file + ".%0.2f", t)
                              , clustering);
          }
        } else {
          Clustering::logger(std::cout) << "assigning low density states to initial clusters" << std::endl;
          clustering = assign_low_density_frames(clustering
                                               , nh_high_dens
                                               , free_energies);
          // sort, rename and save states
          Clustering::logger(std::cout) << "sorting clusters by decreasing population" << std::endl;
          clustering = sorted_cluster_names(clustering);
          Clustering::logger(std::cout) << "writing clusters to file " << output_file << std::endl;
          write_single_column<std::size_t>(output_file, clustering);
        }
      }
      Clustering::logger(std::cout) << "freeing coords" << std::endl;
      free_coords(coords);
    }
#endif
  } // end namespace Density
} // end namespace Clustering

