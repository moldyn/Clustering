
#include "tools.hpp"
#include "density_clustering_cuda.hpp"
#include "density_clustering_cuda_kernels.hpp"
#include "logger.hpp"

#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <iomanip>

#include <cuda.h>
#include <omp.h>


namespace Clustering {
namespace Density {
namespace CUDA {

  void
  check_error(std::string msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: "
                << msg << "\n"
                << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  int
  get_num_gpus() {
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    check_error("trying to get number of available GPUs");
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPUs found" << std::endl;
      exit(EXIT_FAILURE);
    } else {
      return n_gpus;
    }
  }

  Pops
  calculate_populations_per_gpu(const float* coords
                              , std::size_t n_rows
                              , std::size_t n_cols
                              , std::vector<float> radii
                              , std::size_t i_from
                              , std::size_t i_to
                              , int i_gpu) {
    using Clustering::Tools::min_multiplicator;
    ASSUME_ALIGNED(coords);
    unsigned int n_radii = radii.size();
    std::vector<float> rad2(n_radii);
    for (std::size_t i=0; i < n_radii; ++i) {
      rad2[i] = radii[i]*radii[i];
    }
    // GPU setup
    cudaSetDevice(i_gpu);
    float* d_coords;
    float* d_rad2;
    unsigned int* d_pops;
    cudaMalloc((void**) &d_coords
             , sizeof(float) * n_rows * n_cols);
    check_error("pop-calc device mallocs (coords)");
    cudaMalloc((void**) &d_pops
             , sizeof(unsigned int) * n_rows * n_radii);
    check_error("pop-calc device mallocs (pops)");
    cudaMalloc((void**) &d_rad2
             , sizeof(float) * n_radii);
    check_error("pop-calc device mallocs (rad2)");
    cudaMemset(d_pops
             , 0
             , sizeof(unsigned int) * n_rows * n_radii);
    check_error("pop-calc memset");
    cudaMemcpy(d_coords
             , coords
             , sizeof(float) * n_rows * n_cols
             , cudaMemcpyHostToDevice);
    cudaMemcpy(d_rad2
             , rad2.data()
             , sizeof(float) * n_radii
             , cudaMemcpyHostToDevice);
    check_error("pop-calc mem copies");
    int max_shared_mem;
    cudaDeviceGetAttribute(&max_shared_mem
                         , cudaDevAttrMaxSharedMemoryPerBlock
                         , i_gpu);
    check_error("getting max shared mem size");
    unsigned int block_size = BSIZE_POPS;
    unsigned int shared_mem = 2 * block_size * n_cols * sizeof(float);
    if (shared_mem > max_shared_mem) {
      std::cerr << "error: max. shared mem per block too small on this GPU.\n"
                << "       either reduce BSIZE_POPS or get a better GPU."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int block_rng = min_multiplicator(i_to-i_from, block_size);
//    Clustering::logger(std::cout) << "# blocks needed: "
//                                  << block_rng << std::endl;
    for (unsigned int i=0; i*block_size < n_rows; ++i) {
      Clustering::Density::CUDA::Kernel::population_count
      <<< block_rng
        , block_size
        , shared_mem >>> (i*block_size
                        , d_coords
                        , n_rows
                        , n_cols
                        , d_rad2
                        , n_radii
                        , d_pops
                        , i_from
                        , i_to);
    }
    cudaDeviceSynchronize();
    check_error("after kernel loop");
    // get partial results from GPU
    std::vector<unsigned int> partial_pops(n_rows*n_radii);
    cudaMemcpy(partial_pops.data()
             , d_pops
             , sizeof(unsigned int) * n_rows * n_radii
             , cudaMemcpyDeviceToHost);
    // sort into resulting pops
    Pops pops;
    for (unsigned int r=0; r < n_radii; ++r) {
      pops[radii[r]].resize(n_rows, 0);
      for (unsigned int i=i_from; i < i_to; ++i) {
        pops[radii[r]][i] = partial_pops[r*n_rows+i];
      }
    }
    cudaFree(d_coords);
    cudaFree(d_rad2);
    cudaFree(d_pops);
    return pops;
  }

  Pops
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii) {
    using Clustering::Tools::dim1_sorted_coords;
    using Clustering::Tools::boxlimits;
    ASSUME_ALIGNED(coords);
    std::sort(radii.begin(), radii.end(), std::greater<float>());
    int n_gpus = get_num_gpus();
    int gpu_range = n_rows / n_gpus;
    int i;
    std::vector<Pops> partial_pops(n_gpus);
    #pragma omp parallel for default(none)\
      private(i)\
      firstprivate(n_gpus,n_rows,n_cols,gpu_range)\
      shared(partial_pops,radii,coords)\
      num_threads(n_gpus)\
      schedule(dynamic,1)
    for (i=0; i < n_gpus; ++i) {
      // compute partial populations in parallel
      // on all available GPUs
      partial_pops[i] = calculate_populations_per_gpu(coords
                                                    , n_rows
                                                    , n_cols
                                                    , radii
                                                    , i*gpu_range
                                                    , i == (n_gpus-1)
                                                        ? n_rows
                                                        : (i+1)*gpu_range
                                                    , i);
    }
    Pops pops;
    // combine pops
    for (float r: radii) {
      pops[r].resize(n_rows, 0);
      for (i=0; i < n_rows; ++i) {
        for (unsigned int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
          pops[r][i] += partial_pops[i_gpu][r][i];
        }
      }
    }
    return pops;
  }

  std::tuple<Neighborhood, Neighborhood>
  nearest_neighbors_per_gpu(const float* coords
                          , const std::size_t n_rows
                          , const std::size_t n_cols
                          , const std::vector<float>& free_energy
                          , std::size_t i_from
                          , std::size_t i_to
                          , int i_gpu) {
    using Clustering::Tools::min_multiplicator;
    ASSUME_ALIGNED(coords);
    // GPU setup
    cudaSetDevice(i_gpu);
    float* d_coords;
    float* d_fe;
    unsigned int* d_nh_nhhd_ndx;
    float* d_nh_nhhd_dist;
    // allocate memory
    cudaMalloc((void**) &d_coords
             , sizeof(float) * n_rows * n_cols);
    cudaMalloc((void**) &d_fe
             , sizeof(float) * n_rows);
    cudaMalloc((void**) &d_nh_nhhd_ndx
             , sizeof(unsigned int) * n_rows * 2);
    cudaMalloc((void**) &d_nh_nhhd_dist
             , sizeof(float) * n_rows * 2);
    // initialize all min dists and indices to zero
    cudaMemset(d_nh_nhhd_ndx
             , 0
             , sizeof(unsigned int) * n_rows * 2);
    cudaMemset(d_nh_nhhd_dist
             , 0
             , sizeof(float) * n_rows * 2);
    // copy coordinates and free energies to GPU
    cudaMemcpy(d_coords
             , coords
             , sizeof(float) * n_rows * n_cols
             , cudaMemcpyHostToDevice);
    cudaMemcpy(d_fe
             , free_energy.data()
             , sizeof(float) * n_rows
             , cudaMemcpyHostToDevice);
    int max_shared_mem;
    cudaDeviceGetAttribute(&max_shared_mem
                         , cudaDevAttrMaxSharedMemoryPerBlock
                         , i_gpu);
    check_error("retrieving max shared mem");
    unsigned int block_size = BSIZE_NH;
    // compute necessary size of shared memory of block for
    // coordinates (2*n_cols) and free energies (1 col)
    unsigned int shared_mem = (2*n_cols + 1) * block_size * sizeof(float);
    if (shared_mem > max_shared_mem) {
      std::cerr << "error: max. shared mem per block too small on this GPU.\n"
                << "       either reduce block_size for NN search or get a "
                <<        "better GPU." << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int block_rng = min_multiplicator(i_to-i_from, block_size);
    for (unsigned int i=0; i*block_size < n_rows; ++i) {
      Clustering::Density::CUDA::Kernel::nearest_neighbor_search
      <<< block_rng
        , block_size
        , shared_mem >>> (i*block_size
                        , d_coords
                        , n_rows
                        , n_cols
                        , d_fe
                        , d_nh_nhhd_ndx
                        , d_nh_nhhd_dist
                        , i_from
                        , i_to);
    }
    cudaDeviceSynchronize();
    check_error("after kernel loop");
    // initialize neighborhoods (on host)
    Neighborhood nh;
    Neighborhood nhhd;
    // collect results from GPU
    std::vector<unsigned int> buf_ndx(n_rows * 2);
    std::vector<float> buf_dist(n_rows * 2);
    cudaMemcpy(buf_ndx.data()
             , d_nh_nhhd_ndx
             , sizeof(unsigned int) * n_rows * 2
             , cudaMemcpyDeviceToHost);
    cudaMemcpy(buf_dist.data()
             , d_nh_nhhd_dist
             , sizeof(float) * n_rows * 2
             , cudaMemcpyDeviceToHost);
    for (unsigned int i=0; i < n_rows; ++i) {
      nh[i] = {buf_ndx[i]
             , buf_dist[i]};
      nhhd[i] = {buf_ndx[n_rows+i]
               , buf_dist[n_rows+i]};
    }
    // device cleanup
    cudaFree(d_coords);
    cudaFree(d_fe);
    cudaFree(d_nh_nhhd_ndx);
    cudaFree(d_nh_nhhd_dist);
    // results
    return std::make_tuple(nh, nhhd);
  }

  std::tuple<Neighborhood, Neighborhood>
  nearest_neighbors(const float* coords
                  , const std::size_t n_rows
                  , const std::size_t n_cols
                  , const std::vector<float>& free_energy) {
    int n_gpus = get_num_gpus();
    std::vector<std::tuple<Neighborhood, Neighborhood>> partials(n_gpus);
    unsigned int gpu_range = n_rows / n_gpus;
    unsigned int i_gpu;
    #pragma omp parallel for default(none)\
      private(i_gpu)\
      firstprivate(n_gpus,n_rows,n_cols,gpu_range)\
      shared(partials,coords,free_energy)\
      num_threads(n_gpus)
    for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      partials[i_gpu] = nearest_neighbors_per_gpu(coords
                                                , n_rows
                                                , n_cols
                                                , free_energy
                                                , i_gpu*gpu_range
                                                , (i_gpu == (n_gpus-1))
                                                    ? n_rows
                                                    : (i_gpu+1)*gpu_range
                                                , i_gpu);
    }
    // combine partial neighborhood results from gpus
    Neighborhood nh;
    Neighborhood nhhd;
    std::tie(nh, nhhd) = partials[0];
    for (i_gpu=1; i_gpu < n_gpus; ++i_gpu) {
      Neighborhood partial_nh;
      Neighborhood partial_nhhd;
      std::tie(partial_nh, partial_nhhd) = partials[i_gpu];
      for (unsigned int i=0; i < n_rows; ++i) {
        if ((partial_nh[i].second != 0)
         || (partial_nhhd[i].second != 0)) {
          nh[i] = partial_nh[i];
          nhhd[i] = partial_nhhd[i];
        }
      }
    }
    return std::make_tuple(nh, nhhd);
  }

  std::vector<unsigned int>
  clustering_rebased(std::vector<unsigned int> clustering) {
    std::map<unsigned int, unsigned int> dict;
    // construct dictionary
    dict[0] = 0;
    for (unsigned int i=0; i < clustering.size(); ++i) {
      unsigned int s = clustering[i];
      if (dict.count(s) == 0) {
        dict[s] = i+1;
      }
    }
    // rebase
    for (unsigned int& s: clustering) {
      s = dict[s];
    }
    return clustering;
  }

  std::vector<unsigned int>
  merge_results(std::vector<std::vector<unsigned int>> clusterings
              , unsigned int max_row) {
    unsigned int n_results = clusterings.size();
    if (n_results == 0) {
      std::cerr << "error: there are no partial clustering results to merge!"
                << std::endl;
      exit(EXIT_FAILURE);
    } else {
      if (clusterings[0].size() == 0) {
        std::cerr << "error: no sampling, nothing to merge"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    max_row = std::min(max_row
                     , (unsigned int) clusterings[0].size());
    for (unsigned int i=0; i < max_row; ++i) {
      // collect start points, i.e. cluster assignemnts
      // of all partial results
      std::set<unsigned int> start_points;
      for (unsigned int j=0; j < n_results; ++j) {
        start_points.insert(clusterings[j][i]);
      }
      // not interested in zero-ed states (aka no assignment)
      if (start_points.count(0) > 0) {
        start_points.erase(0);
      }
      // follow start_points (id = i_state-1), collect all
      // states and rebase them to min(state)
      std::set<unsigned int> need_update = start_points;
      for (unsigned int s: start_points) {
        unsigned int s_old = 1;
        while (s != 0 && s_old != s) {
          s_old = s;
          s = clusterings[0][s-1];
          need_update.insert(s);
        }
      }
      // std::set is guaranteed to be ordered!
      unsigned int min_s = (*need_update.begin());
      for (unsigned int s: need_update) {
        clusterings[0][s-1] = min_s;
      }
    }
    return clusterings[0];
  }

  std::vector<std::size_t>
  screening(const std::vector<float>& free_energy
          , const Neighborhood& nh
          , const float free_energy_threshold
          , const float* coords
          , const std::size_t n_rows
          , const std::size_t n_cols
          , const std::vector<std::size_t> initial_clusters) {
    using Clustering::Tools::min_multiplicator;
    // data preparation
    std::size_t first_frame_above_threshold;
    double sigma2;
    std::vector<FreeEnergy> fe_sorted;
    std::vector<std::size_t> prev_clustering;
    unsigned int prev_max_state;
    std::tie(prev_clustering
           , first_frame_above_threshold
           , sigma2
           , fe_sorted
           , std::ignore
           , prev_max_state) = prepare_initial_clustering(free_energy
                                                        , nh
                                                        , free_energy_threshold
                                                        , n_rows
                                                        , initial_clusters);
    // measure runtime & give some informative output
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Clustering::logger(std::cout) << "    " << std::setw(6)
                                  << Clustering::Tools::stringprintf("%.3f", free_energy_threshold)
                                  << " " << std::setw(9)
                                  << Clustering::Tools::stringprintf("%i", first_frame_above_threshold)
                                  << std::endl;
    float max_dist2 = 4*sigma2;
    // prepare CUDA environment
    int n_gpus = get_num_gpus();
    std::vector<float*> d_coords_sorted(n_gpus);
    std::vector<unsigned int*> d_clustering(n_gpus);
    // sort coords (and previous clustering results)
    // according to free energies
    std::vector<float> tmp_coords_sorted(n_rows * n_cols);
    std::vector<unsigned int> clustering_sorted(n_rows);
    for (unsigned int i=0; i < n_rows; ++i) {
      for (unsigned int j=0; j < n_cols; ++j) {
        tmp_coords_sorted[i*n_cols+j] = coords[fe_sorted[i].first*n_cols+j];
      }
      // intialize energy-sorted clustering results
      if (i < first_frame_above_threshold) {
        clustering_sorted[i] = prev_clustering[fe_sorted[i].first];
      }
    }
    // initialize new (unclustered) frames
    unsigned int prev_last_frame =
      std::distance(clustering_sorted.begin()
                  , std::find(clustering_sorted.begin()
                            , clustering_sorted.end()
                            , 0));
    for (unsigned int i=prev_last_frame; i < first_frame_above_threshold; ++i) {
      clustering_sorted[i] = ++prev_max_state;
    }
    // computational range of single gpu
    unsigned int gpu_rng =
      min_multiplicator(first_frame_above_threshold - prev_last_frame
                      , n_gpus);
    if (gpu_rng == 0) {
      // nothing to do, since all frames below threshold were already
      // below previous threshold
      return initial_clusters;
    }
    int max_shared_mem;
    // assuming GPUs are of same type with same amount of memory
    cudaDeviceGetAttribute(&max_shared_mem
                         , cudaDevAttrMaxSharedMemoryPerBlock
                         , 0);
    check_error("getting max shared mem size");
    unsigned int shared_mem = 2 * BSIZE_SCR * n_cols * sizeof(float);
    //TODO!!!!!!!!! check shared_mem + const(shared_mem) < max_shared_mem
    
    
    unsigned int block_rng, i_from, i_to, i, i_gpu;
    // initialize GPUs
    for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      cudaSetDevice(i_gpu);
      // allocate memory on GPUs
      cudaMalloc((void**) &d_coords_sorted[i_gpu]
               , sizeof(float) * n_rows * n_cols);
      check_error("after malloc");
      cudaMalloc((void**) &d_clustering[i_gpu]
               , sizeof(unsigned int) * n_rows);
      check_error("after malloc");
      // copy sorted coords and previous clustering results to GPUs
      cudaMemcpy(d_coords_sorted[i_gpu]
               , tmp_coords_sorted.data()
               , sizeof(float) * n_rows * n_cols
               , cudaMemcpyHostToDevice);
      check_error("after memcopy of sorted coords");
    }
    // change state names in clustering to conform to lowest indices
    clustering_sorted = clustering_rebased(clustering_sorted);
    // fill zero-set indices with their own index
    for (unsigned int i=0; i < first_frame_above_threshold; ++i) {
      if (clustering_sorted[i] == 0) {
        clustering_sorted[i] = i+1;
      }
    }
    std::vector<unsigned int> clustering_sorted_old;
    while (clustering_sorted_old != clustering_sorted) {
      // cache old results
      clustering_sorted_old = clustering_sorted;
      // (re-)cluster
      #pragma omp parallel for\
        default(none)\
        private(i,i_gpu,block_rng,i_from,i_to)\
        firstprivate(n_gpus,n_rows,n_cols,gpu_rng,max_dist2,\
                     prev_last_frame,\
                     shared_mem,first_frame_above_threshold)\
        shared(d_coords_sorted,d_clustering,\
               tmp_coords_sorted,clustering_sorted)\
        num_threads(n_gpus)
      for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
        cudaSetDevice(i_gpu);
        check_error("after setting gpu device");
        cudaMemcpy(d_clustering[i_gpu]
                 , clustering_sorted.data()
                 , sizeof(unsigned int) * n_rows
                 , cudaMemcpyHostToDevice);
        check_error("after memcopy of prev clustering");
        i_from = prev_last_frame + i_gpu * gpu_rng;
        i_to = (i_gpu == (n_gpus-1))
             ? first_frame_above_threshold
             : prev_last_frame + (i_gpu+1) * gpu_rng;
        block_rng = min_multiplicator(i_to-i_from
                                    , BSIZE_SCR);
        for (i=0; i*BSIZE_SCR < first_frame_above_threshold; ++i) {
          Clustering::Density::CUDA::Kernel::screening
            <<< block_rng
              , BSIZE_SCR
              , shared_mem >>>
            (i*BSIZE_SCR
           , d_coords_sorted[i_gpu]
           , std::min(first_frame_above_threshold, n_rows)
           , n_cols
           , max_dist2
           , d_clustering[i_gpu]
           , i_from
           , i_to);
        }
        cudaDeviceSynchronize();
        check_error("after kernel loop");
      }
      // collect & merge clustering results from GPUs
      std::vector<std::vector<unsigned int>>
        clstr_results(n_gpus
                    , std::vector<unsigned int>(n_rows));
      for (int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
        cudaMemcpy(clstr_results[i_gpu].data()
                 , d_clustering[i_gpu]
                 , sizeof(unsigned int) * n_rows
                 , cudaMemcpyDeviceToHost);
      }
      clustering_sorted = merge_results(clstr_results
                                      , first_frame_above_threshold);
      // update references by comparing old to new results
      std::unordered_map<unsigned int, unsigned int> dict;
      dict[0] = 0;
      for (i=0; i < first_frame_above_threshold; ++i) {
        unsigned int state_old = clustering_sorted_old[i];
        unsigned int state_new = clustering_sorted[i];
        if (dict.count(state_old) == 0) {
          dict[state_old] = std::min(state_old, state_new);
        } else {
          dict[state_old] = std::min(dict[state_old], state_new);
        }
      }
      for (unsigned int& s: clustering_sorted) {
        s = dict[s];
      }
    } // end while
    // cleanup CUDA environment
    for (int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      cudaFree(d_coords_sorted[i_gpu]);
      cudaFree(d_clustering[i_gpu]);
    }
    // convert state trajectory from
    // FE-sorted order to original order
    std::vector<std::size_t> clustering(n_rows);
    for (unsigned int i=0; i < n_rows; ++i) {
      clustering[fe_sorted[i].first] = clustering_sorted[i];
    }
    // final output
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//    Clustering::logger(std::cout) << "    runtime: "
//                                  << std::chrono::duration_cast
//                                       <std::chrono::seconds>(t1-t0).count()
//                                  << " secs"
//                                  << std::endl;
    return normalized_cluster_names(first_frame_above_threshold
                                  , clustering
                                  , fe_sorted);
  }

}}} // end Clustering::Density::CUDA

