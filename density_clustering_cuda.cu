
#include "tools.hpp"
#include "density_clustering_cuda.hpp"
#include "logger.hpp"

#include <algorithm>

#include <cuda.h>
#include <omp.h>

#include "lts_cuda_kernels.cuh"

// for pops
#define BSIZE_POPS 512
//#define BSIZE_POPS 1024

// for neighborhood search
#define BSIZE_NH 128
#define N_STREAMS_NH 1

// for screening
#define BSIZE_SCR 512

namespace Clustering {
namespace Density {
namespace CUDA {

  __global__ void
  population_count(unsigned int offset
                 , float* coords
                 , unsigned int n_rows
                 , unsigned int n_cols
                 , float* radii2
                 , unsigned int n_radii
                 , unsigned int* pops
                 , unsigned int i_from
                 , unsigned int i_to) {
    extern __shared__ float smem[];
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int gid = bid * bsize + tid + i_from;
    // load frames for comparison into shared memory
    int comp_size = min(bsize, n_rows - offset);
    if (tid < comp_size) {
      for (unsigned int j=0; j < n_cols; ++j) {
        smem[tid*n_cols+j] = coords[(tid+offset)*n_cols+j];
      }
    }
    __syncthreads();
    // count neighbors
    if (gid < i_to) {
      unsigned int ref_id = tid+bsize;
      // load reference coordinates for re-use into shared memory
      for (unsigned int j=0; j < n_cols; ++j) {
        smem[ref_id*n_cols+j] = coords[gid*n_cols+j];
      }
      for (unsigned int r=0; r < n_radii; ++r) {
        unsigned int local_pop = 0;
        float rad2 = radii2[r];
        for (unsigned int i=0; i < comp_size; ++i) {
          float dist2 = 0.0f;
          for (unsigned int j=0; j < n_cols; ++j) {
            float c = smem[ref_id*n_cols+j] - smem[i*n_cols+j];
            dist2 = fma(c, c, dist2);
          }
          if (dist2 <= rad2) {
            ++local_pop;
          }
        }
        // update frame populations (per radius)
        pops[r*n_rows+gid] += local_pop;
      }
    }
  }

  __global__ void
  nearest_neighbor_search(unsigned int offset
                        , float* coords
                        , unsigned int n_rows
                        , unsigned int n_cols
                        , float* fe
                        , float* nh_dist_ndx
                        , float* nhhd_dist_ndx
                        , unsigned int i_from
                        , unsigned int i_to) {
    extern __shared__ float smem[];
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int gid = bid * bsize + tid + i_from;

    float nh_mindist;
    float nh_minndx;
    float nhhd_mindist;
    float nhhd_minndx;
    float ref_fe;
    unsigned int ref_id;

    // load frames for comparison into shared memory
    int comp_size = min(bsize, n_rows - offset);
    if (tid < comp_size) {
      for (unsigned int j=0; j < n_cols; ++j) {
        smem[tid*n_cols+j] = coords[(tid+offset)*n_cols+j];
      }
    }
    __syncthreads();

    if (gid < i_to) {
      ref_id = tid+bsize;
      // load reference coordinates for re-use into shared memory
      for (unsigned int j=0; j < n_cols; ++j) {
        smem[ref_id*n_cols+j] = coords[gid*n_cols+j];
      }
      ref_fe = fe[gid];
      // load current best mindists into registers
      nh_mindist = nh_dist_ndx[gid];
      nh_minndx = nh_dist_ndx[n_rows+gid];
      nhhd_mindist = nhhd_dist_ndx[gid];
      nhhd_minndx = nhhd_dist_ndx[n_rows+gid];
      // compare squared distances of reference
      // compare squared distances of reference
      // to (other) frames in shared mem
      for (unsigned int i=0; i < comp_size; ++i) {
        float dist2=0.0f;
        for (unsigned int j=0; j < n_cols; ++j) {
          float c = smem[ref_id*n_cols+j] - smem[i*n_cols+j];
          dist2 = fma(c, c, dist2);
        }
        // frame with min distance (i.e. nearest neighbor)
        if ((nh_mindist == 0)
         || (dist2 < nh_mindist && dist2 != 0)) {
          nh_mindist = dist2;
          nh_minndx = i+offset;
        }
        // frame with min distance and lower energy
        if ((nhhd_mindist == 0 && fe[i+offset] < ref_fe)
         || (dist2 < nhhd_mindist && fe[i+offset] < ref_fe && dist2 != 0)) {
          nhhd_mindist = dist2;
          nhhd_minndx = i+offset;
        }
      }
      // write results (dist & ndx) to global buffers
      nh_dist_ndx[gid] = nh_mindist;
      nh_dist_ndx[n_rows+gid] = nh_minndx;
      nhhd_dist_ndx[gid] = nhhd_mindist;
      nhhd_dist_ndx[n_rows+gid] = nhhd_minndx;
    }
  }

  ////

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
    cudaMalloc((void**) &d_pops
             , sizeof(unsigned int) * n_rows * n_radii);
    cudaMalloc((void**) &d_rad2
             , sizeof(float) * n_radii);
    check_error("pop-calc device mallocs");
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
    Clustering::logger(std::cout) << "# blocks needed: "
                                  << block_rng << std::endl;
    for (unsigned int i=0; i*block_size < n_rows; ++i) {
      population_count <<< block_rng
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
    cudaStream_t streams[N_STREAMS_NH];
    float* d_coords;
    float* d_fe;
    float* d_nh[N_STREAMS_NH];
    float* d_nhhd[N_STREAMS_NH];
    cudaMalloc((void**) &d_coords
             , sizeof(float) * n_rows * n_cols);
    cudaMalloc((void**) &d_fe
             , sizeof(float) * n_rows);
    for (unsigned int i=0; i < N_STREAMS_NH; ++i) {
      cudaMalloc((void**) &d_nh[i]
               , sizeof(float) * n_rows * 2);
      cudaMalloc((void**) &d_nhhd[i]
               , sizeof(float) * n_rows * 2);
      cudaMemset(d_nh[i]
               , 0
               , sizeof(float) * n_rows * 2);
      cudaMemset(d_nhhd[i]
               , 0
               , sizeof(float) * n_rows * 2);
      cudaStreamCreate(&streams[i]);
    }
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
    unsigned int shared_mem = 2 * block_size * n_cols * sizeof(float);
    if (shared_mem > max_shared_mem) {
      std::cerr << "error: max. shared mem per block too small on this GPU.\n"
                << "       either reduce block_size for NN search or get a "
                <<        "better GPU." << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int block_rng = min_multiplicator(i_to-i_from, block_size);
    for (unsigned int i=0; i*block_size < n_rows; ++i) {
      unsigned int i_stream = i % N_STREAMS_NH;
      nearest_neighbor_search <<< block_rng
                                , block_size
                                , shared_mem
                                , streams[i_stream] >>> (i*block_size
                                                       , d_coords
                                                       , n_rows
                                                       , n_cols
                                                       , d_fe
                                                       , d_nh[i_stream]
                                                       , d_nhhd[i_stream]
                                                       , i_from
                                                       , i_to);
    }
    cudaDeviceSynchronize();
    check_error("after kernel loop");
    // initialize neighborhoods
    Neighborhood nh;
    Neighborhood nhhd;
    for (unsigned int i=0; i < n_rows; ++i) {
      nh[i] = {i, std::numeric_limits<float>::max()};
      nhhd[i] = {i, std::numeric_limits<float>::max()};
    }
    // collect partial results from streams
    for (unsigned int i_stream=0; i_stream < N_STREAMS_NH; ++i_stream) {
      std::vector<float> dist_ndx(n_rows * 2);
      auto update_nh = [&dist_ndx,n_rows] (Neighborhood& _nh) -> void {
        for (unsigned int i=0; i < n_rows; ++i) {
          if (dist_ndx[i] < _nh[i].second && dist_ndx[i] != 0) {
            _nh[i] = {(unsigned int) dist_ndx[n_rows+i]
                    , dist_ndx[i]};
          }
        }
      };
      cudaMemcpy(dist_ndx.data()
               , d_nh[i_stream]
               , sizeof(float) * n_rows * 2
               , cudaMemcpyDeviceToHost);
      update_nh(nh);
      cudaMemcpy(dist_ndx.data()
               , d_nhhd[i_stream]
               , sizeof(float) * n_rows * 2
               , cudaMemcpyDeviceToHost);
      update_nh(nhhd);
    }
    // device cleanup
    cudaFree(d_coords);
    cudaFree(d_fe);
    for (unsigned int i=0; i < N_STREAMS_NH; ++i) {
      cudaFree(d_nh[i]);
      cudaFree(d_nhhd[i]);
    }
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
                                                , i_gpu == (n_gpus-1)
                                                        ? n_rows
                                                        : (i_gpu+1)*gpu_range
                                                , i_gpu);
    }
    // combine partial neighborhood results from different gpus
    Neighborhood nh;
    Neighborhood nhhd;
    std::tie(nh, nhhd) = partials[0];
    for (i_gpu=1; i_gpu < n_gpus; ++i_gpu) {
      Neighborhood partial_nh;
      Neighborhood partial_nhhd;
      std::tie(partial_nh, partial_nhhd) = partials[i_gpu];
      for (unsigned int i=0; i < n_rows; ++i) {
        if (partial_nh[i].second < nh[i].second) {
          nh[i] = partial_nh[i];
        }
        if (partial_nhhd[i].second < nhhd[i].second) {
          nhhd[i] = partial_nhhd[i];
        }
      }
    }
    return std::make_tuple(nh, nhhd);
  }


  __global__ void
  initial_density_clustering_krnl(unsigned int offset
                                , float* sorted_coords
                                , unsigned int n_rows
                                , unsigned int n_cols
                                , float max_dist2
                                , unsigned int* clustering
                                , unsigned int i_from
                                , unsigned int i_to) {
    extern __shared__ float smem[];
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int gid = bid * bsize + tid + i_from;
    // load frames for comparison into shared memory
    int comp_size = min(bsize, n_rows - offset);
    if (tid < comp_size) {
      for (unsigned int j=0; j < n_cols; ++j) {
        smem[tid*n_cols+j] = sorted_coords[(tid+offset)*n_cols+j];
      }
    }
    __syncthreads();
    if (gid < i_to) {
      unsigned int self_id = gid + 1;
      unsigned int cluster_id = clustering[gid];
      if (cluster_id == 0) {
        cluster_id = self_id;
      }
      unsigned int ref_id = tid+bsize;
      // load reference coordinates for re-use into shared memory
      for (unsigned int j=0; j < n_cols; ++j) {
        smem[ref_id*n_cols+j] = sorted_coords[gid*n_cols+j];
      }
      for (unsigned int i=0; i < comp_size; ++i) {
        float dist2 = 0.0f;
        for (unsigned int j=0; j < n_cols; ++j) {
          float c = smem[ref_id*n_cols+j] - smem[i*n_cols+j];
          dist2 = fma(c, c, dist2);
        }
        if (dist2 < max_dist2) {
          unsigned int min_id = min(cluster_id, i+offset+1);
          if (min_id == self_id) {
            cluster_id = max(cluster_id, i+offset+1);
          } else {
            cluster_id = min_id;
          }
        }
      }
      clustering[gid] = cluster_id;
    }
  }

//  unsigned int
//  follow(const std::vector<unsigned int>& clustering
//       , unsigned int id
//       , unsigned int min_id = 0) {
//    if (id == 0
//     || id == min_id) {
//      return id;
//    } else {
//      return follow(clustering
//                  , clustering[id-1]
//                  , (min_id == 0)
//                      ? id
//                      : std::min(id, min_id));
//    }
//  }

  std::vector<unsigned int>
  sanitize_state_names(std::vector<unsigned int> clustering) {

    //TODO: lump frames into microstates.
    //      be careful: cluster ids do not necessarily
    //                  denote frame ids (especially not with previous
    //                  clustering)

    // map names to ascending index
    unsigned int id=0;
    std::map<unsigned int, unsigned int> name_dict;
    for (unsigned int name: unique_names) {
      ++id;
      name_dict[name] = id;
    }
    // relabel states
    for (unsigned int& i: clustering) {
      i = name_dict[i];
    }
    return clustering;
  }

//TODO remove old code
//    for (unsigned int i=0; i < clustering.size(); ++i) {
//      clustering[i] = follow(clustering
//                           , clustering[i]);
//    }
//    return clustering;
//  }


  std::vector<std::size_t>
  initial_density_clustering(const std::vector<float>& free_energy
                           , const Neighborhood& nh
                           , const float free_energy_threshold
                           , const float* coords
                           , const std::size_t n_rows
                           , const std::size_t n_cols
                           , const std::vector<std::size_t> initial_clusters) {
    using Clustering::Tools::min_multiplicator;
    std::size_t first_frame_above_threshold;
    double sigma2;
    std::vector<FreeEnergy> fe_sorted;
    std::vector<std::size_t> prev_clustering;
    std::size_t prev_max_state;
    // data preparation
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
    // write log
//    screening_log(sigma2
//                , first_frame_above_threshold
//                , fe_sorted);
    float max_dist2 = 4*sigma2;
    // prepare CUDA environment
    int n_gpus = get_num_gpus();
    std::vector<float*> d_coords_sorted(n_gpus);
    std::vector<unsigned int*> d_clustering(n_gpus);
    // sort coords (and previous clustering results)
    // according to free energies
    std::vector<float> tmp_coords_sorted(n_rows * n_cols);
    std::vector<unsigned int> prev_clustering_sorted(n_rows);
    for (unsigned int i=0; i < n_rows; ++i) {
      for (unsigned int j=0; j < n_cols; ++j) {
        tmp_coords_sorted[i*n_cols+j] = coords[fe_sorted[i].first*n_cols+j];
      }
      prev_clustering_sorted[i] = prev_clustering[fe_sorted[i].first];
    }
    unsigned int block_rng;
//TODO: check: what happens if no new frames are added compared to prev run?
    unsigned int i_from;
    unsigned int i_to;
    unsigned int i;
    // re-use initial clustering results
    unsigned int prev_last_frame = 0;
    if (prev_clustering_sorted[0] != 0) {
      for (i=1; i < n_rows; ++i) {
        if (prev_clustering_sorted[i] == 0) {
          prev_last_frame = i;
          break;
        }
      }
    }
    int i_gpu;
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
    #pragma omp parallel for\
      default(none)\
      private(i,i_gpu,block_rng,i_from,i_to)\
      firstprivate(n_gpus,n_rows,n_cols,gpu_rng,max_dist2,\
                   prev_last_frame,prev_max_state,\
                   shared_mem,first_frame_above_threshold)\
      shared(d_coords_sorted,d_clustering,\
             tmp_coords_sorted,prev_clustering_sorted)\
      num_threads(n_gpus)
    for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      // allocate memory and copy sorted coords to GPUs
      cudaSetDevice(i_gpu);
      cudaMalloc((void**) &d_coords_sorted[i_gpu]
               , sizeof(float) * n_rows * n_cols);
      cudaMalloc((void**) &d_clustering[i_gpu]
               , sizeof(unsigned int) * n_rows);
      cudaMemcpy(d_coords_sorted[i_gpu]
               , tmp_coords_sorted.data()
               , sizeof(float) * n_rows * n_cols
               , cudaMemcpyHostToDevice);


      //TODO change kernel to use two lists: own state + min next-neighbor state


      // copy prev results to GPU-buffer (and set the rest to zero)
      cudaMemset(d_clustering[i_gpu]
               , 0
               , sizeof(unsigned int) * n_rows);
      cudaMemcpy(d_clustering[i_gpu]
               , prev_clustering_sorted.data()
               , sizeof(unsigned int) * n_rows
               , cudaMemcpyHostToDevice);
      // perform initial clustering on yet unclustered frames
      i_from = prev_last_frame + i_gpu * gpu_rng;
      i_to = (i_gpu == (n_gpus-1))
           ? first_frame_above_threshold
           : prev_last_frame + (i_gpu+1) * gpu_rng;
      block_rng = min_multiplicator(i_to-i_from
                                  , BSIZE_SCR);
      for (i=0; i*BSIZE_SCR < first_frame_above_threshold; ++i) {
        initial_density_clustering_krnl
          <<< block_rng
            , BSIZE_SCR
            , shared_mem >>>
          (i*BSIZE_SCR
         , d_coords_sorted[i_gpu]
         , n_rows
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
    std::vector<unsigned int> clustering_sorted = prev_clustering_sorted;
    for (int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      std::vector<unsigned int> tmp_clust(n_rows, 0);
      cudaMemcpy(tmp_clust.data()
               , d_clustering[i_gpu]
               , sizeof(unsigned int) * first_frame_above_threshold
               , cudaMemcpyDeviceToHost);
      for (i=prev_last_frame; i < first_frame_above_threshold; ++i) {
        if (i_gpu == 0) {
          clustering_sorted[i] = tmp_clust[i];
        } else {
          clustering_sorted[i] = std::max(clustering_sorted[i]
                                        , tmp_clust[i]);
        }
      }
    }
    // reduce clustering to min number of ids
    clustering_sorted = sanitize_state_names(clustering_sorted);
    //TODO: refined clustering (by rerunning with alternative kernel)




    // convert state trajectory from
    // FE-sorted order to original order
    std::vector<std::size_t> clustering(n_rows, 0);
    for (unsigned int i=0; i < n_rows; ++i) {
      clustering[fe_sorted[i].first] = clustering_sorted[i];
    }
    // cleanup CUDA environment
    for (int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      cudaFree(d_coords_sorted[i_gpu]);
      cudaFree(d_clustering[i_gpu]);
    }
    return normalized_cluster_names(first_frame_above_threshold
                                  , clustering
                                  , fe_sorted);
  }

}}} // end Clustering::Density::CUDA

