
#include "tools.hpp"
#include "density_clustering_cuda.hpp"
#include "logger.hpp"

#include <algorithm>
#include <unordered_set>
#include <unordered_map>

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
                                , unsigned int* ref_states
                                , unsigned int i_from
                                , unsigned int i_to) {
    // dynamic shared mem for ref coords
    extern __shared__ float smem_coords[];
    // static shared mem for cluster ids
    __shared__ unsigned int smem_ref_states[BSIZE_SCR];
    // thread dimensions
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int gid = bid * bsize + tid + i_from;
    // load frames for comparison into shared memory
    int comp_size = min(bsize, n_rows - offset);
    if (tid < comp_size) {
      for (unsigned int j=0; j < n_cols; ++j) {
        smem_coords[tid*n_cols+j] = sorted_coords[(tid+offset)*n_cols+j];
      }
      smem_ref_states[tid] = ref_states[(tid+offset)];
    }
    __syncthreads();
    if (gid < i_to) {
      unsigned int min_id = clustering[gid];
      // load reference coordinates for re-use into shared memory
      unsigned int ref_id = tid+bsize;
      for (unsigned int j=0; j < n_cols; ++j) {
        smem_coords[ref_id*n_cols+j] = sorted_coords[gid*n_cols+j];
      }
      // check against reference structures: if close enough, they'll be
      // considered as the same state (choosing min. state id).
      // state lumping will be done on host.
      for (unsigned int i=0; i < comp_size; ++i) {
        if (min_id != smem_ref_states[i]
         && smem_ref_states[i] != 0) {
          float dist2 = 0.0f;
          for (unsigned int j=0; j < n_cols; ++j) {
            float c = smem_coords[ref_id*n_cols+j] - smem_coords[i*n_cols+j];
            dist2 = fma(c, c, dist2);
          }
          if (dist2 < max_dist2
           && smem_ref_states[i] < min_id) {
            min_id = smem_ref_states[i];
          }
        }
      }
      clustering[gid] = min_id;
    }
  }

  std::vector<unsigned int>
  lumped_clusters(std::vector<unsigned int> clustering
                , std::vector<unsigned int> ref_states
                , unsigned int first_frame_above_threshold) {
    std::unordered_set<unsigned int> sstates;
    sstates.insert(clustering.begin()
                 , clustering.begin() + first_frame_above_threshold);
    sstates.insert(ref_states.begin()
                 , ref_states.begin() + first_frame_above_threshold);
    sstates.erase(0);
    std::vector<unsigned int> states(sstates.begin(), sstates.end());
    unsigned int n_states = states.size();
    // identify lumps
    unsigned int i, j, s, s1, s2;
    std::map<unsigned int, unsigned int> state_mapping;
    for (i=0; i < n_states; ++i) {
      state_mapping[states[i]] = states[i];
    }
    #pragma omp parallel for\
      default(none)\
      private(i,j,s1,s2,s)\
      firstprivate(n_states,first_frame_above_threshold)\
      shared(state_mapping,states,clustering,ref_states)
    for (i=0; i < n_states; ++i) {
      s = states[i];
      for (j=0; j < first_frame_above_threshold; ++j) {
        s1 = clustering[j];
        s2 = ref_states[j];
        //TODO: no loop over states, just over frames,
        //      then select 's' from {s1,s2}, compute min and update
        //      state_mapping in critical section
        if (s1 == s || s2 == s) {
          state_mapping[s] = std::min(state_mapping[s]
                                    , std::min(s1, s2));
        }
      }
    }
    // assign new (ordered) names to lumps
    std::set<unsigned int> distinct_states;
    for (auto sm: state_mapping) {
      distinct_states.insert(sm.second);
    }
    std::map<unsigned int, unsigned int> remapped_distinct;
    unsigned int i_state = 0;
    for (unsigned int s: distinct_states) {
      remapped_distinct[s] = ++i_state;
    }
    for (auto& sm: state_mapping) {
      sm.second = remapped_distinct[sm.second];
    }
    // remap states to lumps
    #pragma omp parallel for\
      default(none)\
      private(i)\
      firstprivate(first_frame_above_threshold)\
      shared(clustering,state_mapping)
    for (i=0; i < first_frame_above_threshold; ++i) {
      clustering[i] = state_mapping[clustering[i]];
    }
    return clustering;
  }

  std::vector<unsigned int>
  lumped_clusters__old(std::vector<unsigned int> clustering
                , std::vector<unsigned int> ref_states
                , unsigned int first_frame_above_threshold) {
    unsigned int i_state = 0;
    std::map<unsigned int, std::unordered_set<unsigned int>> lumps;
    std::set<std::pair<unsigned int, unsigned int>> pairings;
    for (unsigned int i=0; i < first_frame_above_threshold; ++i) {
      unsigned int s1 = clustering[i];
      unsigned int s2 = ref_states[i];
      pairings.emplace(s1, s2);
    }
    std::unordered_set<unsigned int> seen_states;
    for (auto p: pairings) {
      unsigned int s1 = p.first;
      unsigned int s2 = p.second;
      if (s1 == s2 && seen_states.count(s1) > 0) {
        continue;
      } else {
        seen_states.insert(s1);
        seen_states.insert(s2);
      }
      unsigned int l1=0, l2=0;
      // find lumps of states
      for (auto l: lumps) {
        if (l.second.count(s1) > 0) {
          l1 = l.first;
        }
        if (l.second.count(s2) > 0) {
          l2 = l.first;
        }
        if (l1 != 0 && l2 != 0) {
          break;
        }
      }
      if (l1 == 0 && l2 == 0) {
        // no lump found: make new one
        lumps[++i_state] = {s1, s2};
      } else if (l1 == 0 && l2 != 0) {
        // s2 has lump: add s1
        lumps[l2].insert(s1);
      } else if (l1 != 0 && l2 == 0) {
        // s1 has lump: add s2
        lumps[l1].insert(s2);
      } else if (l1 != l2) {
        // s1 and s2 have different lumps: join them
        auto mm = std::minmax(l1,l2);
        lumps[mm.first].insert(lumps[mm.second].begin()
                             , lumps[mm.second].end());
        lumps.erase(mm.second);
      }
    }
    // recluster trajectory
    i_state = 0;
    std::unordered_map<unsigned int, unsigned int> dict;
    for (auto l: lumps) {
      ++i_state;
      for (auto s: l.second) {
        dict[s] = i_state;
      }
    }
    for (unsigned int i=0; i < clustering.size(); ++i) {
      if (clustering[i] > 0) {
        clustering[i] = dict[clustering[i]];
      }
    }
    return clustering;
  }

  std::vector<std::size_t>
  initial_density_clustering(const std::vector<float>& free_energy
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
    // write log
    screening_log(sigma2
                , first_frame_above_threshold
                , fe_sorted);
    float max_dist2 = 4*sigma2;
    // prepare CUDA environment
    int n_gpus = get_num_gpus();
    std::vector<float*> d_coords_sorted(n_gpus);
    std::vector<unsigned int*> d_clustering(n_gpus);
    std::vector<unsigned int*> d_ref_states(n_gpus);
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
    unsigned int block_rng, i_from, i_to, i, i_gpu;
    // lump microstates until nothing changes
    bool microstates_lumped = true;
    unsigned int i_loop = 0;
    // initialize GPUs
    for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      cudaSetDevice(i_gpu);
      // allocate memory on GPUs
      cudaMalloc((void**) &d_coords_sorted[i_gpu]
               , sizeof(float) * n_rows * n_cols);
      cudaMalloc((void**) &d_clustering[i_gpu]
               , sizeof(unsigned int) * n_rows);
      cudaMalloc((void**) &d_ref_states[i_gpu]
               , sizeof(unsigned int) * n_rows);
      // copy sorted coords and previous clustering results to GPUs
      cudaMemcpy(d_coords_sorted[i_gpu]
               , tmp_coords_sorted.data()
               , sizeof(float) * n_rows * n_cols
               , cudaMemcpyHostToDevice);
    }
    while (microstates_lumped) {
      std::vector<unsigned int> clustering_sorted_orig = clustering_sorted;
      std::cerr << "microstate lumping iteration " << ++i_loop << std::endl;
      #pragma omp parallel for\
        default(none)\
        private(i,i_gpu,block_rng,i_from,i_to)\
        firstprivate(n_gpus,n_rows,n_cols,gpu_rng,max_dist2,\
                     prev_last_frame,\
                     shared_mem,first_frame_above_threshold)\
        shared(d_coords_sorted,d_clustering,d_ref_states,\
               tmp_coords_sorted,clustering_sorted)\
        num_threads(n_gpus)
      for (i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
        cudaSetDevice(i_gpu);
        cudaMemcpy(d_clustering[i_gpu]
                 , clustering_sorted.data()
                 , sizeof(unsigned int) * n_rows
                 , cudaMemcpyHostToDevice);
        cudaMemcpy(d_ref_states[i_gpu]
                 , clustering_sorted.data()
                 , sizeof(unsigned int) * n_rows
                 , cudaMemcpyHostToDevice);
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
           , d_ref_states[i_gpu]
           , i_from
           , i_to);
        }
        cudaDeviceSynchronize();
        check_error("after kernel loop");
      }
      // collect & merge clustering results from GPUs
      for (int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
        std::vector<unsigned int> tmp_clust(n_rows, 0);
        cudaMemcpy(tmp_clust.data()
                 , d_clustering[i_gpu]
                 , sizeof(unsigned int) * n_rows
                 , cudaMemcpyDeviceToHost);
        for (i=0; i < first_frame_above_threshold; ++i) {
          if (i_gpu == 0) {
            clustering_sorted[i] = tmp_clust[i];
          } else {
            clustering_sorted[i] = std::min(clustering_sorted[i]
                                          , tmp_clust[i]);
          }
        }
      }
      std::cout << "lumping CUDA-results ..." << std::endl;
      // lump microstates
      clustering_sorted = lumped_clusters(clustering_sorted
                                        , clustering_sorted_orig
                                        , first_frame_above_threshold);
      std::cout << "     ... finished" << std::endl;
      // compare prev_clustering to clustering_sorted:
      // if equal, end while
      microstates_lumped = (clustering_sorted_orig != clustering_sorted);
    } // end while (if microstates_lumped == false)
    // cleanup CUDA environment
    for (int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
      cudaFree(d_coords_sorted[i_gpu]);
      cudaFree(d_clustering[i_gpu]);
      cudaFree(d_ref_states[i_gpu]);
    }
    // convert state trajectory from
    // FE-sorted order to original order
    std::vector<std::size_t> clustering(n_rows);
    for (unsigned int i=0; i < n_rows; ++i) {
      clustering[fe_sorted[i].first] = clustering_sorted[i];
    }
    return normalized_cluster_names(first_frame_above_threshold
                                  , clustering
                                  , fe_sorted);
  }

}}} // end Clustering::Density::CUDA

