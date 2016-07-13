
#include "tools.hpp"
#include "density_clustering_cuda.hpp"
#include "logger.hpp"

#include <algorithm>

#include <cuda.h>
#include <omp.h>

#include "lts_cuda_kernels.cuh"

// for pops
#define BSIZE_POPS 64
//#define N_STREAMS 8

// for neighborhood search
#define BSIZE_NH 128
#define N_STREAMS_NH 1

namespace Clustering {
namespace Density {
namespace CUDA {

/*
  template <unsigned int _BLOCKSIZE>
  __global__ void
  reduce_sum_uint(unsigned int offset
                , unsigned int* vals
                , unsigned int n_vals
                , unsigned int* results
                , unsigned int i_result) {
    __shared__ unsigned int sum_block[_BLOCKSIZE];
    unsigned int stride;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid*_BLOCKSIZE+tid;
    unsigned int gid2 = gid + _BLOCKSIZE*gridDim.x;
    // store probs locally for reduction
    if (gid2 < n_vals) {
      // initial double load and first reduction
      sum_block[tid] = vals[gid+offset] + vals[gid2+offset];
    } else if (gid < n_vals) {
      sum_block[tid] = vals[gid+offset];
    } else {
      sum_block[tid] = 0;
    }
    for (stride=_BLOCKSIZE/2; stride > 0; stride /= 2) {
      __syncthreads();
      if (tid < stride) {
        sum_block[tid] += sum_block[tid+stride];
      }
    }
    if (tid == 0) {
      atomicAdd(&results[i_result], sum_block[0]);
    }
  }

  __global__ void
  in_radius(unsigned int offset
          , float* sorted_coords
          , float* coords
          , unsigned int i_ref
          , unsigned int n_rows
          , unsigned int n_cols
          , float* radii2
          , unsigned int n_radii
          , unsigned int* in_radius) {
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid*BSIZE+tid;
    float c;
    float dist2 = 0.0f;
    unsigned int j,r;
    // load reference coordinates into shared buffer
    extern __shared__ float ref_coords[];
    if (tid < n_cols) {
      ref_coords[tid] = coords[i_ref*n_cols+tid];
    }
    __syncthreads();
    if (gid+offset < n_rows) {
      // compute squared euclidean distance
      for (j=0; j < n_cols; ++j) {
        c = ref_coords[j] - sorted_coords[(gid + offset)*n_cols+j];
        dist2 = fma(c, c, dist2);
      }
      // write results: 1.0 if in radius, 0.0 if not
      for (r=0; r < n_radii; ++r) {
        if (dist2 <= radii2[r]) {
          in_radius[r*n_rows + gid] = 1;
        } else {
          in_radius[r*n_rows + gid] = 0;
        }
      }
    }
  }
*/

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

  void check_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
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
    // square radii
    for (float& r: radii) {
      r *= r;
    }
    // GPU setup
    cudaSetDevice(i_gpu);
    float* d_coords;
    float* d_radii2;
    unsigned int* d_pops;
    cudaMalloc((void**) &d_coords
             , sizeof(float) * n_rows * n_cols);
    cudaMalloc((void**) &d_pops
             , sizeof(unsigned int) * n_rows * n_radii);
    cudaMalloc((void**) &d_radii2
             , sizeof(float) * n_radii);
    cudaMemset(d_pops
             , 0
             , sizeof(unsigned int) * n_rows * n_radii);
    cudaMemcpy(d_coords
             , coords
             , sizeof(float) * n_rows * n_cols
             , cudaMemcpyHostToDevice);
    cudaMemcpy(d_radii2
             , radii.data()
             , sizeof(float) * n_radii
             , cudaMemcpyHostToDevice);
    int max_shared_mem;
    cudaDeviceGetAttribute(&max_shared_mem
                         , cudaDevAttrMaxSharedMemoryPerBlock
                         , i_gpu);
    check_error();
    unsigned int block_size = BSIZE_POPS;
    unsigned int shared_mem = 2 * block_size * n_cols * sizeof(float);
    if (shared_mem > max_shared_mem) {
      std::cerr << "error: max. shared mem per block too small on this GPU.\n"
                << "       either reduce BSIZE_POPS or get a better GPU."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    unsigned int block_rng = min_multiplicator(i_to-i_from, block_size);
    for (unsigned int i=0; i*block_size < n_rows; ++i) {
      population_count <<< block_rng
                         , block_size
                         , shared_mem >>> (i*block_size
                                         , d_coords
                                         , n_rows
                                         , n_cols
                                         , d_radii2
                                         , n_radii
                                         , d_pops
                                         , i_from
                                         , i_to);
    }
    cudaDeviceSynchronize();
    check_error();
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
    cudaFree(d_radii2);
    cudaFree(d_pops);
    return pops;
  }


/*
  Pops
  calculate_populations_partial(const float* coords
                              , const std::vector<float>& sorted_coords
                              , const std::vector<float>& blimits
                              , std::size_t n_rows
                              , std::size_t n_cols
                              , std::vector<float> radii
                              , std::size_t i_from
                              , std::size_t i_to
                              , int i_gpu) {
    ASSUME_ALIGNED(coords);
    unsigned int n_radii = radii.size();
    // make sure radii are in descending order
    std::sort(radii.begin(), radii.end(), std::greater<float>());
    // setup device & streams
    cudaSetDevice(i_gpu);
    cudaStream_t streams[N_STREAMS];
    for (unsigned int s=0; s < N_STREAMS; ++s) {
      cudaStreamCreate(&streams[s]);
    }
    // copy coords to device
    float* d_coords;
    cudaMalloc((void**) &d_coords
             , sizeof(float) * n_rows * n_cols);
    cudaMemcpy(d_coords
             , coords
             , sizeof(float) * n_rows * n_cols
             , cudaMemcpyHostToDevice);
    float* d_sorted_coords;
    cudaMalloc((void**) &d_sorted_coords
             , sizeof(float) * n_rows * n_cols);
    cudaMemcpy(d_sorted_coords
             , sorted_coords.data()
             , sizeof(float) * n_rows * n_cols
             , cudaMemcpyHostToDevice);
    // copy squared radii to device
    float* d_radii2;
    cudaMalloc((void**) &d_radii2
             , sizeof(float) * n_radii);
    std::vector<float> radii2(radii);
    for (float& r: radii2) {
      r *= r;
    }
    cudaMemcpy(d_radii2
             , radii2.data()
             , sizeof(float) * n_radii
             , cudaMemcpyHostToDevice);
    // tmp buffer for in/out info & reference coords (per stream)
    unsigned int* d_in_radius[N_STREAMS];
    for (unsigned int s=0; s < N_STREAMS; ++s) {
      cudaMalloc((void**) &d_in_radius[s]
               , sizeof(unsigned int) * n_rows * n_radii);
    }
    // result buffer
    unsigned int* d_pops;
    cudaMalloc((void**) &d_pops
             , sizeof(unsigned int) * n_rows * n_radii);
    cudaMemset(d_pops
             , 0
             , sizeof(unsigned int) * n_rows * n_radii);
    // populations per frame
    for (std::size_t i=i_from; i < i_to; ++i) {
      unsigned int i_stream = i % N_STREAMS;
      // prune range for faster computation
      // (using largest radius in first dimension)
      auto min_max_box = Clustering::Tools::min_max_box(blimits
                                                      , coords[i*n_cols]
                                                      , radii[0]);
      unsigned int offset = min_max_box.first * BSIZE;
      unsigned int rng = (min_max_box.second-min_max_box.first+1);
      // +1 for subsequent reduction over otherwise uninitialized range
      in_radius <<< rng+1
                  , BSIZE
                  // shared mem for reference coords
                  , sizeof(float) * n_cols
                  , streams[i_stream] >>> (offset
                                         , d_sorted_coords
                                         , d_coords
                                         , i
                                         , n_rows
                                         , n_cols
                                         , d_radii2
                                         , n_radii
                                         , d_in_radius[i_stream]);
      // compute pops per radius
      if (rng % 2 == 0) {
        rng /= 2;
      } else {
        rng = rng/2 + 1;
      }
      for (unsigned int r=0; r < n_radii; ++r) {
        // pops stored col-wise -> just set an offset ...
        offset = r*n_rows;
        reduce_sum_uint<BSIZE> <<< rng
                                 , BSIZE
                                 , 0
                                 , streams[i_stream] >>> (offset
                                                        , d_in_radius[i_stream]
                                                        , n_rows
                                                        , d_pops
                                                        , r*n_rows + i);
      }
    }
    cudaThreadSynchronize();
    // retrieve pops
    std::vector<unsigned int> tmp_pops(n_rows*n_radii);
    cudaMemcpy(tmp_pops.data()
             , d_pops
             , sizeof(unsigned int) * n_rows * n_radii
             , cudaMemcpyDeviceToHost);
    // sort tmp_pops into pops
    Pops pops;
    for (unsigned int r=0; r < n_radii; ++r) {
      pops[radii[r]].resize(n_rows, 0);
      for (unsigned int i=i_from; i < i_to; ++i) {
        pops[radii[r]][i] = tmp_pops[r*n_rows+i];
      }
    }
    cudaFree(d_sorted_coords);
    cudaFree(d_radii2);
    cudaFree(d_pops);
    for (unsigned int s=0; s < N_STREAMS; ++s) {
      cudaFree(d_in_radius[s]);
    }
    return pops;
  }
*/

  Pops
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii) {
    using Clustering::Tools::dim1_sorted_coords;
    using Clustering::Tools::boxlimits;
    ASSUME_ALIGNED(coords);
    std::sort(radii.begin(), radii.end(), std::greater<float>());
    std::size_t n_radii = radii.size();
    std::vector<float> rad2(n_radii);
    for (std::size_t i=0; i < n_radii; ++i) {
      rad2[i] = radii[i]*radii[i];
    }
    // sort coordinates on first dimension for neighbor pruning
    std::vector<float> sorted_coords = dim1_sorted_coords(coords
                                                        , n_rows
                                                        , n_cols);
    // box limits for pruning
    std::vector<float> blimits = boxlimits(sorted_coords
                                         , BSIZE_POPS
                                         , n_rows
                                         , n_cols);
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPUs found" << std::endl;
      exit(EXIT_FAILURE);
    }
    int gpu_range = n_rows / n_gpus;
    int i;
    std::vector<Pops> partial_pops(n_gpus);
    #pragma omp parallel for default(none)\
      private(i)\
      firstprivate(n_gpus,n_rows,n_cols,gpu_range)\
      shared(partial_pops,radii,coords,sorted_coords,blimits)\
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
    check_error();
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
    check_error();
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
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPUs found" << std::endl;
      exit(EXIT_FAILURE);
    } else {
      Clustering::logger(std::cout) << "running nearest neighbor search on "
                                    << n_gpus << " GPUS" << std::endl;
    }
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

}}} // end Clustering::Density::CUDA

