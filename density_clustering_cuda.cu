
#include "tools.hpp"
#include "density_clustering_cuda.hpp"

#include <algorithm>

#include <cuda.h>
#include <omp.h>

#include "lts_cuda_kernels.cuh"


#define BSIZE 128
#define N_STREAMS 2

namespace Clustering {
namespace Density {
namespace CUDA {

  __global__ void
  in_radius(float* coords
          , float* coords_ref
          , unsigned int n_rows
          , unsigned int n_cols
          , float* radii2
          , unsigned int n_radii
          , float* in_radius) {
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid*BSIZE+tid;
    float c;
    float rad2 = 0.0f;
    unsigned int j;
    if (gid < n_rows) {
      // compute rad2
      for (j=0; j < n_cols; ++j) {
        c = coords_ref[j] - coords[j*n_rows + gid];
        rad2 = fma(c, c, rad2);
      }
      // write results: 1.0 if in radius, 0.0 if not
      for (j=0; j < n_radii; ++j) {
        if (rad2 <= radii2[j]) {
          in_radius[j*n_rows + gid] = 1.0f;
        } else {
          in_radius[j*n_rows + gid] = 0.0f;
        }
      }
    }
  }

  //              radius  ->   pops
  typedef std::map<float, std::vector<std::size_t>> Pops;

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
    unsigned int n_rows_ext = Tools::min_multiplicator(n_rows, BSIZE) * BSIZE;
    unsigned int n_radii = radii.size();
    // make sure radii are in descending order
    std::sort(radii.begin()
            , radii.end()
            , [](float lhs, float rhs) -> bool {
                return lhs > rhs;
              });
    // setup device & streams
    cudaSetDevice(i_gpu);
    cudaStream_t streams[N_STREAMS];
    for (unsigned int s=0; s < N_STREAMS; ++s) {
      cudaStreamCreate(&streams[i]);
    }
    // copy coords to device
    float* d_sorted_coords;
    cudaMalloc((void**) &d_sorted_coords
             , sizeof(float) * n_rows_ext * n_cols);
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
    float* d_in_radius[N_STREAMS];
    float* d_coords_ref[N_STREAMS];
    for (unsigned int s=0; s < N_STREAMS; ++s) {
      cudaMalloc((void**) &d_in_radius[s]
               , sizeof(float) * n_rows * n_radii);
      cudaMalloc((void**) &d_coords_ref[s]
               , sizeof(float) * n_cols);
    }
    // result buffer
    float* d_pops;
    cudaMalloc((void**) &d_pops
             , sizeof(float) * n_rows * n_radii);
    // populations per frame
    for (std::size_t i=i_from; i < i_to; ++i) {
      std::vector<float> coords_ref(n_cols);
      for (std::size_t j=0; j < n_cols; ++j) {
        coords_ref[j] = coords[j*n_rows+i];
      }
      unsigned int i_stream = i % N_STREAMS;
      cudaMemcpy(d_coords_ref[i_stream]
               , coords_ref.data()
               , sizeof(float) * n_cols
               , cudaMemcpyHostToDevice);
      // prune range for faster computation
      auto min_max_box = Tools::Clustering::min_max_box(blimits
                                                      , coords_ref[0]
                                                      , radii[0]);
      unsigned int offset = min_max_box.first * BSIZE;
      unsigned int rng = (min_max_box.second-min_max_box.first+1) * BSIZE;
      in_radius <<< rng
                  , BSIZE
                  , 0
                  , streams[i_stream] >>> (offset
                                         , d_sorted_coords
                                         , d_coords_ref
                                         , n_rows
                                         , n_cols
                                         , d_radii2
                                         , n_radii
                                         , d_in_radius);
      // compute pops per radius
      for (unsigned int r=0; r < n_radii; ++r) {
        // pops stored col-wise -> just set an offset ...
        offset = r*n_rows;
        reduce_sum<BSIZE> <<< n_rows_ext
                            , BSIZE
                            , 0
                            , streams[i_stream] >>> (offset
                                                   , d_in_radius
                                                   , n_rows
                                                   , d_pops
                                                   , r*n_rows + i);
      }
    }
    cudaThreadSynchronize();
    // retrieve pops
    std::vector<float> tmp_pops(n_rows*n_radii);
    cudaMemcpy(tmp_pops.data()
             , d_pops
             , sizeof(float) * n_rows * n_radii
             , cudaMemcpyDeviceToHost);
    // sort tmp_pops into pops
    Pops pops;
    for (unsigned int j=0; j < n_radii; ++j) {
      pops[radii[j]].resize(n_rows);
      for (unsigned int i=0; i < n_rows; ++i) {
        pops[radii[j]][i] = tmp_pops[j*n_rows+i];
    }
    cudaFree(d_sorted_coords);
    cudaFree(d_radii2);
    cudaFree(d_pops);
    for (unsigned int s=0; s < N_STREAMS; ++s) {
      cudaFree(d_in_radius[s]);
      cudaFree(d_coords_ref[s]);
    }
    return pops;
  }

  Pops
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii) {
    ASSUME_ALIGNED(coords);
    using namespace Clustering::Tools;
    for (float rad: radii) {
      pops[rad].resize(n_rows, 1);
    }
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
    std::vector<float> blimits = Tools::boxlimits(sorted_coords
                                                , BSIZE
                                                , n_cols);
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPUs found" << std::endl;
      exit(EXIT_FAILURE);
    }
    int gpu_range = n_rows / n_gpus;
    int i,j;
    std::vector<Pops> partial_pops(n_gpus);
    #pragma omp parallel for default(none)\
                             private(i)\
                             firstprivate(n_gpus,n_rows,n_cols,gpu_range)\
                             shared(radii,coords,sorted_coords)\
                             num_threads(n_gpus)
    for (i=0; i < n_gpus; ++i) {
      // compute partial populations in parallel
      // on all available GPUs
      partial_pops[i] = calculate_populations_partial(coords
                                                    , sorted_coords
                                                    , blimits
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
        for (j=0; j < n_gpus; ++j) {
          pops[r][i] += partial_pops[j][r][i];
        }
      }
    }
    return pops;
  }

}}} // end Clustering::Density::CUDA

