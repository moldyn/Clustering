
#include "tools.hpp"
#include "density_clustering_cuda.hpp"

#include <algorithm>

#include <cuda.h>
#include <omp.h>

#include "lts_cuda_kernels.cuh"


#define BSIZE 128
#define N_STREAMS 1

namespace Clustering {
namespace Density {
namespace CUDA {

  void check_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
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
          , float* in_radius) {
    //TODO store ref in local group?
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid*BSIZE+tid;
    float c;
    float dist2 = 0.0f;
    unsigned int j,r;
if (gid == 0) {
  for (j=0; j < n_cols; ++j){
    printf("### %f\n", sorted_coords[j*n_rows+i_ref]);
  }
}
    if (gid+offset < n_rows) {
      // compute squared dist
      for (j=0; j < n_cols; ++j) {
        c = coords[j*n_rows+i_ref] - sorted_coords[j*n_rows + gid + offset];
        dist2 = fma(c, c, dist2);
      }
      // write results: 1.0 if in radius, 0.0 if not
      for (r=0; r < n_radii; ++r) {
//TODO
        if (dist2 <= radii2[j]) {
          in_radius[r*n_rows + gid] = dist2;
        } else {
          in_radius[r*n_rows + gid] = dist2;
        }
      }
    }
  }

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

    for (int ii=0; ii < n_cols; ++ii) {
      std::cerr << "@ " << coords[ii*n_rows] << std::endl;
      std::cerr << "@s " << sorted_coords[ii*n_rows] << std::endl;
    }


    unsigned int n_rows_ext = Tools::min_multiplicator(n_rows, BSIZE) * BSIZE;
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
    float* d_in_radius[N_STREAMS];
    for (unsigned int s=0; s < N_STREAMS; ++s) {
      cudaMalloc((void**) &d_in_radius[s]
               , sizeof(float) * n_rows * n_radii);
    }
    // result buffer
    float* d_pops;
    cudaMalloc((void**) &d_pops
             , sizeof(float) * n_rows * n_radii);
    cudaMemset(d_pops
             , 0
             , sizeof(float) * n_rows * n_radii);
    // populations per frame
    for (std::size_t i=i_from; i < i_to; ++i) {
      unsigned int i_stream = i % N_STREAMS;
      //  initialize d_in_radius to 0 -> per default not in hypersphere
      cudaMemset(d_in_radius[i_stream]
               , 0
               , sizeof(float) * n_rows * n_radii);
      // prune range for faster computation
      // (using largest radius in first dimension)
//      auto min_max_box = Clustering::Tools::min_max_box(blimits
//                                                      , coords[i]
//                                                      , radii[0]);
//
//      unsigned int offset = min_max_box.first * BSIZE;
//      unsigned int rng = (min_max_box.second-min_max_box.first+1);


      //TODO
      unsigned int offset = 0;
      unsigned int rng = n_rows_ext/BSIZE;

      in_radius <<< rng
                  , BSIZE
                  , 0
                  , streams[i_stream] >>> (offset
                                         , d_sorted_coords
                                         , d_coords
                                         , i
                                         , n_rows
                                         , n_cols
                                         , d_radii2
                                         , n_radii
                                         , d_in_radius[i_stream]);
      //TODO
      check_error();

      cudaDeviceSynchronize();
      if (i == 0) {
        std::vector<float> tmp_in_rad(n_radii*n_rows);
        cudaMemcpy(tmp_in_rad.data()
                 , d_in_radius[i_stream]
                 , sizeof(float) * n_rows * n_radii
                 , cudaMemcpyDeviceToHost);
        for (auto f: tmp_in_rad) {
          std::cout << f << std::endl;
        }
        exit(EXIT_FAILURE);
      }





      // compute pops per radius
      for (unsigned int r=0; r < n_radii; ++r) {
        // pops stored col-wise -> just set an offset ...
        offset = r*n_rows;
        //TODO stupid: don't run over all rows for reduction, use boxlimits!
        reduce_sum<BSIZE> <<< n_rows_ext/BSIZE
                            , BSIZE
                            , 0
                            , streams[i_stream] >>> (offset
                                                   , d_in_radius[i_stream]
                                                   , n_rows
                                                   , d_pops
                                                   , r*n_rows + i);
        //TODO
        check_error();
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

  Pops
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii) {
//    ASSUME_ALIGNED(coords);
    for (int ii=0; ii < n_cols; ++ii) {
      std::cerr << "@@ " << coords[ii*n_rows] << std::endl;
    }




    std::sort(radii.begin(), radii.end(), std::greater<float>());
    std::size_t n_radii = radii.size();
    std::vector<float> rad2(n_radii);
    for (std::size_t i=0; i < n_radii; ++i) {
      rad2[i] = radii[i]*radii[i];
    }
    // sort coordinates on first dimension for neighbor pruning
    std::vector<float> sorted_coords = Clustering::Tools::dim1_sorted_coords(coords
                                                                           , n_rows
                                                                           , n_cols);
    // box limits for pruning
    std::vector<float> blimits = Clustering::Tools::boxlimits(sorted_coords
                                                            , BSIZE
                                                            , n_cols);
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPUs found" << std::endl;
      exit(EXIT_FAILURE);
    }

//TODO
    n_gpus = 1;


    int gpu_range = n_rows / n_gpus;
    int i;
    std::vector<Pops> partial_pops(n_gpus);
    #pragma omp parallel for default(none)\
                             private(i)\
                             firstprivate(n_gpus,n_rows,n_cols,gpu_range)\
                             shared(partial_pops,radii,coords,sorted_coords,blimits)\
                             num_threads(1)
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
        for (unsigned int i_gpu=0; i_gpu < n_gpus; ++i_gpu) {
          pops[r][i] += partial_pops[i_gpu][r][i];
        }
      }
    }
    return pops;
  }

}}} // end Clustering::Density::CUDA

