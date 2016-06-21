
#include "tools.hpp"
#include "density_clustering_cuda.hpp"

#include <algorithm>

#include <cuda.h>
#include <omp.h>


#define WGSIZE 128
#define N_STREAMS 2

namespace Clustering {
namespace Density {
namespace CUDA {

  typedef std::map<float, std::vector<std::size_t>> Pops;

  Pops
  calculate_populations_partial(const std::vector<float>& sorted_coords
                              , std::size_t n_rows
                              , std::size_t n_cols
                              , std::vector<float> radii
                              , std::size_t i_from
                              , std::size_t i_to
                              , int i_gpu) {
    //TODO buffers (n_rows) for every radius if 1 for 'in radius', else 0
    //TODO result buffers (n_rows) per radius

    //TODO loop over frames:
    //     - set ref
    //     - compute dist2
    //     - set buffers / radius
    //     - reduce on buffers

    //TODO retrieve pops
  }

  Pops
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii) {
    ASSUME_ALIGNED(coords);
    using namespace Clustering::Tools;
    Pops pops;
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
                                                , WGSIZE
                                                , n_cols);
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
      std::cerr << "error: no CUDA-compatible GPUs found" << std::endl;
      exit(EXIT_FAILURE);
    }
    int gpu_range = n_rows / n_gpus;
    //TODO OMP threads
    for (int i=0; i < n_gpus; ++i) {
      //TODO separate n_rows into equal chunks
      //TODO run chunks on different GPUs
    }
    //TODO combine pops


    return pops;
  }

}}} // end Clustering::Density::CUDA

