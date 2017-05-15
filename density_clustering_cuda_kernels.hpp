#pragma once

//// hard-coded settings

// for pops
#define BSIZE_POPS 512

// for neighborhood search
#define BSIZE_NH 128

// for screening
#define BSIZE_SCR 256

////


namespace Clustering {
namespace Density {
namespace CUDA {
namespace Kernel {

  __global__ void
  population_count(unsigned int offset
                 , float* coords
                 , unsigned int n_rows
                 , unsigned int n_cols
                 , float* radii2
                 , unsigned int n_radii
                 , unsigned int* pops
                 , unsigned int i_from
                 , unsigned int i_to);

  __global__ void
  nearest_neighbor_search(unsigned int offset
                        , float* coords
                        , unsigned int n_rows
                        , unsigned int n_cols
                        , float* fe
                        , unsigned int* nh_nhhd_ndx
                        , float* nh_nhhd_dist
                        , unsigned int i_from
                        , unsigned int i_to);


  __global__ void
  screening(unsigned int offset
          , float* sorted_coords
          , unsigned int n_rows
          , unsigned int n_cols
          , float max_dist2
          , unsigned int* clustering
          , unsigned int i_from
          , unsigned int i_to);

}}}} // end Clustering::Density::CUDA::Kernel

