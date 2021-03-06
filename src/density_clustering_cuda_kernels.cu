
#include "density_clustering_cuda_kernels.hpp"

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
                        , unsigned int* nh_nhhd_ndx
                        , float* nh_nhhd_dist
                        , unsigned int i_from
                        , unsigned int i_to) {
    // grid ids
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int gid = bid * bsize + tid + i_from;
    // free energy and local id for this frame
    float ref_fe;
    unsigned int ref_id;
    // shared memory for fast free energy / coordinate retrieval
    int comp_size = min(bsize, n_rows - offset);
    extern __shared__ float smem[];
    float* smem_fe = (float*) smem;
    float* smem_coords = (float*) &smem[comp_size];
    // load data of current block to compare with into shared memory 
    if (tid < comp_size) {
      for (unsigned int j=0; j < n_cols; ++j) {
        smem_coords[tid*n_cols+j] = coords[(tid+offset)*n_cols+j];
      }
      smem_fe[tid] = fe[tid+offset];
    }
    __syncthreads();
    if (gid < i_to) {
      ref_id = tid+bsize;
      // load reference coordinates for re-use into shared memory
      for (unsigned int j=0; j < n_cols; ++j) {
        smem_coords[ref_id*n_cols+j] = coords[gid*n_cols+j];
      }
      // reference free energy
      ref_fe = fe[gid];
      // load current best mindists for this frame into registers
      float nh_mindist = nh_nhhd_dist[gid];
      float nhhd_mindist = nh_nhhd_dist[gid+n_rows];
      unsigned int nh_minndx = nh_nhhd_ndx[gid];
      unsigned int nhhd_minndx = nh_nhhd_ndx[gid+n_rows];
      // compare squared distances of reference
      // to (other) frames in shared mem
      for (unsigned int i=0; i < comp_size; ++i) {
        float dist2=0.0f;
        for (unsigned int j=0; j < n_cols; ++j) {
          float c = smem_coords[ref_id*n_cols+j] - smem_coords[i*n_cols+j];
          dist2 = fma(c, c, dist2);
        }
        // frame with min distance (i.e. nearest neighbor)
        if ((nh_mindist == 0)
         || (dist2 < nh_mindist && dist2 != 0)) {
          nh_mindist = dist2;
          nh_minndx = i+offset;
        }
        // frame with min distance and lower energy
        if ((nhhd_mindist == 0 && smem_fe[i] < ref_fe)
         || (dist2 < nhhd_mindist
          && smem_fe[i] < ref_fe && dist2 != 0)) {
          nhhd_mindist = dist2;
          nhhd_minndx = i+offset;
        }
      }
      // write results (ndx & dist) to global buffers
      nh_nhhd_ndx[gid] = nh_minndx;
      nh_nhhd_ndx[n_rows+gid] = nhhd_minndx;
      nh_nhhd_dist[gid] = nh_mindist;
      nh_nhhd_dist[n_rows+gid] = nhhd_mindist;
    }
  }

  __global__ void
  screening(unsigned int offset
          , float* sorted_coords
          , unsigned int n_rows
          , unsigned int n_cols
          , float max_dist2
          , unsigned int* clustering
          , unsigned int i_from
          , unsigned int i_to) {
    // dynamic shared mem for ref coords
    extern __shared__ float smem_coords[];
    __shared__ unsigned int smem_states[BSIZE_SCR];
    // thread dimensions
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int gid = bid * bsize + tid + i_from;
    int comp_size = min(bsize, n_rows - offset);
    if (tid < comp_size) {
      // load reference coordinates to cache
      for (unsigned int j=0; j < n_cols; ++j) {
        smem_coords[tid*n_cols+j] = sorted_coords[(tid+offset)*n_cols+j];
      }
      // load reference state information to cache
      smem_states[tid] = clustering[tid+offset];
    }
    __syncthreads();
    if (gid < i_to) {
      for (unsigned int j=0; j < n_cols; ++j) {
        // load coordinates of current frame for re-use into shared memory
        smem_coords[(tid+bsize)*n_cols+j] = sorted_coords[gid*n_cols+j];
      }
      // load previous state information to cache
      unsigned int tmp_result = clustering[gid];
      // compare current frame (tid) against reference block (k)
      for (unsigned int k=0; k < comp_size; ++k) {
        unsigned int tmp_state = smem_states[k];
        if (tmp_state != tmp_result) {
          float dist2 = 0.0f;
          for (unsigned int j=0; j < n_cols; ++j) {
            float c = smem_coords[(tid+bsize)*n_cols+j]
                    - smem_coords[          k*n_cols+j];
            dist2 = fma(c, c, dist2);
          }
          // update intermediate results
          if (dist2 < max_dist2) {
            tmp_result = min(tmp_state
                           , tmp_result);
            smem_states[k] = tmp_result;
          }
        }
      }
      clustering[gid] = tmp_result;
    }
    __syncthreads();
    // update reference states
    if (tid < comp_size) {
      atomicMin(&clustering[tid+offset]
              , smem_states[tid]);
    }
  }

}}}} // end Clustering::Density::CUDA::Kernel

