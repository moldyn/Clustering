#pragma once

#include <cuda.h>

/**
 * perform reduction (sum) on given values.
 * stores result to results-array at index i_result.
 *
 * implicit first reduction on call, so run with half the total range
 *
 * begin reduction at offset.
 */
template <unsigned int _BLOCKSIZE>
__global__ void
reduce_sum(unsigned int offset
         , float* vals
         , unsigned int n_vals
         , float* results
         , unsigned int i_result) {
  __shared__ float sum_block[_BLOCKSIZE];
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
    sum_block[tid] = 0.0f;
  }
//TODO ifdef to set intrinsic sync mode for specific architectures (e.g. Tesla K40)
//  for (stride=_BLOCKSIZE/2; stride > 32; stride /= 2) {
//    __syncthreads();
//    if (tid < stride) {
//      sum_block[tid] += sum_block[tid+stride];
//    }
//  }
//  // unroll loop inside warp (intrinsic sync!)
//  __syncthreads();
//  if (tid < 32) {
//    sum_block[tid] += sum_block[tid+32];
//    sum_block[tid] += sum_block[tid+16];
//    sum_block[tid] += sum_block[tid+8];
//    sum_block[tid] += sum_block[tid+4];
//    sum_block[tid] += sum_block[tid+2];
//    sum_block[tid] += sum_block[tid+1];
//  }
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

