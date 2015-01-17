
/*

TODO proper doc explaining memory/computation model

i_block (begin of block)
coords
pops

global work size: full length + fake length to be divisible by 64
  -> reference
  -> ref-line to private memory
  -> index in resulting vector (pops)
local work size:  64
  -> block to compute against
  -> read block into local mem in parallel by workgroup

 */

__kernel void
pops(  const unsigned int i_block_ref
     , const unsigned int i_block
     , const unsigned int n_rows
     , const unsigned int n_cols
     , __global const float * coords
     , const float rad2
     , __global unsigned int * pops
     , __local float * tmp_block) {

  float row_i[32];
  unsigned int i_global = i_block + get_global_id(0);
  unsigned int i_local = get_local_id(0);
  unsigned int i_block_global = i_block_ref + i_local;
  unsigned int n_local = min(get_local_size(0), (n_cols - i_block_ref));
  unsigned int j,k;
  unsigned int tmp_pops = 0;
  float dist2, tmp;
  if (i_global < n_rows) {
    // copy row i_global to local memory for fast retrieval
    for (k=0; k < n_cols; ++k) {
      row_i[k] = coords[i_global*n_cols+k];
    }
    // copy block of reference coords to local memory for fast retrieval
    for (k=0; (k < n_cols) && (i_local < n_local); ++k) {
      tmp_block[i_local*n_cols+k] = coords[i_block_global*n_cols+k];
    }
    // sync workgroup
    barrier(CLK_LOCAL_MEM_FENCE);
    // compute squared distances of i to reference structures
    // in block and compare them to rad2
    for (j=0; j < n_local; ++j) {
      dist2 = 0.0f;
      for (k=0; k < n_cols; ++k) {
        tmp = row_i[k] - tmp_block[j*n_cols+k];
        //dist2 += tmp * tmp;
        dist2 = fma(tmp, tmp, dist2);
      }
      if (dist2 <= rad2) {
        ++tmp_pops;
      }
    }
    pops[i_global] += tmp_pops;
  } else {
    return;
  }
}

