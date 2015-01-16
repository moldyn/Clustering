
__kernel void
pops(const unsigned int n_rows,
     const unsigned int n_cols,
     __global const float* coords,
     const float rad2,
     __global unsigned int* pops) {
  float row_i[20];
  int k,j;
  unsigned int pop_i = 0;
  float dist2, tmp_d;
  int i = get_global_id(0);
  for (k=0; k < n_cols; ++k) {
    row_i[k] = coords[i*n_cols+k];
  }
  for (j=0; j < n_rows; ++j) {
    dist2 = 0.0f;
    for (k=0; k < n_cols; ++k) {
      tmp_d = row_i[k] - coords[j*n_cols+k];
      dist2 += tmp_d * tmp_d;
    }
    pop_i += (unsigned int) (dist2 <= rad2);
  }
  pops[i] = pop_i;
}

