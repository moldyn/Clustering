
#include "tools.hpp"

#include <tuple>


int main() {

  std::vector<float> c;
  std::size_t n_rows;
  std::size_t n_cols;

  std::tie(c, n_rows, n_cols) = read_coords("test.xyz", usecols = {0, 2, 4});

  for (std::size_t i=0; i < n_rows; ++i) {
    for (std::size_t j=0; j < n_cols; ++j) {
      std::cout << " " << c[i*n_cols + j];
    }
    std::cout << std::endl;
  }

  return 0;
}

