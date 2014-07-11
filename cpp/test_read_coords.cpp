
#include "config.hpp"
#include "tools.hpp"

#include <iostream>
#include <iomanip>
#include <tuple>
#include <memory>


int main() {

  std::size_t n_rows, n_cols;
  CoordsPointer<float> c;

  auto t = read_coords<float>("test/coords.dat", {0, 2, 4});
  std::tie(c, n_rows, n_cols) = read_coords<float>("test/coords.dat", {0, 2, 4});

  std::cout << n_rows << " x " << n_cols << std::endl;

  // get easy access
  float* cp = c.get();

  std::cout << std::setprecision(1) << std::fixed;
  for (std::size_t i=0; i < n_rows; ++i) {
    for (std::size_t j=0; j < n_cols; ++j) {
      std::cout << " " << cp[i*n_cols + j];
    }
    std::cout << std::endl;
  }

  return 0;
}

