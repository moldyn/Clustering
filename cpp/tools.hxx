
#include "tools.hpp"

#include <fstream>
#include <sstream>
#include <iterator>
#include <map>



/// aligned memory allocation for Xeon Phi, SSE or AVX
 
//#if defined(__INTEL_COMPILER)
//#include <malloc.h>
//#else
//#include <mm_malloc.h>
//#endif // defined(__GNUC__)



template <typename NUM>
std::tuple<std::vector<NUM>, std::size_t, std::size_t>
read_coords(std::string filename, std::vector<std::size_t> usecols) {
  std::size_t n_rows=0;
  std::size_t n_cols=0;
  std::size_t n_cols_used=0;

  std::ifstream ifs(filename);
  {
    // determine n_cols
    std::string linebuf;
    std::getline(ifs, linebuf);
    std::stringstream ss(linebuf);
    n_cols = std::distance(std::istream_iterator<std::string>(ss),
                           std::istream_iterator<std::string>());
    // go back to beginning to read complete file
    ifs.seekg(0);
    // determine n_rows
    while (ifs.good()) {
      std::getline(ifs, linebuf);
      if ( ! linebuf.empty()) {
        ++n_rows;
      }
    }
    //TODO: check if this works after !ifs.good()
    // go back again
    ifs.seekg(0);
  }

  // allocate memory
  std::vector<NUM> coords(n_rows * n_cols);

  std::map<std::size_t, bool> col_used;
  if (usecols.size() == 0) {
    // use all columns
    n_cols_used = n_cols;
    for (std::size_t i=0; i < n_cols; ++i) {
      col_used[i] = true;
    }
  } else {
    // use only defined columns
    n_cols_used = usecols.size();
    for (std::size_t i=0; i < n_cols; ++i) {
      col_used[i] = false;
    }
    for (std::size_t i: usecols) {
      col_used[i] = true;
    }
  }

  for (std::size_t cur_row = 0; cur_row < n_rows; ++cur_row) {
    std::size_t cur_col = 0;
    for (std::size_t i=0; i < n_cols; ++i) {
      NUM buf;
      ifs >> buf;
      if (col_used[i]) {
        coords[cur_row*n_cols + cur_col] = buf;
        ++cur_col;
      }
    }
  }

  return std::make_tuple(coords, n_rows, n_cols_used);
}

