
#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>


template <typename NUM>
std::tuple<CoordsPointer<NUM>, std::size_t, std::size_t>
read_coords(std::string filename, std::vector<std::size_t> usecols) {
  std::size_t n_rows=0;
  std::size_t n_cols=0;
  std::size_t n_cols_used=0;

  std::ifstream ifs(filename);
  {
    // determine n_cols
    std::string linebuf;
    while (linebuf.empty() && ifs.good()) {
      std::getline(ifs, linebuf);
    }
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
    // go back again
    ifs.clear();
    ifs.seekg(0, std::ios::beg);
  }
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
  // allocate memory
  CoordsPointer<NUM> coords((NUM*) _mm_malloc(sizeof(NUM)*n_rows*n_cols_used, MEM_ALIGNMENT), CoordsDeleter());
  // just for easy access ...
  NUM* c = coords.get();
  // read data
  for (std::size_t cur_row = 0; cur_row < n_rows; ++cur_row) {
    std::size_t cur_col = 0;
    for (std::size_t i=0; i < n_cols; ++i) {
      NUM buf;
      ifs >> buf;
      if (col_used[i]) {
        c[cur_row*n_cols_used + cur_col] = buf;
        ++cur_col;
      }
    }
  }
  return std::make_tuple(std::move(coords), n_rows, n_cols_used);
}

