
#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>

namespace Clustering {
namespace Tools {

template <typename NUM>
std::tuple<NUM*, std::size_t, std::size_t>
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
  // DC_MEM_ALIGNMENT is defined during cmake and
  // set depending on usage of SSE2, SSE4_1, AVX or Xeon Phi
  NUM* coords = (NUM*) _mm_malloc(sizeof(NUM)*n_rows*n_cols_used, DC_MEM_ALIGNMENT);
  ASSUME_ALIGNED(coords);
  // read data
  for (std::size_t cur_row = 0; cur_row < n_rows; ++cur_row) {
    std::size_t cur_col = 0;
    for (std::size_t i=0; i < n_cols; ++i) {
      NUM buf;
      ifs >> buf;
      if (col_used[i]) {
        coords[cur_row*n_cols_used + cur_col] = buf;
        ++cur_col;
      }
    }
  }
  return std::make_tuple(coords, n_rows, n_cols_used);
}


template <typename NUM>
void
free_coords(NUM* coords) {
  _mm_free(coords);
}

template <typename KEY, typename VAL>
void
write_map(std::string filename, std::map<KEY, VAL> mapping) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  for (auto key_val: mapping) {
    ofs << key_val.first << " " << key_val.second << "\n";
  }
}

template <typename NUM>
std::vector<NUM>
read_single_column(std::string filename) {
  std::vector<NUM> dat;
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    while (ifs.good()) {
      NUM buf;
      ifs >> buf;
      if ( ! ifs.fail()) {
        dat.push_back(buf);
      }
    }
  }
  return dat;
}


template <typename NUM>
void
write_single_column(std::string filename, std::vector<NUM> dat, bool with_scientific_format) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (with_scientific_format) {
    ofs << std::scientific;
  }
  for (NUM i: dat) {
    ofs << i << "\n";
  }
}

template <typename NUM>
NUM
string_to_num(const std::string &s) {
  std::stringstream ss(s);
  NUM buf;
  ss >> buf;
  return buf;
}

} // end namespace Tools
} // end namespace Clustering

