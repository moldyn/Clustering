
#include "tools.hpp"

#include <fstream>
#include <sstream>
#include <iterator>
#include <map>

template <typename NUM>
std::tuple<std::vector<NUM>, std::size_t, std::size_t>
read_coords(std::string filename, std::vector<std::size_t> usecols) {
  std::size_t n_cols=0;
  std::size_t n_cols_used=0;
  std::vector<NUM> coords;

  std::ifstream ifs(filename);
  {
    // load full file with all columns
    std::string linebuf;
    std::getline(ifs, linebuf);
    std::stringstream ss(linebuf);
    n_cols = std::distance(std::istream_iterator<std::string>(ss),
                           std::istream_iterator<std::string>());
    // go back to beginning to read complete file
    ifs.seekg(0);
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

  while (ifs.good()) {
    for (std::size_t i=0; i < n_cols; ++i) {
      NUM buf;
      ifs >> buf;
      if (col_used[i]) {
        coords.push_back(buf);
      }
    }
  }

  return std::make_tuple(coords, coords.size() / n_cols_used, n_cols_used);
}

