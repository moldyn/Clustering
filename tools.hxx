/*
Copyright (c) 2015, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "tools.hpp"
#include "logger.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>
#include <algorithm>

namespace Clustering {
namespace Tools {

template <typename NUM>
std::tuple<NUM*, std::size_t, std::size_t>
read_coords(std::string filename, std::vector<std::size_t> usecols) {
  std::size_t n_rows=0;
  std::size_t n_cols=0;
  std::size_t n_cols_used=0;
  std::ifstream ifs(filename);
  Clustering::logger(std::cout) << "~~~ reading coordinates" << std::endl;
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  }
  Clustering::logger(std::cout) << "    from file: " << filename << std::endl;
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
  Clustering::logger(std::cout) << "    with dimesions: " << n_rows << "x"
                                << n_cols << "\n" << std::endl;
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

template <typename NUM>
std::vector<NUM>
dim1_sorted_coords(const NUM* coords
                 , std::size_t n_rows
                 , std::size_t n_cols) {
  std::vector<NUM> sorted_coords(n_rows*n_cols);
  if (n_cols == 1) {
    // directly sort on data if just one column
    for (std::size_t i=0; i < n_rows; ++i) {
      sorted_coords[i] = coords[i];
    }
    std::sort(sorted_coords.begin(), sorted_coords.end());
  } else {
    std::vector<std::vector<NUM>> c_tmp(n_rows
                                      , std::vector<float>(n_cols));
    for (std::size_t i=0; i < n_rows; ++i) {
      for (std::size_t j=0; j < n_cols; ++j) {
        c_tmp[i][j] = coords[i*n_cols+j];
      }
    }
    // sort on first index
    std::sort(c_tmp.begin()
            , c_tmp.end()
            , [] (const std::vector<NUM>& lhs
                , const std::vector<NUM>& rhs) {
                return lhs[0] < rhs[0];
              });
    // feed sorted data into 1D-array
    for (std::size_t i=0; i < n_rows; ++i) {
      for (std::size_t j=0; j < n_cols; ++j) {
        sorted_coords[i*n_cols+j] = c_tmp[i][j];
      }
    }
  }
  return sorted_coords;
}

template <typename NUM>
std::vector<NUM>
boxlimits(const std::vector<NUM>& xs
        , std::size_t boxsize
        , std::size_t n_rows
        , std::size_t n_cols) {
  //std::size_t n_xs = xs.size() / n_dim;
  std::size_t n_boxes = n_rows / boxsize;
  if (n_boxes * boxsize < n_rows) {
    ++n_boxes;
  }
  std::vector<NUM> boxlimits(n_boxes);
  for (std::size_t i=0; i < n_boxes; ++i) {
    // split into boxes on 1st dimension
    // (i.e. col-index == 0)
    boxlimits[i] = xs[i*boxsize*n_cols];
  }
  return boxlimits;
}

template <typename NUM>
std::pair<std::size_t, std::size_t>
min_max_box(const std::vector<NUM>& limits
          , NUM val
          , NUM radius) {
  std::size_t n_boxes = limits.size();
  if (n_boxes == 0) {
    return {0,0};
  } else {
    std::size_t i_min = n_boxes - 1;
    std::size_t i_max = 0;
    NUM lbound = val - radius;
    NUM ubound = val + radius;
    for (std::size_t i=1; i < n_boxes; ++i) {
      if (lbound < limits[i]) {
        i_min = i-1;
        break;
      }
    }
    for (std::size_t i=n_boxes; 0 < i; --i) {
      if (limits[i-1] < ubound) {
        i_max = i-1;
        break;
      }
    }
    return {i_min, i_max};
  }
}


template <typename KEY, typename VAL>
void
write_map(std::string filename, std::map<KEY, VAL> mapping,
          std::string header_comment, bool val_then_key) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << header_comment;
  if (val_then_key) {
    for (auto key_val: mapping) {
      ofs << key_val.second << " " << key_val.first << "\n";
    }
  } else {
    for (auto key_val: mapping) {
      ofs << key_val.first << " " << key_val.second << "\n";
    }
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
    while (!ifs.eof() && !ifs.bad()) {
      NUM buf;
      ifs >> buf;
      if ( ! ifs.fail()) {
        dat.push_back(buf);
      } else {  // if conversion error, skip (comment) line
        ifs.clear();
        ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
    }
  }
  if (dat.empty()) {
    std::cerr << "error: opened empty file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  }
  return dat;
}


template <typename NUM>
void
write_single_column(std::string filename, std::vector<NUM> dat,
                    std::string header_comment, bool with_scientific_format) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << header_comment;
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

template <typename T>
std::vector<T>
unique_elements(std::vector<T> xs) {
  std::sort(xs.begin()
          , xs.end());
  auto last = std::unique(xs.begin()
                        , xs.end());
  xs.erase(last
         , xs.end());
  return xs;
}


} // end namespace Tools
} // end namespace Clustering

