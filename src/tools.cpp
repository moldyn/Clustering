/*
Copyright (c) 2015-2017, Florian Sittel (www.lettis.net)
Copyright (c) 2018-2020, Daniel Nagel
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

#include <cmath>
#include <stdarg.h>

namespace Clustering {
namespace Tools {

unsigned int
min_multiplicator(unsigned int orig
                , unsigned int mult) {
  return (unsigned int) std::ceil(orig / ((float) mult));
};

void
write_fes(std::string filename, std::vector<float> fes, std::string header_comment,
          std::map<std::string,float> stringMap) {
  append_commentsMap(header_comment, stringMap);
  header_comment.append("#\n# free energy of each frame\n");
  write_single_column<float>(filename, fes, header_comment, true);
}

void
write_pops(std::string filename, std::vector<std::size_t> pops, std::string header_comment,
           std::map<std::string,float> stringMap) {
  append_commentsMap(header_comment, stringMap);
  header_comment.append("#\n# point density of each frame\n");
  write_single_column<std::size_t>(filename, pops, header_comment, false);
}

std::vector<std::size_t>
read_clustered_trajectory(std::string filename) {
  return read_single_column<std::size_t>(filename);
}

void
write_clustered_trajectory(std::string filename, std::vector<std::size_t> traj,
                           std::string header_comment, std::map<std::string,float> stringMap) {
  append_commentsMap(header_comment, stringMap);
  header_comment.append("#\n# state/cluster id frames are assigned to\n");
  write_single_column<std::size_t>(filename, traj, header_comment, false);
}

//// from: https://github.com/lettis/Kubix
/**
behaves like sprintf(char*, ...), but with c++ strings and returns the result

\param str pattern to be printed to
\return resulting string
The function internally calls sprintf, but converts the result to a c++ string and returns that one.
Problems of memory allocation are taken care of automatically.
*/
std::string
stringprintf(const std::string& str, ...) {
  unsigned int size = 256;
  va_list args;
  char* buf = (char*) malloc(size * sizeof(char));
  va_start(args, str);
  while (size <= (unsigned int) vsnprintf(buf, size, str.c_str(), args)) {
    size *= 2;
    buf = (char*) realloc(buf, size * sizeof(char));
  }
  va_end(args);
  std::string result(buf);
  free(buf);
  return result;
}

std::vector<float>
read_free_energies(std::string filename) {
  return read_single_column<float>(filename);
}

std::pair<Neighborhood, Neighborhood>
read_neighborhood(const std::string filename) {
  Neighborhood nh;
  Neighborhood nh_high_dens;
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for reading." << std::endl;
    exit(EXIT_FAILURE);
  } else {
    std::size_t i=0;
    while (!ifs.eof() && !ifs.bad()) {
      std::size_t buf1;
      float buf2;
      std::size_t buf3;
      float buf4;
      ifs >> buf1;
      ifs >> buf2;
      ifs >> buf3;
      ifs >> buf4;
      if ( ! ifs.fail()) {
        nh[i] = std::pair<std::size_t, float>(buf1, buf2);
        nh_high_dens[i] = std::pair<std::size_t, float>(buf3, buf4);
        ++i;
      } else {  // if conversion error, skip (comment) line
        ifs.clear();
        ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
    }
  }
  return {nh, nh_high_dens};
}

std::vector<std::size_t>
read_concat_limits(std::string filename) {
  std::vector<std::size_t> concat_limits = read_single_column<std::size_t>(filename);

  for (std::size_t i=0; i+1 < concat_limits.size(); ++i){
    concat_limits[i+1] += concat_limits[i];
  }

  return concat_limits;
}

void
write_neighborhood(const std::string filename,
                   const Neighborhood& nh,
                   const Neighborhood& nh_high_dens,
                   std::string header_comment,
                   std::map<std::string,float> stringMap) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "' for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  append_commentsMap(header_comment, stringMap);
  header_comment.append("#\n# column definitions:\n"
                        "#        nn = nearest neighbor\n"
                        "#     nn_hd = nearest neighbor with higher density\n"
                        "#     id(i) = id/line number of i\n"
                        "#   dsqr(i) = squared euclidean distance to i\n#\n"
                        "# id(nn)  dsqr(nn) id(nn_hd) dsqr(nn_hd)\n");
  ofs << header_comment;
  auto p = nh.begin();
  auto p_hd = nh_high_dens.begin();
  while (p != nh.end() && p_hd != nh_high_dens.end()) {
    // first: key (not used)
    // second: neighbor
    // second.first: id; second.second: squared dist
    ofs << p->second.first    << " " << p->second.second    << " "
        << p_hd->second.first << " " << p_hd->second.second << "\n";
    ++p;
    ++p_hd;
  }
}

std::map<std::size_t, std::size_t>
microstate_populations(std::vector<std::size_t> traj) {
  std::map<std::size_t, std::size_t> populations;
  for (std::size_t state: traj) {
    if (populations.count(state) == 0) {
      populations[state] = 1;
    } else {
      ++populations[state];
    }
  }
  return populations;
}

void
check_concat_limits(std::vector<std::size_t> concat_limits, std::size_t n_frames) {
  if (concat_limits.back() < n_frames) {
    Clustering::logger(std::cout) << "warning: last " << n_frames - concat_limits.back()
                                  << " frames are ignored. check concat-limits/nframes"
                                  << std::endl;
  }
  if (concat_limits.front() == 0) {
    Clustering::logger(std::cout) << "warning: first trajectory is of zero length. check\n"
                                  << "         help for correct usage of --concat-limits"
                                  << std::endl;
  }
  if (concat_limits.back() > n_frames) {
    Clustering::logger(std::cout) << "warning: limits are larger than the file length.\n"
                                  << "         Check your limits!" <<std::endl;
  }
}

float
read_next_float(std::ifstream &ifs) {
  std::string s;
  if (ifs.peek() != '\n') {
    ifs >> std::ws;
    while (!std::isdigit(ifs.peek()) && ifs.peek() != '\n') {
      ifs >> s;
      if (ifs.peek() != '\n') {
        ifs >> std::ws;
      }
    }
  }
  float val;
  // catch if line end was case
  if (ifs.peek() == '\n') {
    val = -1.;
  } else {
    ifs >> val;
  }
  return val;
}

void
read_comments(std::string filename, std::map<std::string,float> &stringMap) {
  std::vector<std::size_t> dat;
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    while (!ifs.eof() && !ifs.bad()) {
      std::size_t buf;
      ifs >> buf;
      if (ifs.fail()) {
        ifs.clear();
        std::string s;
        ifs >> s;
        if (s == "#@" && ifs.peek() != '\n') {
          ifs >> s;
          for (auto & x : stringMap)
          {
            if (s == x.first) {
              float val = read_next_float(ifs);
              if ((x.second != 0) && (std::abs(x.second-val) > 0.001)) {
                Clustering::logger(std::cout) << "warning: the values of " << x.first
                                              << " are not in agreement\n"
                                              << "        " << val << " vs. "
                                              << x.second<< std::endl;
              }
              stringMap[x.first] = val;
            }
          }
        }
        ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
    }
  }
  return;
}

void
append_commentsMap(std::string &header_comment, std::map<std::string,float> &stringMap) {
  header_comment.append("#\n# The following comments are reused for identifying\n# user-based mistakes and should not be modified.\n");
  for (auto &x : stringMap)
  {
    if (x.second != 0.) {
      header_comment.append(stringprintf("#@   %s = %.5f\n", x.first.c_str(), x.second));
    }
  }
  return;
}


} // end namespace Tools
} // end namespace Clustering
