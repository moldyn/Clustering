
#include "tools.hpp"

#include <stdarg.h>

namespace Clustering {
namespace Tools {

void
write_fes(std::string fname, std::vector<float> fes) {
  std::ofstream ofs(fname);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << fname << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    ofs << std::scientific;
    for (float f: fes) {
      ofs << f << "\n";
    }
  }
}

void
write_pops(std::string fname, std::vector<std::size_t> pops) {
  // technically the same ...
  write_clustered_trajectory(fname, pops);
}

std::vector<std::size_t>
read_clustered_trajectory(std::string filename) {
  std::vector<std::size_t> traj;
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    while (ifs.good()) {
      std::size_t buf;
      ifs >> buf;
      if ( ! ifs.fail()) {
        traj.push_back(buf);
      }
    }
  }
  return traj;
}

void
write_clustered_trajectory(std::string filename, std::vector<std::size_t> traj) {
  std::ofstream ofs(filename);
  if (ofs.fail()) {
    std::cerr << "error: cannot open file '" << filename << "'" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    for (std::size_t c: traj) {
      ofs << c << "\n";
    }
  }
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
read_neighborhood(const std::string fname) {
  Neighborhood nh;
  Neighborhood nh_high_dens;
  std::ifstream ifs(fname);
  if (ifs.fail()) {
    std::cerr << "error: cannot open file '" << fname << "' for reading." << std::endl;
    exit(EXIT_FAILURE);
  } else {
    std::size_t i=0;
    while (ifs.good()) {
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
      }
    }
  }
  return {nh, nh_high_dens};
}

void
write_neighborhood(const std::string fname,
                   const Neighborhood& nh,
                   const Neighborhood& nh_high_dens) {
  std::ofstream ofs(fname);
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

} // end namespace Tools
} // end namespace Clustering

