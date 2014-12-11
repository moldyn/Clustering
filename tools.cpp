
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

