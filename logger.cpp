
#include "logger.hpp"

namespace Clustering {
  bool verbose = false;
  std::ostream devnull(0);

  std::ostream& logger(std::ostream& s) {
    if (verbose) {
      return s;
    } else {
      return devnull; 
    }
  }

  std::ostream& debug() {
    std::cout << "DEBUG: ";
    return std::cout;
  }
} // end namespace Clustering

