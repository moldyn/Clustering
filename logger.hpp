#pragma once

#include <iostream>

namespace Clustering {
  extern bool verbose;
  extern std::ostream devnull;

  std::ostream& logger(std::ostream& s);
  std::ostream& debug();
} // end namespace Clustering

