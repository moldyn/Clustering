#pragma once

#include <ostream>

namespace {
  bool verbose = false;
  std::ostream devnull(0);
  std::ostream& logger(std::ostream& s) {
    if (verbose) {
      return s;
    } else {
      return devnull; 
    }
  }
} // end local namespace

