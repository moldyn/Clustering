#pragma once

#include "config.hpp"

#include <map>
#include <vector>

namespace Clustering {
namespace Density {
namespace CUDA {

  std::map<float, std::vector<std::size_t>>
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii);

}}} // end Clustering::Density::CUDA

