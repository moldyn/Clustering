#pragma once

#include "config.hpp"
#include "density_clustering_common.hpp"

#include <map>
#include <vector>

namespace Clustering {
namespace Density {
namespace CUDA {

  using Neighborhood = Clustering::Tools::Neighborhood;

  Pops
  calculate_populations_partial(const float* coords
                              , const std::vector<float>& sorted_coords
                              , const std::vector<float>& blimits
                              , std::size_t n_rows
                              , std::size_t n_cols
                              , std::vector<float> radii
                              , std::size_t i_from
                              , std::size_t i_to
                              , int i_gpu);
  
  Pops
  calculate_populations(const float* coords
                      , const std::size_t n_rows
                      , const std::size_t n_cols
                      , std::vector<float> radii);

  std::tuple<Neighborhood, Neighborhood>
  nearest_neighbors(const float* coords,
                    const std::size_t n_rows,
                    const std::size_t n_cols,
                    const std::vector<float>& free_energy);

}}} // end Clustering::Density::CUDA

