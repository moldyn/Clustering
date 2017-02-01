#pragma once

#include "config.hpp"
#include "density_clustering_common.hpp"

#include <map>
#include <vector>

namespace Clustering {
namespace Density {
namespace CUDA {

  void
  check_error(std::string msg="");

  int
  get_num_gpus();

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

  std::vector<std::size_t>
  sanitize_state_names(std::vector<std::size_t> clustering);

  std::vector<std::size_t>
  screening(const std::vector<float>& free_energy
          , const Neighborhood& nh
          , const float free_energy_threshold
          , const float* coords
          , const std::size_t n_rows
          , const std::size_t n_cols
          , const std::vector<std::size_t> initial_clusters);

  std::set<std::size_t>
  high_density_neighborhood(const float* coords,
                            const std::size_t n_cols,
                            const std::vector<FreeEnergy>& sorted_fe,
                            const std::size_t i_frame,
                            const std::size_t limit,
                            const float max_dist);

}}} // end Clustering::Density::CUDA

