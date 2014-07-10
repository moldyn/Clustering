#pragma once

#include <vector>
#include <map>
#include <string>

using Density = std::pair<std::size_t, float>;
using Neighborhood = std::map<std::size_t, std::map<std::size_t, float>>;

//TODO doc
std::vector<std::size_t>
calculate_populations(const std::vector<float>& coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const float radius,
                      const int n_threads);

//TODO doc
std::vector<float>
calculate_densities(const std::vector<std::size_t>& pops);

// 3-column float with col1 = x-val, col2 = y-val, col3 = density
// addressed by [row*3+col] with n_rows = n_bins^2
std::vector<float>
calculate_density_histogram(const std::vector<float>& dens,
                            const std::string& projections,
                            std::pair<std::size_t, std::size_t> dims,
                            std::size_t n_bins);

//TODO doc
const std::pair<std::size_t, float>
nearest_neighbor(const std::vector<float>& coords,
                 const std::vector<Density>& sorted_density,
                 std::size_t n_cols,
                 std::size_t frame_id,
                 std::pair<std::size_t, std::size_t> search_range);

//TODO doc
std::vector<std::size_t>
density_clustering(std::vector<float> dens,
                   float density_threshold,
                   float density_radius,
                   std::string coords_file);

