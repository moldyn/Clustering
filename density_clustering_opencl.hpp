#pragma once

#include "config.hpp"

#include <CL/cl.hpp>

#include <vector>
#include <map>


namespace Clustering {
namespace Density {
namespace OpenCL {

//TODO doc
std::map<float, std::vector<std::size_t>>
calculate_populations(const float* coords, const std::size_t n_rows, const std::size_t n_cols, std::vector<float> radii);

} // end namespace OpenCL
} // end namespace Density
} // end namespace Clustering
