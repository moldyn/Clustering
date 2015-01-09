#pragma once

#include "config.hpp"

#include <vector>

namespace DC_OpenCL {

void
setup();

std::vector<std::size_t>
calculate_populations(const float* coords, const float radius);

} // end namespace DC_OpenCL
