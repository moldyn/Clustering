#pragma once

#include "config.hpp"

#include <vector>

namespace DC_MPI {

void
density_main(boost::program_options::variables_map args);

void
setup();

std::vector<std::size_t>
calculate_populations(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const float radius);

} // end namespace DC_MPI
