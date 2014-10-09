#pragma once

#include <boost/program_options.hpp>

//TODO doc
template <typename NUM>
std::vector<NUM>
read_single_column(std::string filename);

////TODO doc
//std::vector<std::size_t>
//read_clustered_trajectory(std::string filename);

std::vector<float>
read_free_energies(std::string filename);

//TODO: doc
void
density_main(boost::program_options::variables_map args);

//TODO doc
std::map<std::size_t, std::size_t>
microstate_populations(std::vector<std::size_t> traj);

/*
 * MPP clustering
 *  args:
 *   input (string)
 *   lagtime (int)
 *   qmin-from (float)
 *   qmin-to (float)
 *   qmin-step (float)
 */
void
mpp_main(boost::program_options::variables_map args);

