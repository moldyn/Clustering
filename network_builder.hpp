#pragma once

#include <vector>
#include <string>

using uint = unsigned int;

//TODO doc
std::vector<uint>
read_clustered_trajectory(std::string filename);

//TODO doc
void
write_clustered_trajectory(std::string filename, std::vector<uint> traj);

