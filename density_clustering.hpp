#pragma once

#include <vector>
#include <map>
#include <utility>
#include <string>

using FreeEnergy = std::pair<std::size_t, float>;
using SizePair = std::pair<std::size_t, std::size_t>;
using Neighbor = std::pair<std::size_t, float>;
using Neighborhood = std::map<std::size_t, std::pair<std::size_t, float>>;

                            // assigned,  unassigned,  distance
using Candidate = std::tuple<std::size_t, std::size_t, float>;

