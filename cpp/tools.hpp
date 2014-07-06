#pragma once

#include <string>
#include <vector>
#include <tuple>

// read coordinates from space-separated ASCII file.
// will write data with precision of MY_FLOAT into vector.
// format: [row * n_cols + col]
// return value: tuple of {data (vector), n_rows (size_t), n_cols (size_t)}.
template <typename MY_FLOAT>
std::tuple<std::vector<MY_FLOAT>, std::size_t, std::size_t>
read_coords(std::string filename,
            std::vector<std::size_t> usecols = {});

// template implementations
#include "tools.hxx"

