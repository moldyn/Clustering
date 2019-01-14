/*
Copyright (c) 2015-2019, Florian Sittel (www.lettis.net) and Daniel Nagel
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "config.hpp"

#include <CL/cl.hpp>

#include <vector>
#include <map>


namespace Clustering {
namespace Density {
//! OpenCL implementations of compute intensive functions
namespace OpenCL {
  //! OpenCL implementation of
  //! \link Clustering::Density::calculate_populations(const float* coords, const std::size_t n_rows, const std::size_t n_cols, std::vector<float> radii);
  //!
  //! **ATTENTION**: OpenCL support is unfinished and not expected to return sound results.
  std::map<float, std::vector<std::size_t>>
  calculate_populations(const float* coords, const std::size_t n_rows, const std::size_t n_cols, std::vector<float> radii);
} // end namespace OpenCL
} // end namespace Density
} // end namespace Clustering
