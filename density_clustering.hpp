/*
Copyright (c) 2015, Florian Sittel (www.lettis.net)
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

#include <vector>
#include <map>
#include <set>
#include <array>
#include <utility>
#include <string>

#include <boost/program_options.hpp>

#include "tools.hpp"

//! general namespace for clustering package
namespace Clustering {
  //! namespace for density-based clustering functions
  namespace Density {
    //! matches frame id to free energy
    using FreeEnergy = std::pair<std::size_t, float>;
    //! matches neighbor's frame id to distance
    using Neighbor = Clustering::Tools::Neighbor;
    //! map frame id to neighbors
    using Neighborhood = Clustering::Tools::Neighborhood;
    //! encodes 2D box for box-assisted search algorithm
    using Box = std::array<int, 2>;
    //! encodes box differences in 2D, i.e. if you are at
    //! the center box, the 9 different tuples hold the steppings
    //! to the 9 spacial neighbors (including the center box itself).
    constexpr int BOX_DIFF[9][2] = {{-1, 1}, { 0, 1}, { 1, 1}
                                  , {-1, 0}, { 0, 0}, { 1, 0}
                                  , {-1,-1}, { 0,-1}, { 1,-1}};
    //! number of neigbor boxes in 2D grid (including center box).
    const int N_NEIGHBOR_BOXES = 9;
//    //! encodes box differences in 3D, i.e. if you are at
//    //! the center box, the 27 different tuples hold the steppings
//    //! to the 27 spacial neighbors (including the center box itself).
//    constexpr int BOX_DIFF[27][3] = {{-1, 1,-1}, { 0, 1,-1}, { 1, 1,-1}
//                                   , {-1, 0,-1}, { 0, 0,-1}, { 1, 0,-1}
//                                   , {-1,-1,-1}, { 0,-1,-1}, { 1,-1,-1}
//                                   , {-1, 1, 0}, { 0, 1, 0}, { 1, 1, 0}
//                                   , {-1, 0, 0}, { 0, 0, 0}, { 1, 0, 0}
//                                   , {-1,-1, 0}, { 0,-1, 0}, { 1,-1, 0}
//                                   , {-1, 1, 1}, { 0, 1, 1}, { 1, 1, 1}
//                                   , {-1, 0, 1}, { 0, 0, 1}, { 1, 0, 1}
//                                   , {-1,-1, 1}, { 0,-1, 1}, { 1,-1, 1}};
//    //! number of neigbor boxes in cubic 3D grid (including center box).
//    const int N_NEIGHBOR_BOXES = 27;
    //! the full grid constructed for boxed-assisted nearest neighbor
    //! search with fixed distance criterion.
    struct BoxGrid {
      //! total number of boxes
      std::vector<int> n_boxes;
      //! matching frame id to the frame's assigned box
      std::vector<Box> assigned_box;
      //! the boxes with a list of assigned frame ids
      std::map<Box, std::vector<int>> boxes;
    };
    //! returns neighbor box given by neighbor index (in 3D: 27 different neighbors, including center itself)
    //! and the given center box.
    constexpr Box
    neighbor_box(const Box center, const int i_neighbor);
    //! uses fixed radius to separate coordinate space in equally sized
    //! boxes for box-assisted nearest neighbor search.
    BoxGrid
    compute_box_grid(const float* coords,
                     const std::size_t n_rows,
                     const std::size_t n_cols,
                     const float radius);
    //! returns true, if the box is a valid box in the grid.
    //! return false, if the box is outside of the grid.
    bool
    is_valid_box(const Box box,
                 const BoxGrid& grid);
    //! calculate population of n-dimensional hypersphere per frame for one fixed radius.
    std::vector<std::size_t>
    calculate_populations(const float* coords,
                          const std::size_t n_rows,
                          const std::size_t n_cols,
                          const float radius);
    //! calculate populations of n-dimensional hypersphere per frame for
    //! different radii in one go. computationally much more efficient than
    //! running single-radius version for every radius.
    std::map<float, std::vector<std::size_t>>
    calculate_populations(const float* coords,
                          const std::size_t n_rows,
                          const std::size_t n_cols,
                          const std::vector<float> radii);
    //! re-use populations to calculate local free energy estimate
    //! via $\Delta G = -k_B T \\ln(P)$.
    std::vector<float>
    calculate_free_energies(const std::vector<std::size_t>& pops);
    //! returns the given free energies sorted lowest to highest.
    //! original indices are retained.
    std::vector<FreeEnergy>
    sorted_free_energies(const std::vector<float>& fe);
    //! for every frame: compute the nearest neighbor (first tuple field)
    //! and the nearest neighbor with lower free energy, i.e. higher density (second tuple field).
    std::tuple<Neighborhood, Neighborhood>
    nearest_neighbors(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const std::vector<float>& free_energy);
    //! compute local neighborhood of a given frame.
    //! neighbor candidates are all frames below a given limit,
    //! effectively limiting the frames to the ones below a free energy cutoff.
    std::set<std::size_t>
    high_density_neighborhood(const float* coords,
                              const std::size_t n_cols,
                              const std::vector<FreeEnergy>& sorted_fe,
                              const std::size_t i_frame,
                              const std::size_t limit,
                              const float max_dist);
    //! compute sigma2 as deviation of squared nearest-neighbor distances.
    //! sigma2 is given by E[x^2] > Var(x) = E[x^2] - E[x]^2,
    //! with x being the distances between nearest neighbors).
    double
    compute_sigma2(const Neighborhood& nh);
    //! given an initial clustering computed from free energy cutoff screenings,
    //! assign all yet unclustered frames (those in 'state 0') to their geometrically
    //! next cluster. do this by starting at the lowest free energy of unassigned frames,
    //! then assigning the next lowest, etc.
    //! thus, all initial clusters will be filled with growing free energy, effectively producing
    //! microstates separated close to the free energy barriers.
    std::vector<std::size_t>
    assign_low_density_frames(const std::vector<std::size_t>& initial_clustering,
                              const Neighborhood& nh_high_dens,
                              const std::vector<float>& free_energy);
    //! user interface and controlling function for density-based geometric clustering.\n\n
    //! *parsed parameters*:\n
    //!   - **file** : input file with coordinates\n
    //!   - **free-energy-input**: previously computed free energies (input)\n
    //!   - **free-energy**: computed free energies (output)\n
    //!   - **population**: computed populations (output)\n
    //!   - **output**: clustered trajectory\n
    //!   - **radii**: list of radii for free energy / population computations (input)\n
    //!   - **radius**: radius for clustering (input)\n
    //!   - **nearest-neighbors-input**: previously computed nearest neighbor list (input)\n
    //!   - **nearest-neighbors**: nearest neighbor list (output)\n
    //!   - **threshold-screening**: option for automated free energy threshold screening (input)\n
    //!   - **threshold**: threshold for single run with limited free energy (input)\n
    //!   - **only-initial**: if true, do not fill microstates up to barriers,
    //!                       but keep initial clusters below free energy cutoff (bool flag)
    void
    main(boost::program_options::variables_map args);
  } // end namespace Density
} // end namespace Clustering

