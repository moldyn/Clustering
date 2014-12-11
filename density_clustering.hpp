#pragma once

#include <vector>
#include <map>
#include <set>
#include <utility>
#include <string>

#include <boost/program_options.hpp>

namespace Clustering {
  namespace Density {
    using FreeEnergy = std::pair<std::size_t, float>;
    using SizePair = std::pair<std::size_t, std::size_t>;
    using Neighbor = std::pair<std::size_t, float>;
    using Neighborhood = std::map<std::size_t, std::pair<std::size_t, float>>;
  
    //TODO doc
    std::vector<std::size_t>
    calculate_populations(const float* coords,
                          const std::size_t n_rows,
                          const std::size_t n_cols,
                          const float radius);
  
    //TODO doc
    std::map<float, std::vector<std::size_t>>
    calculate_populations(const float* coords,
                          const std::size_t n_rows,
                          const std::size_t n_cols,
                          const std::vector<float> radii);


    //TODO doc
    std::vector<float>
    calculate_free_energies(const std::vector<std::size_t>& pops);
  
    //TODO doc
    std::vector<FreeEnergy>
    sorted_free_energies(const std::vector<float>& fe);
  
    //TODO doc
    std::tuple<Neighborhood, Neighborhood>
    nearest_neighbors(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const std::vector<float>& free_energy);
  
    // returns neighborhood set of single frame.
    // all ids are sorted in free energy.
    std::set<std::size_t>
    high_density_neighborhood(const float* coords,
                              const std::size_t n_cols,
                              const std::vector<FreeEnergy>& sorted_fe,
                              const std::size_t i_frame,
                              const std::size_t limit,
                              const float max_dist);
    
    //TODO doc
    double
    compute_sigma2(const Neighborhood& nh);

    //TODO doc
    std::vector<std::size_t>
    assign_low_density_frames(const std::vector<std::size_t>& initial_clustering,
                              const Neighborhood& nh_high_dens,
                              const std::vector<float>& free_energy);


    //TODO doc
    std::pair<Neighborhood, Neighborhood>
    read_neighborhood(const std::string fname);

    //TODO doc
    void
    write_neighborhood(const std::string fname,
                       const Neighborhood& nh,
                       const Neighborhood& nh_high_dens);

    // TODO doc
    void
    main(boost::program_options::variables_map args);
  } // end namespace Density
} // end namespace Clustering

