
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>
#include <map>
#include <utility>
#include <functional>
#include <algorithm>
#include <limits>

#include <time.h>

#include <omp.h>
#include <boost/program_options.hpp>


#include "tools.hpp"


namespace b_po = boost::program_options;

using Density = std::pair<std::size_t, float>;


// 3-column float with col1 = x-val, col2 = y-val, col3 = density
// addressed by [row*3+col] with n_rows = n_bins^2
std::vector<float> calculate_density_histogram(const std::vector<float>& dens,
                                               const std::string& projections,
                                               std::pair<std::size_t, std::size_t> dims,
                                               std::size_t n_bins) {
  auto coords_tuple = read_coords<float>(projections, {dims.first, dims.second});
  std::vector<float> coords = std::get<0>(coords_tuple);
  std::size_t n_rows = std::get<1>(coords_tuple);

  auto X = [&](std::size_t i) { return coords[i*2]; };
  auto Y = [&](std::size_t i) { return coords[i*2 + 1]; };

  float x_min = X(0);
  float x_max = X(0);
  float y_min = Y(0);
  float y_max = Y(0);

  for (std::size_t i=1; i < n_rows; ++i) {
    auto mm = std::minmax({X(i), x_min, x_max});
    x_min = mm.first;
    x_max = mm.second;
    mm = std::minmax({Y(i), y_min, y_max});
    y_min = mm.first;
    y_max = mm.second;
  }

  float dx = (x_max - x_min) / n_bins;
  float dy = (y_max - y_min) / n_bins;

  // setup bins
  std::vector<float> hist(3*n_bins*n_bins);
  // addressing shortcuts
  auto HX = [&](std::size_t i, std::size_t j) -> float& {return hist[(i*n_bins+j)*3];};
  auto HY = [&](std::size_t i, std::size_t j) -> float& {return hist[(i*n_bins+j)*3+1];};
  auto HZ = [&](std::size_t i, std::size_t j) -> float& {return hist[(i*n_bins+j)*3+2];};
  for (std::size_t i=0; i < n_bins; ++i) {
    for (std::size_t j=0; j < n_bins; ++j) {
      HX(i,j) = x_min + i*dx;
      HY(i,j) = y_min + j*dy;
    }
  }
  // sort data into bins
  for (std::size_t i_frame=0; i_frame < n_rows; ++i_frame) {
    std::size_t i_xbin = n_bins-1;
    std::size_t i_ybin = n_bins-1;
    // find index for x-bin
    for (std::size_t ix=0; ix < n_bins-1; ++ix) {
      float x_val = coords[i_frame*2];
      if (HX(ix,0) < x_val && x_val < HX(ix+1,0)) {
        i_xbin = ix;
        break;
      }
    }
    // find index for y-bin
    for (std::size_t iy=0; iy < n_bins-1; ++iy) {
      float y_val = coords[i_frame*2+1];
      if (HY(i_xbin,iy) < y_val && y_val < HY(i_xbin,iy+1)) {
        i_ybin = iy;
        break;
      }
    }
    // add density to histogram
    HZ(i_xbin, i_ybin) += dens[i_frame];
  }
  return hist;
}


std::vector<float> calculate_densities(const std::vector<std::size_t>& pops) {
  const std::size_t n_frames = pops.size();
  std::vector<float> dens(n_frames);
  float max_pop = (float) ( * std::max_element(pops.begin(), pops.end()));
  for (std::size_t i=0; i < n_frames; ++i) {
    dens[i] = (float) pops[i] / max_pop;
  }
  return dens;
}


std::vector<std::size_t> calculate_populations(const std::string& neighborhood, const std::string& projections) {
  std::size_t n_frames = 0;
  // read projections linewise to determine number of frames
  {
    std::string line;
    std::ifstream ifs(projections);
    while (ifs.good()) {
      std::getline(ifs, line);
      if ( ! line.empty()) {
        ++n_frames;
      }
    }
  }
  // set initial population = 1 for every frame
  // (i.e. the frame itself is in population)
  std::vector<std::size_t> pops(n_frames, 1);
  // read neighborhood info and calculate populations
  {
    std::ifstream ifs(neighborhood);
    std::size_t buf;
    while (ifs.good()) {
      // from
      ifs >> buf;
      ++pops[buf];
      // to
      ifs >> buf;
      ++pops[buf];
      // next line
      ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
  }
  return pops;
}


const std::pair<std::size_t, float> nearest_neighbor(const std::vector<float>& coords,
                                                     const std::vector<Density>& sorted_density,
                                                     std::size_t n_cols,
                                                     std::size_t frame_id,
                                                     std::pair<std::size_t, std::size_t> search_range) {
  std::size_t min_ndx = 0;
  float min_dist = std::numeric_limits<float>::max();
  for (std::size_t j=search_range.first; j < search_range.second; ++j) {
    float dist = 0.0f;
    for (std::size_t c=0; c < n_cols; ++c) {
      float d = coords[sorted_density[frame_id].first*n_cols+c] - coords[sorted_density[j].first*n_cols+c];
      dist += d*d;
    }
    if (dist < min_dist) {
      min_dist = dist;
      min_ndx = j;
    }
  }
  return {min_ndx, min_dist};
}


std::vector<std::size_t> density_clustering(std::vector<float> dens,
                                           float density_threshold,
                                           float density_radius,
                                           std::string coords_file) {
  std::vector<Density> density_sorted;
  for (std::size_t i=0; i < dens.size(); ++i) {
    density_sorted.push_back({i, dens[i]});
  }
  // sort for density: highest to lowest
  std::sort(density_sorted.begin(),
            density_sorted.end(),
            [] (const Density& d1, const Density& d2) -> bool {return d1.second > d2.second;});

  auto coords_tuple = read_coords<float>(coords_file);
  std::vector<float>& coords = std::get<0>(coords_tuple);
  std::size_t n_rows = std::get<1>(coords_tuple);
  std::size_t n_cols = std::get<2>(coords_tuple);


  std::vector<std::size_t> clustering(n_rows);
  std::size_t n_clusters = 0;

  auto lb = std::lower_bound(density_sorted.begin(),
                             density_sorted.end(),
                             Density(0, density_threshold), 
                             [](const Density& d1, const Density& d2) -> bool {return d1.second < d2.second;});

  std::size_t last_frame_below_threshold = (lb - density_sorted.begin());

  for (std::size_t i=0; i < last_frame_below_threshold; ++i) {
    auto nn_pair = nearest_neighbor(coords, density_sorted, n_cols, i, {0,i});
    if (nn_pair.second < density_radius) {
      // add to existing cluster of frame with 'min_dist'
      clustering[density_sorted[i].first] = clustering[density_sorted[nn_pair.first].first];
    } else {
      // create new cluster
      ++n_clusters;
      clustering[density_sorted[i].first] = n_clusters;
    }
  }

  // find nearest neighbors for all unassigned frames
  std::map<std::size_t, std::size_t> nearest_neighbors;
  for (std::size_t i=last_frame_below_threshold; i < density_sorted.size(); ++i) {
    auto nn_pair = nearest_neighbor(coords, density_sorted, n_cols, i, {0, density_sorted.size()});
    nearest_neighbors[i] = nn_pair.first;
  }
  // assign clusters to unassigned frames via neighbor-info
  while (nearest_neighbors.size() > 0) {
    for (auto it=nearest_neighbors.begin(); it != nearest_neighbors.end(); ++it) {
      if (clustering[it->second] != 0) {
        clustering[it->first] = it->second;
        nearest_neighbors.erase(it);
      }
    }
    //TODO: check: will this always converge? better add a 'no_change'-bool?
  }
  return clustering;
}

int main(int argc, char* argv[]) {

  b_po::variables_map var_map;
  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "output: populations of frame environment\n"
    "\n"
    "options"));
  desc.add_options()
    ("help,h", "show this help")

    ("input,i", b_po::value<std::string>()->required(), "projected data")
    ("neighborhood,N", b_po::value<std::string>()->required(), "neighborhood info")

    ("population,p", b_po::bool_switch()->default_value(false), "print populations instead of densities")

    ("histogram,H", b_po::bool_switch()->default_value(false), "print histogram data for densities")
    ("nbins", b_po::value<int>()->default_value(200), "#bins for histogram (default: 200)")

    ("density-clustering,C", b_po::value<float>(), "print clustering given by density threshold");


  try {
    b_po::positional_options_description p;
    p.add("input", -1);
    b_po::store(b_po::command_line_parser(argc, argv).options(desc).positional(p).run(), var_map);
    b_po::notify(var_map);
  } catch (b_po::error& e) {
    if ( ! var_map.count("help")) {
      std::cout << "\n" << e.what() << "\n\n" << std::endl;
    }
    std::cout << desc << std::endl;
    return 2;
  }

  if (var_map.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  ////

  std::vector<std::size_t> pops = calculate_populations(var_map["neighborhood"].as<std::string>(),
                                                        var_map["input"].as<std::string>());

  if (var_map["population"].as<bool>()) {
    for (std::size_t i=0; i < pops.size(); ++i) {
      std::cout << i << " " << pops[i] << "\n";
    }
  } else if (var_map["histogram"].as<bool>()) {
    std::vector<float> dens = calculate_densities(pops);
    int n_bins = var_map["nbins"].as<int>();
    std::vector<float> hist = calculate_density_histogram(dens,
                                                          var_map["input"].as<std::string>(),
                                                          {0,1},
                                                          n_bins);
    for (std::size_t i=0; i < hist.size() / 3; ++i) {
      std::cout << hist[i*3] << " " << hist[i*3+1] << " " << hist[i*3+2] << "\n";
    }
  } else if (var_map.count("density-clustering")) {
    std::vector<std::size_t> clusters = density_clustering(calculate_densities(pops),
                                                           var_map["density_clustering"].as<float>(),
                                                           var_map["radius"].as<float>(),
                                                           var_map["input"].as<std::string>());
  } else {
    std::vector<float> dens = calculate_densities(pops);
    for (std::size_t i=0; i < dens.size(); ++i) {
      std::cout << i << " " << dens[i] << "\n";
    }
  }

  return 0;
}

