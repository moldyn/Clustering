
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

namespace b_po = boost::program_options;


// 3-column float with col1 = x-val, col2 = y-val, col3 = density
// addressed by [row*3+col]
std::vector<float> calculate_density_histogram(const std::vector<float>& dens, const std::string& projections) {
//TODO
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
    ("population,p", b_po::bool_switch()->default_value(false), "print populations instead of densities");
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

  //std::time_t start, finish;
  //time(&start);

  std::vector<std::size_t> pops = calculate_populations(var_map["neighborhood"].as<std::string>(),
                                                        var_map["input"].as<std::string>());

  //time(&finish);
  //double elapsed = difftime(finish, start);
  //std::cerr << "time for neighborhood search [s]: " << elapsed << std::endl;
 

  if (var_map["population"].as<bool>()) {
    for (std::size_t i=0; i < pops.size(); ++i) {
      std::cout << i << " " << pops[i] << "\n";
    }
  } else {
    std::vector<float> dens = calculate_densities(pops);
    for (std::size_t i=0; i < dens.size(); ++i) {
      std::cout << i << " " << dens[i] << "\n";
    }
  }

  return 0;
}

