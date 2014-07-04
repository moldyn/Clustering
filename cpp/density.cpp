
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>
#include <map>
#include <utility>
#include <functional>
#include <limits>

#include <time.h>

#include <omp.h>
#include <boost/program_options.hpp>

namespace b_po = boost::program_options;

void calculate_density(const std::string neighborhood, const std::string projections) {
  std::map<std::size_t, std::size_t> dens;
  {
    std::ifstream ifs(neighborhood);
    std::size_t buf;
    while (ifs.good()) {
      // from
      ifs >> buf;
      dens[buf] = dens[buf] + 1;
      // to
      ifs >> buf;
      dens[buf] = dens[buf] + 1;
      // next line
      ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
  }




  for (auto it=dens.begin(); it != dens.end(); ++it) {
    std::cout << it->first << " " << it->second << "\n";
  }
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
    ("input", b_po::value<std::string>()->required(), "projected data")
    ("neighborhood", b_po::value<std::string>()->required(), "neighborhood info");
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

  calculate_density(var_map["neighborhood"].as<std::string>(), var_map["input"].as<std::string>());
  
  //time(&finish);
  //double elapsed = difftime(finish, start);
  //std::cerr << "time for neighborhood search [s]: " << elapsed << std::endl;
  return 0;
}

