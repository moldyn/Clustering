
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <functional>

#include <time.h>

#include <omp.h>
#include <boost/program_options.hpp>

#include "tools.hpp"


namespace b_po = boost::program_options;


typedef std::map<std::size_t, std::map<std::size_t, float>> Neighborhood;


void calculate_neighborhood(const std::vector<float>& coords,
                            const std::size_t n_rows,
                            const std::size_t n_cols,
                            const float radius,
                            const int n_threads) {
  const float rad2 = radius * radius;
  std::size_t i, j, k;
  float dist, c;

  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #pragma omp parallel for default(shared) private(i,j,k,c,dist) schedule(dynamic)
  for (i=0; i < n_rows; ++i) {
    std::vector<std::pair<std::size_t, float>> dist_to;
    for (j=i+1; j < n_rows; ++j) {
      dist = 0.0f;
      for (k=0; k < n_cols; ++k) {
        c = coords[i*n_cols+k] - coords[j*n_cols+k];
        dist += c*c;
      }
      if (dist < rad2) {
        dist_to.push_back({j, dist});
      }
    }
    #pragma omp critical
    {
      for (auto it=dist_to.begin(); it != dist_to.end(); ++it) {
        std::cout << i << " " << it->first << " " << it->second << "\n";
      }
    }
  }
}



int main(int argc, char* argv[]) {

  b_po::variables_map var_map;
  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "calculate neigborhood around frames.\n"
    "output is of the form\n\n"
    "   index1 index2 squared_dist\n\n"
    "and by no means guaranteed to be sorted along any axis.\n"
    "additionally, the output is to be interpreted as a kind of sparse symmetric matrix.\n"
    "i.e. the entry\n"
    "   index1 index2 squared_dist\n"
    "encodes\n"
    "   index2 index1 squared_dist\n"
    "as well!\n"
    "\n"
    "options"));
  desc.add_options()
    ("help,h", "show this help")
    ("input", b_po::value<std::string>()->required(),
        "input file (whitespace-separated ASCII)")
    ("radius,r", b_po::value<float>()->required(),
        "radius of hypersphere")
    ("nthreads,n", b_po::value<int>()->default_value(0),
        "number of OpenMP threads to use. if set to zero, will use value of OMP_NUM_THREADS; default: 0");
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

  auto coords_tuple = read_coords<float>(var_map["input"].as<std::string>());

  std::vector<float> coords = std::get<0>(coords_tuple);
  std::size_t n_cols = std::get<1>(coords_tuple);
  std::size_t n_rows = std::get<2>(coords_tuple);

  //std::time_t start, finish;
  //time(&start);
  calculate_neighborhood(coords,
                         n_rows,
                         n_cols,
                         var_map["radius"].as<float>(),
                         var_map["nthreads"].as<int>());
  //time(&finish);
  //double elapsed = difftime(finish, start);
  //std::cerr << "time for neighborhood search [s]: " << elapsed << std::endl;
  return 0;
}

