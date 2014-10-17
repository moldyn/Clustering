
#include "density_clustering_mpi.hpp"

#include "density_clustering.hpp"
#include "tools.hpp"
#include "logger.hpp"

#include <mpi.h>

namespace Clustering {
namespace Density {
namespace MPI {

  namespace {
    const int MAIN_PROCESS = 0;
  } // end local namespace

  std::vector<std::size_t>
  calculate_populations(const float* coords,
                        const std::size_t n_rows,
                        const std::size_t n_cols,
                        const float radius,
                        const int mpi_n_nodes,
                        const int mpi_node_id) {
    //TODO change this to quadratically falling number of rows
    //     per chunk due to uneven workload of first chunks
    //     compared to last chunks (shorter j-loop!)
    unsigned int rows_per_chunk = n_rows / mpi_n_nodes;
    unsigned int i_row_from = mpi_node_id * rows_per_chunk;
    unsigned int i_row_to = i_row_from + rows_per_chunk;
    // last process has to do slightly more work
    // in case of uneven separation of workload
    if (mpi_node_id == mpi_n_nodes-1 ) {
      i_row_to = n_rows;
    }
    std::vector<unsigned int> pops(n_rows, 0);
    // parallel computation of pops using shared memory
    {
      const float rad2 = radius * radius;
      std::size_t i, j, k;
      float dist, c;
      ASSUME_ALIGNED(coords);
      #pragma omp parallel for default(none) private(i,j,k,c,dist) \
                               firstprivate(i_row_from,i_row_to,n_rows,n_cols,rad2) \
                               shared(coords,pops) \
                               schedule(dynamic,1024)
      for (i=i_row_from; i < i_row_to; ++i) {
        for (j=i+1; j < n_rows; ++j) {
          dist = 0.0f;
          #pragma simd reduction(+:dist)
          for (k=0; k < n_cols; ++k) {
            c = coords[i*n_cols+k] - coords[j*n_cols+k];
            dist += c*c;
          }
          if (dist < rad2) {
            #pragma omp atomic
            pops[i] += 1;
            #pragma omp atomic
            pops[j] += 1;
          }
        }
      }
    }
    // accumulate pops in main process and send result to slaves
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_node_id == MAIN_PROCESS) {
      // collect slave results
      for (int slave_id=1; slave_id < mpi_n_nodes; ++slave_id) {
        std::vector<unsigned int> pops_buf(n_rows);
        MPI_Recv(pops_buf.data(), n_rows, MPI_UNSIGNED, slave_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (std::size_t i=0; i < n_rows; ++i) {
          pops[i] += pops_buf[i];
        }
      }
    } else {
      // send pops from slaves
      MPI_Send(pops.data(), n_rows, MPI_UNSIGNED, MAIN_PROCESS, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // broadcast accumulated pops to slaves
    MPI_Bcast(pops.data(), n_rows, MPI_UNSIGNED, MAIN_PROCESS, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // cast unsigned int to size_t and add 1 for own structure
    std::vector<std::size_t> pops_result(n_rows);
    for (std::size_t i=0; i < n_rows; ++i) {
      pops_result[i] = (std::size_t) pops[i] + 1;
    }
    return pops_result;
  }


  //TODO MPI
  std::tuple<Neighborhood, Neighborhood>
  nearest_neighbors(const float* coords,
                    const std::size_t n_rows,
                    const std::size_t n_cols,
                    const std::vector<float>& free_energy,
                    const int mpi_n_nodes,
                    const int mpi_node_id) {

    //TODO i_row_from, i_row_to

    Neighborhood nh;
    Neighborhood nh_high_dens;
    // initialize neighborhood
    for (std::size_t i=0; i < n_rows; ++i) {
      nh[i] = Neighbor(n_rows+1, std::numeric_limits<float>::max());
      nh_high_dens[i] = Neighbor(n_rows+1, std::numeric_limits<float>::max());
    }
    // calculate nearest neighbors with distances
    std::size_t i, j, c, min_j, min_j_high_dens;
    float dist, d, mindist, mindist_high_dens;
    ASSUME_ALIGNED(coords);
    #pragma omp parallel for default(none) \
                             private(i,j,c,dist,d,mindist,mindist_high_dens,min_j,min_j_high_dens) \
                             firstprivate(n_rows,n_cols) \
                             shared(coords,nh,nh_high_dens,free_energy) \
                             schedule(dynamic, 2048)
    for (i=0; i < n_rows; ++i) {
      mindist = std::numeric_limits<float>::max();
      mindist_high_dens = std::numeric_limits<float>::max();
      min_j = n_rows+1;
      min_j_high_dens = n_rows+1;
      for (j=0; j < n_rows; ++j) {
        if (i != j) {
          dist = 0.0f;
          #pragma simd reduction(+:dist)
          for (c=0; c < n_cols; ++c) {
            d = coords[i*n_cols+c] - coords[j*n_cols+c];
            dist += d*d;
          }
          // direct neighbor
          if (dist < mindist) {
            mindist = dist;
            min_j = j;
          }
          // next neighbor with higher density / lower free energy
          if (free_energy[j] < free_energy[i] && dist < mindist_high_dens) {
            mindist_high_dens = dist;
            min_j_high_dens = j;
          }
        }
      }
      nh[i] = Neighbor(min_j, mindist);
      nh_high_dens[i] = Neighbor(min_j_high_dens, mindist_high_dens);
    }
    return std::make_tuple(nh, nh_high_dens);
  }


  void
  main(boost::program_options::variables_map args) {
    // initialize MPI
    MPI_Init(NULL, NULL);
    int n_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
    int node_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    // read basic inputs
    const std::string input_file = args["file"].as<std::string>();
    const float radius = args["radius"].as<float>();
    // setup coords
    float* coords;
    std::size_t n_rows;
    std::size_t n_cols;
    if (node_id == MAIN_PROCESS) {
      Clustering::logger(std::cout) << "reading coords" << std::endl;
    }
    std::tie(coords, n_rows, n_cols) = Clustering::Tools::read_coords<float>(input_file);
    //// free energies
    std::vector<float> free_energies;
    if (args.count("free-energy-input")) {
      if (node_id == MAIN_PROCESS) {
        Clustering::logger(std::cout) << "re-using free energy data." << std::endl;
      }
      free_energies = Clustering::Tools::read_free_energies(args["free-energy-input"].as<std::string>());
    } else if (args.count("free-energy") || args.count("population") || args.count("output")) {
      if (node_id == MAIN_PROCESS) {
        Clustering::logger(std::cout) << "calculating populations" << std::endl;
      }
      std::vector<std::size_t> pops = calculate_populations(coords, n_rows, n_cols, radius, n_nodes, node_id);
      if (node_id == MAIN_PROCESS && args.count("population")) {
        Clustering::Tools::write_single_column<std::size_t>(args["population"].as<std::string>(), pops);
      }
      if (node_id == MAIN_PROCESS) {
        Clustering::logger(std::cout) << "calculating free energies" << std::endl;
      }
      free_energies = Clustering::Density::calculate_free_energies(pops);
      if (node_id == MAIN_PROCESS && args.count("free-energy")) {
        Clustering::Tools::write_single_column<float>(args["free-energy"].as<std::string>(), free_energies, true);
      }
    }

    // TODO use mpi for following code, too

    // run the rest on a single node only
    if (node_id == MAIN_PROCESS) {
      //// nearest neighbors
      Neighborhood nh;
      Neighborhood nh_high_dens;
      if (args.count("nearest-neighbors-input")) {
        Clustering::logger(std::cout) << "re-using nearest neighbor data." << std::endl;
        std::ifstream ifs(args["nearest-neighbors-input"].as<std::string>());
        if (ifs.fail()) {
          std::cerr << "error: cannot open file '" << args["nearest-neighbors-input"].as<std::string>() << "'" << std::endl;
          exit(EXIT_FAILURE);
        } else {
          std::size_t i=0;
          while (ifs.good()) {
            std::size_t buf1;
            float buf2;
            std::size_t buf3;
            float buf4;
            ifs >> buf1;
            ifs >> buf2;
            ifs >> buf3;
            ifs >> buf4;
            if ( ! ifs.fail()) {
              nh[i] = std::pair<std::size_t, float>(buf1, buf2);
              nh_high_dens[i] = std::pair<std::size_t, float>(buf3, buf4);
              ++i;
            }
          }
        }
      } else if (args.count("nearest-neighbors") || args.count("output")) {
        Clustering::logger(std::cout) << "calculating nearest neighbors" << std::endl;
        auto nh_tuple = Clustering::Density::nearest_neighbors(coords, n_rows, n_cols, free_energies);
        nh = std::get<0>(nh_tuple);
        nh_high_dens = std::get<1>(nh_tuple);
        if (args.count("nearest-neighbors")) {
          std::ofstream ofs(args["nearest-neighbors"].as<std::string>());
          auto p = nh.begin();
          auto p_hd = nh_high_dens.begin();
          while (p != nh.end() && p_hd != nh_high_dens.end()) {
            // first: key (not used)
            // second: neighbor
            // second.first: id; second.second: squared dist
            ofs << p->second.first    << " " << p->second.second    << " "
                << p_hd->second.first << " " << p_hd->second.second << "\n";
            ++p;
            ++p_hd;
          }
        }
      }
      //// clustering
      if (args.count("output")) {
        const std::string output_file = args["output"].as<std::string>();
        std::vector<std::size_t> clustering;
        if (args.count("input")) {
          Clustering::logger(std::cout) << "reading initial clusters from file." << std::endl;
          clustering = Clustering::Tools::read_clustered_trajectory(args["input"].as<std::string>());
        } else {
          Clustering::logger(std::cout) << "calculating initial clusters" << std::endl;
          if (args.count("threshold") == 0) {
            std::cerr << "error: need threshold value for initial clustering" << std::endl;
            exit(EXIT_FAILURE);
          }
          float threshold = args["threshold"].as<float>();
          clustering = Clustering::Density::initial_density_clustering(free_energies, nh, threshold, coords, n_rows, n_cols);
        }
        if ( ! args["only-initial"].as<bool>()) {
          Clustering::logger(std::cout) << "assigning low density states to initial clusters" << std::endl;
          clustering = Clustering::Density::assign_low_density_frames(clustering, nh_high_dens, free_energies);
        }
        Clustering::logger(std::cout) << "writing clusters to file " << output_file << std::endl;
        Clustering::Tools::write_single_column<std::size_t>(output_file, clustering);
      }
    }

    // clean up
    if (node_id == MAIN_PROCESS) {
      Clustering::logger(std::cout) << "freeing coords" << std::endl;
    }
    Clustering::Tools::free_coords(coords);
    MPI_Finalize();
  }

} // end namespace MPI
} // end namespace Density
} // end namespace Clustering

