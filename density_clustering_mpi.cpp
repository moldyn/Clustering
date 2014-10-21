
#include "density_clustering_mpi.hpp"
#include "density_clustering_common.hpp"

#include "tools.hpp"
#include "logger.hpp"

#include <mpi.h>

namespace Clustering {
namespace Density {
namespace MPI {

  namespace {
    /*
     * use no. of rows and no. of MPI nodes to calculate the optimal
     * indices for equal load balancing on all machines to
     * calculate an upper triangular matrix by a nested loop
     * of the form
     *   FOR i=0 TO n_rows
     *     FOR j=i+1 TO n_rows
     *       [...]
     *
     * the returned indices are for the initial index of the outer loop.
     * the inner loop will be local.
     * i.e. on every MPI node, the loop will be of the form
     *   FOR i=indices[node_id] TO (is_last_node ? n_rows : indices[next_node])
     *     FOR j=i+1 TO n_rows
     *       [...]
     *
     * computation of optimal indices is heavily based on triangular summation
     * (please recall the 'algorithm of young C.F. Gauss').
     */
    std::vector<std::size_t>
    triangular_load_balance(std::size_t n_rows,
                            std::size_t n_nodes) {
      auto young_gauss = [](std::size_t n) -> std::size_t {
        return n*(n+1) / 2;
      };
      std::size_t workload = young_gauss(n_rows) / n_nodes;
      std::size_t last_index = 0;
      std::vector<std::size_t> load_balanced_indices(n_nodes);
      for (int i=n_nodes-1; i >= 0; --i) {
        if (i == 0) {
          load_balanced_indices[i] = 0;
        } else {
          last_index = (std::size_t) sqrt(2*(young_gauss(last_index) + workload));
          load_balanced_indices[i] = n_rows - last_index;
        }
      }
      return load_balanced_indices;
    }
  } // end local namespace


  std::vector<std::size_t>
  calculate_populations(const float* coords,
                        const std::size_t n_rows,
                        const std::size_t n_cols,
                        const float radius,
                        const int mpi_n_nodes,
                        const int mpi_node_id) {
    std::vector<std::size_t> load_balanced_indices = triangular_load_balance(n_rows, mpi_n_nodes);
    unsigned int i_row_from = load_balanced_indices[mpi_node_id];
    unsigned int i_row_to;
    if (mpi_node_id == mpi_n_nodes-1) {
      // last node: run to end
      i_row_to = n_rows;
    } else {
      i_row_to = load_balanced_indices[mpi_node_id+1];
    }
    std::vector<unsigned int> pops(n_rows, 0);
    // per-node parallel computation of pops using shared memory
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


  std::tuple<Neighborhood, Neighborhood>
  nearest_neighbors(const float* coords,
                    const std::size_t n_rows,
                    const std::size_t n_cols,
                    const std::vector<float>& free_energy,
                    const int mpi_n_nodes,
                    const int mpi_node_id) {
    unsigned int rows_per_chunk = n_rows / mpi_n_nodes;
    unsigned int i_row_from = mpi_node_id * rows_per_chunk;
    unsigned int i_row_to = i_row_from + rows_per_chunk;
    // last process has to do slightly more work
    // in case of uneven separation of workload
    if (mpi_node_id == mpi_n_nodes-1 ) {
      i_row_to = n_rows;
    }
    Neighborhood nh;
    Neighborhood nh_high_dens;
    // initialize neighborhood
    for (std::size_t i=i_row_from; i < i_row_to; ++i) {
      nh[i] = Neighbor(n_rows+1, std::numeric_limits<float>::max());
      nh_high_dens[i] = Neighbor(n_rows+1, std::numeric_limits<float>::max());
    }
    // calculate nearest neighbors with distances
    {
      std::size_t i, j, c, min_j, min_j_high_dens;
      float dist, d, mindist, mindist_high_dens;
      ASSUME_ALIGNED(coords);
      #pragma omp parallel for default(none) \
                               private(i,j,c,dist,d,mindist,mindist_high_dens,min_j,min_j_high_dens) \
                               firstprivate(i_row_from,i_row_to,n_rows,n_cols) \
                               shared(coords,nh,nh_high_dens,free_energy) \
                               schedule(dynamic, 2048)
      for (i=i_row_from; i < i_row_to; ++i) {
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
    }
    // collect results in MAIN_PROCESS
    MPI_Barrier(MPI_COMM_WORLD);
    {
      // buf: I_FROM, I_TO_NH, DIST, I_TO_NH_HD, DIST_HD
      int BUF_SIZE = 5;
      float buf[BUF_SIZE];
      if (mpi_node_id == MAIN_PROCESS) {
        while (nh.size() != n_rows) {
          MPI_Recv(buf, BUF_SIZE, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          nh[(int) buf[0]] = Neighbor((int) buf[1], buf[2]);
          nh_high_dens[(int) buf[0]] = Neighbor((int) buf[3], buf[4]);
        }
      } else {
        for (std::size_t i=i_row_from; i < i_row_to; ++i) {
          buf[0] = (float) i;
          buf[1] = (float) nh[i].first;
          buf[2] = nh[i].second;
          buf[3] = nh_high_dens[i].first;
          buf[4] = nh_high_dens[i].second;
          MPI_Send(buf, BUF_SIZE, MPI_FLOAT, MAIN_PROCESS, 0, MPI_COMM_WORLD);
        }
      }
    }
    // broadcast result to slaves
    MPI_Barrier(MPI_COMM_WORLD);
    {
      // buf: n_rows X {I_FROM, I_TO_NH, DIST, I_TO_NH_HD, DIST_HD}
      int BUF_SIZE = 4*n_rows;
      std::vector<float> buf(BUF_SIZE);
      if (mpi_node_id == MAIN_PROCESS) {
        for (std::size_t i=0; i < n_rows; ++i) {
          buf[4*i] = (float) nh[i].first;
          buf[4*i+1] = nh[i].second;
          buf[4*i+2] = (float) nh_high_dens[i].first;
          buf[4*i+3] = nh_high_dens[i].second;
        }
      }
      MPI_Bcast(buf.data(), BUF_SIZE, MPI_FLOAT, MAIN_PROCESS, MPI_COMM_WORLD);
      if (mpi_node_id != MAIN_PROCESS) {
        // unpack broadcasted neighborhood data
        for (std::size_t i=0; i < n_rows; ++i) {
          nh[i] = Neighbor((int) buf[4*i], buf[4*i+1]);
          nh_high_dens[i] = Neighbor((int) buf[4*i+2], buf[4*i+3]);
        }
      }
    }
    return std::make_tuple(nh, nh_high_dens);
  }


  std::set<std::size_t>
  high_density_neighborhood(const float* coords,
                            const std::size_t n_cols,
                            const std::vector<FreeEnergy>& sorted_fe,
                            const std::size_t i_frame,
                            const std::size_t limit,
                            const float max_dist,
                            const int mpi_n_nodes,
                            const int mpi_node_id) {
    //TODO calculate i_from, i_to for MPI processes
    
    // buffer to hold information whether frame i is
    // in neighborhood (-> assign 1) or not (-> keep 0)
    std::vector<int> frame_in_nh(limit, 0);
    std::set<std::size_t> nh;
    std::size_t j,c;
    const std::size_t i_frame_sorted = sorted_fe[i_frame].first * n_cols;
    float d,dist2;
    ASSUME_ALIGNED(coords);
    #pragma omp parallel for default(none) private(j,c,d,dist2) \
                             firstprivate(i_frame,i_frame_sorted,limit,max_dist,n_cols) \
                             shared(coords,sorted_fe,frame_in_nh)
    for (j=0; j < limit; ++j) {
      if (i_frame != j) {
        dist2 = 0.0f;
        #pragma simd reduction(+:dist2)
        for (c=0; c < n_cols; ++c) {
          d = coords[i_frame_sorted+c] - coords[sorted_fe[j].first*n_cols+c];
          dist2 += d*d;
        }
        if (dist2 < max_dist) {
          frame_in_nh[j] = 1;
        }
      }
    }
    // reduce buffer data to real neighborhood structure
    for (j=0; j < limit; ++j) {
      if (frame_in_nh[j] > 0) {
        nh.insert(j);
      }
    }
    nh.insert(i_frame);

    //TODO collect data from all slaves & broadcast back

    return nh;
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
    //// nearest neighbors
    Neighborhood nh;
    Neighborhood nh_high_dens;
    if (args.count("nearest-neighbors-input")) {
      Clustering::logger(std::cout) << "re-using nearest neighbor data." << std::endl;
      auto nh_pair = Clustering::Density::read_neighborhood(args["nearest-neighbors-input"].as<std::string>());
      nh = nh_pair.first;
      nh_high_dens = nh_pair.second;
    } else if (args.count("nearest-neighbors") || args.count("output")) {
      Clustering::logger(std::cout) << "calculating nearest neighbors" << std::endl;
      auto nh_tuple = nearest_neighbors(coords, n_rows, n_cols, free_energies, n_nodes, node_id);
      nh = std::get<0>(nh_tuple);
      nh_high_dens = std::get<1>(nh_tuple);
      if (node_id == MAIN_PROCESS && args.count("nearest-neighbors")) {
        Clustering::Density::write_neighborhood(args["nearest-neighbors"].as<std::string>(), nh, nh_high_dens);
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
        clustering = Clustering::Density::initial_density_clustering(free_energies, nh, threshold, coords, n_rows, n_cols, n_nodes, node_id);
      }
      if (node_id == MAIN_PROCESS) {
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

