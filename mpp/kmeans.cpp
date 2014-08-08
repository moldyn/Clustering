
#include <random>
#include <ctime>
#include <functional>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>
#include <set>
#include <stdexcept>

#include "kmeans.h"

namespace {

double norm_p2(const std::vector<double>& m1,
               const std::vector<double>& m2,
               const std::size_t i1, const std::size_t i2, const std::size_t n_cols) {
  //TODO: SIMD? FMA? OpenMP?
  double norm = 0.0;
  for (std::size_t j=0; j < n_cols; ++j) {
    double d = m1[i1*n_cols+j] - m2[i2*n_cols+j];
    norm += d*d;
  }
  return norm;
}

void add_vector(std::vector<double>& v_acc,
                const std::vector<double>& v_ref,
                const std::size_t i1,
                const std::size_t i2,
                const std::size_t n_cols) {
  for (std::size_t j=0; j < n_cols; ++j) {
    v_acc[i1*n_cols+j] += v_ref[i2*n_cols+j];
  }
}

void subtract_vector(std::vector<double>& v_acc,
                     const std::vector<double>& v_ref,
                     const std::size_t i1,
                     const std::size_t i2,
                     const std::size_t n_cols) {
  for (std::size_t j=0; j < n_cols; ++j) {
    v_acc[i1*n_cols+j] -= v_ref[i2*n_cols+j];
  }
}

std::vector<double> normalize_to_centroids(const std::vector<double>& center_accs,
                                           const std::vector<std::size_t>& populations,
                                           const std::size_t n_cols) {
  std::vector<double> centers(center_accs);
  for (std::size_t i=0; i < populations.size(); ++i) {
    for (std::size_t j=0; j < n_cols; ++j) {
      centers[i*n_cols+j] /= populations[i];
    }
  }
  return centers;
}

double calculate_cluster_variance(const std::vector<double>& coords,
                                  const std::vector<double>& cluster_centers,
                                  const std::vector<std::size_t>& cluster_assignments,
                                  const std::size_t n_cols) {
  double variance = 0.0;
  for (std::size_t i=0; i < cluster_assignments.size(); ++i) {
    for (std::size_t j=0; j < n_cols; ++j) {
      double r = coords[i*n_cols+j] - cluster_centers[cluster_assignments[i]*n_cols+j];
      variance += r*r;
    }
  }
  return variance;
}

MicroStates kmeans(const std::size_t n_microstates,
                   const std::vector<double>& coords,
                   const std::size_t n_rows,
                   const std::size_t n_cols,
                   const std::size_t n_threads) {
  // initialize cluster centers with random points in coordinate space
  std::vector<double> cluster_centers(n_microstates*n_cols);
  {
    std::set<int> rnd_indices;
    auto rnd_num = std::bind(std::uniform_int_distribution<int>(0, n_rows-1),
                             std::default_random_engine(std::time(NULL)));
    while (rnd_indices.size() < n_microstates) {
      rnd_indices.insert(rnd_num());
    }
    auto rnd_index = rnd_indices.begin();
    for (std::size_t i=0; i < n_microstates; ++i) {
      for (std::size_t j=0; j < n_cols; ++j) {
        cluster_centers[i*n_cols+j] = coords[(*rnd_index)*n_cols+j];
      }
      rnd_index++;
    }
  }
  // calculate initial cluster assignments
  std::vector<std::size_t> cluster_assignment(n_rows);
  std::vector<std::size_t> cluster_population(n_microstates);
  std::vector<double> cluster_accumulation(n_microstates*n_cols);
  {
    for (std::size_t i=0; i < n_rows; ++i) {
      cluster_assignment[i] = 0;
      double min_dist = norm_p2(coords, cluster_centers, i, 0, n_cols);
      for (std::size_t c=1; c < n_microstates; ++c) {
        double dist = norm_p2(coords, cluster_centers, i, c, n_cols);
        if (dist < min_dist) {
          cluster_assignment[i] = c;
          min_dist = dist;
        }
      }
      cluster_population[cluster_assignment[i]] += 1;
      add_vector(cluster_accumulation, coords, cluster_assignment[i], i, n_cols);
    }
    cluster_centers = normalize_to_centroids(cluster_accumulation, cluster_population, n_cols);
  }
  // convergence: percentage no. of states changing cluster
  // TODO make these parameters
  float convergence_limit = 0.1f;
  const std::size_t MAX_ITER = 500;

  float convergence_delta = 1.0f;
  std::size_t n_iterations = 0;

  while (convergence_delta >= convergence_limit && n_iterations < MAX_ITER) {
    convergence_delta = 0.0f;
    // by triangle inequality:  (d(b,c) >= 2d(x,b)) => (d(x,c) > d(x,b))
    // assuming b,c are cluster centers and x is some frame-coordinate,
    // we can use this to skip calculation of d(x,c), if e.g. x is already assigned to b.
    // courtesy of: "Using the Triangle Inequality to Accelerate k-Means" by Charles Elkan.
    std::vector<double> cluster_cluster_dist(n_microstates*n_microstates);
    for (std::size_t i=0; i < n_microstates; ++i) {
      for (std::size_t j=0; j < n_microstates; ++j) {
        cluster_cluster_dist[i*n_microstates+j] = norm_p2(cluster_centers, cluster_centers, i, j, n_cols);
      }
    }
    // re-check cluster assignments
    for (std::size_t i=0; i < n_rows; ++i) {
      std::size_t assigned_c = cluster_assignment[i];
      std::size_t new_assigned_c = assigned_c;
      double min_dist = norm_p2(coords, cluster_centers, i, assigned_c, n_cols);
      for (std::size_t c=0; c < n_microstates; ++c) {
        // check 4*min_dist instead of 2*min_dist, since d(...) = norm_p2(...)^2,
        // thus, to compare the equivalent of 2*d(...), we need 2^2 * norm_p2.
        if (c != assigned_c
        &&  cluster_cluster_dist[c*n_microstates+new_assigned_c] <  (4*min_dist)) {
          double dist = norm_p2(coords, cluster_centers, i, c, n_cols);
          if (dist < min_dist) {
            new_assigned_c = c;
            min_dist = dist;
          }
        }
      }
      if (new_assigned_c != assigned_c) {
        cluster_population[assigned_c] -= 1;
        cluster_population[new_assigned_c] += 1;
        cluster_assignment[i] = new_assigned_c;
        convergence_delta += 1.0f;
        // correct cluster centers
        add_vector(cluster_accumulation, coords, new_assigned_c, i, n_cols);
        subtract_vector(cluster_accumulation, coords, assigned_c, i, n_cols);
      }
    }
    convergence_delta /= n_rows;
    cluster_centers = normalize_to_centroids(cluster_accumulation, cluster_population, n_cols);
  }
  double var = calculate_cluster_variance(coords, cluster_centers, cluster_assignment, n_cols);
  return {n_microstates, cluster_assignment, cluster_centers, n_cols, var};
}

} // end local namespace


void run_kmeans(ProjectData& project,
                std::size_t n_microstates,
                std::size_t n_iter,
                std::size_t n_threads) {
  if (project.coord_file.empty()) {
    throw std::invalid_argument("no coordinate file specified for kmeans");
  }
  std::vector<double> coords;
  std::size_t n_cols;
  std::size_t n_rows;
  // load coords from file and determine number of rows & columns
  {
    std::ifstream ifs(project.coord_file);
    // determine no. of columns from first line
    std::string linebuf;
    std::getline(ifs, linebuf);
    std::stringstream ss(linebuf);
    n_cols = std::distance(std::istream_iterator<std::string>(ss),
                           std::istream_iterator<std::string>());
    // go back to beginning and read complete file
    ifs.seekg(0);
    double buf;
    while (ifs.good()) {
      ifs >> buf;
      coords.push_back(buf);
    }
    n_rows = coords.size() / n_cols;
  }
  // run the actual algorithm 'n_iter' times ...
  std::size_t i;
  for (i=0; i < n_iter; ++i) {
    MicroStates ms = kmeans(n_microstates, coords, n_rows, n_cols, n_threads);
    if (project.microstates.count(n_microstates) == 0
    ||  ms.cluster_variance < project.microstates[n_microstates].cluster_variance) {
      project.microstates[n_microstates] = ms;
    }
  }
}

