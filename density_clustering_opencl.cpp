
#include "density_clustering_opencl.hpp"

#define VEXCL_BACKEND_OPENCL

#include "vexcl/vexcl.hpp"
#include "vexcl/devlist.hpp"

namespace DC_OpenCL {

  namespace { // local
    std::size_t n_gpus;
    std::vector<vex::Context> gpu_ctxs;

    std::size_t n_rows;
    std::size_t n_cols;
  } // end local namespace
  
  void
  setup(const float* coords, const std::size_t n_rows, const std::size_t n_cols) {
    n_gpus = vex::backend::device_list(vex::Filter::Type(CL_DEVICE_TYPE_GPU)).size();
    if (n_gpus == 0) {
      std::cerr << "error: no GPUs are available for OpenCL. "
                   "please compile the code for CPU-use on this machine or check your drivers." << std::endl;
      exit(EXIT_FAILURE);
    } else {
      for (std::size_t i=0; i < n_gpus; ++i) {
        // setup context for every GPU to use reshaping, slicing, etc.
        gpu_ctxs.push_back(vex::Context(vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::Position(i)));
      }
    }

  //TODO: transmit coords to gpus



  }
  
  std::vector<std::size_t>
  calculate_populations(const float radius) {
  
  //TODO: finish
  
  }


} // end namespace DC_OpenCL

