
#define __CL_ENABLE_EXCEPTIONS

#include "density_clustering_opencl.hpp"

#include <iostream>

namespace Clustering {
namespace Density {
namespace OpenCL {

  std::map<float, std::vector<std::size_t>>
  calculate_populations(const float* coords,
                        const std::size_t n_rows,
                        const std::size_t n_cols,
                        std::vector<float> radii) {
    
    // TODO split up work into smaller chunks and run
    //      these separately with individual kernels.

    unsigned int uint_n_rows = (unsigned int) n_rows;
    unsigned int uint_n_cols = (unsigned int) n_cols;

    try {

      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) (platforms[0])(),
        0
      };
      cl::Context context(CL_DEVICE_TYPE_GPU, cps);
      std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
      std::vector<cl::CommandQueue> queues;
      for (cl::Device dev: devices) {
        queues.push_back(cl::CommandQueue(context, dev));
      }

      // compute needed/avail mem sizes
      std::size_t n_bytes_per_row;
      std::size_t n_bytes_global_mem;
      {
        cl::Device d = devices[0];
        n_bytes_per_row = n_cols * sizeof(float);
        n_bytes_global_mem = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
      }
      // have devices enough memory for full data set?
      if (n_bytes_per_row * n_rows + (sizeof(unsigned int)*n_rows) < n_bytes_global_mem) {
        // copy data to devices
        cl::Buffer buf_coords = cl::Buffer(context, CL_MEM_READ_ONLY, n_rows*n_cols*sizeof(float));
        queues[0].enqueueWriteBuffer(buf_coords, CL_TRUE, 0, n_rows*n_cols*sizeof(float), coords);
      } else {
        // TODO compute in chunks
        std::cerr << "error: not enough memory on device for full data set.\n"
                  << "       will support memory-size independet computation in a later release.\n"
                  << "       please check for updates." << std::endl;
        exit(EXIT_FAILURE);
      }

      // TODO load kernel src as include

      cl::Program::Sources src(1, {kernel_src.c_str(), kernel_src.length()+1});
      cl::Program prog = cl::Program(context, src);
      int err = prog.build(devices);
      if (err != CL_SUCCESS) {
        std::cerr << "error during kernel compilation" << std::endl;
        std::cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        exit(EXIT_FAILURE);
      }
      cl::Kernel kernel(prog, "pops");
      // set kernel args
      float rad2 = radii[0]*radii[0];
      err = kernel.setArg(0, sizeof(unsigned int), &uint_n_rows);
      err |= kernel.setArg(1, sizeof(unsigned int), &uint_n_cols);
      err |= kernel.setArg(2, buf_coords);
      err |= kernel.setArg(3, sizeof(float), &rad2);
      err |= kernel.setArg(4, buf_pops);
      if (err != CL_SUCCESS) {
        std::cerr << "error while setting kernel arguments" << std::endl;
        exit(EXIT_FAILURE);
      }
      // run kernel
      // TODO play with workgroup sizes
      cl::NDRange global(n_rows);
      cl::NDRange local(1);
      queues[0].enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
      // retrieve results
      std::vector<unsigned int> result(n_rows);
      queues[0].enqueueReadBuffer(buf_pops, CL_TRUE, 0, n_rows*sizeof(int), result.data());
      std::vector<std::size_t> pops;
      for (auto i: result) {
        pops.push_back(i);
      }
      return {{radii[0], pops}};
    } catch(cl::Error error) {
       std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
       exit(EXIT_FAILURE);
    }
  }

} // end namespace OpenCL
} // end namespace Density
} // end namespace Clustering

