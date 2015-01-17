
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
      std::size_t n_queues = queues.size();
      // compute needed/avail mem sizes
      std::size_t n_bytes_per_row;
      std::size_t n_bytes_global_mem;
      n_bytes_per_row = n_cols * sizeof(float);
      n_bytes_global_mem = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
      // do devices have enough memory for full data set?
      if (n_bytes_per_row * n_rows + (sizeof(unsigned int)*n_rows) > n_bytes_global_mem) {
        // TODO compute in chunks
        std::cerr << "error: not enough memory on device for full data set.\n"
                  << "       will support memory-size independet computation in a later release.\n"
                  << "       please check for updates." << std::endl;
        exit(EXIT_FAILURE);
      }
      // create buffers and copy data to device(s)
      cl::Buffer buf_coords = cl::Buffer(context, CL_MEM_READ_ONLY, n_rows*n_cols*sizeof(float));
      //TODO initialize pops to zero (use init kernel)
      cl::Buffer buf_pops = cl::Buffer(context, CL_MEM_WRITE_ONLY, n_rows*sizeof(unsigned int));
      for (int iq=0; iq < n_queues; ++iq) {
        queues[iq].enqueueWriteBuffer(buf_coords, CL_TRUE, 0, n_rows*n_cols*sizeof(float), coords);
      }
      // load kernel source
      std::string kernel_src =
        #include "kernel/pops.h"
      ;
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
      // TODO play with workgroup sizes
      const unsigned int GLOBAL_SIZE = 1024;
      const unsigned int WORKGROUP_SIZE = 128;
      err  = kernel.setArg(2, sizeof(unsigned int), &uint_n_rows);
      err |= kernel.setArg(3, sizeof(unsigned int), &uint_n_cols);
      err |= kernel.setArg(4, buf_coords);
      err |= kernel.setArg(5, sizeof(float), &rad2);
      err |= kernel.setArg(6, buf_pops);
      err |= kernel.setArg(7, sizeof(float)*WORKGROUP_SIZE*n_cols, NULL);
      if (err != CL_SUCCESS) {
        std::cerr << "error while setting kernel arguments" << std::endl;
        exit(EXIT_FAILURE);
      }
      std::cout << "running kernel" << std::endl;
      //// run kernel
      // extend range to a size that is evenly divisible by the workgroup size.
      // the kernel will ignore all workitems with index >= n_cols.
      unsigned int range_length = n_rows;
      while (range_length % WORKGROUP_SIZE != 0) {
        ++range_length;
      }
      cl::NDRange global(GLOBAL_SIZE);
      cl::NDRange local(WORKGROUP_SIZE);
      // run pops kernel repeatedly, until the full range has been sampled
      for (unsigned int i_block_ref=0; i_block_ref < range_length; i_block_ref += WORKGROUP_SIZE) {
        kernel.setArg(1, sizeof(unsigned int), &i_block_ref);
        for (unsigned int i_block=0; i_block < range_length; i_block += n_queues*GLOBAL_SIZE) {
          std::vector<cl::Event> events(n_queues);
          for (unsigned int iq=0; iq < n_queues; ++iq) {
            unsigned int q_block = i_block + iq*GLOBAL_SIZE;
            kernel.setArg(0, sizeof(unsigned int), &q_block);
            queues[iq].enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &events[iq]);
          }
          cl::Event::waitForEvents(events);
        }
      }
      // retrieve results
      // TODO from all devices!
      std::vector<unsigned int> result(n_rows);
      queues[0].enqueueReadBuffer(buf_pops, CL_TRUE, 0, n_rows*sizeof(unsigned int), result.data());
      std::vector<std::size_t> pops;
      for (auto i: result) {
        pops.push_back(i);
      }
      return {{radii[0], pops}};
    } catch(cl::Error error) {
       std::cerr << "error in OpenCL call: " << error.what() << "(" << error.err() << ")" << std::endl;
       exit(EXIT_FAILURE);
    }
  }

} // end namespace OpenCL
} // end namespace Density
} // end namespace Clustering

