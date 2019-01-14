/*
Copyright (c) 2015-2019, Florian Sittel (www.lettis.net) and Daniel Nagel
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
      std::vector<cl::Buffer> buf_pops(n_queues);
      for (int iq=0; iq < n_queues; ++iq) {
        queues[iq].enqueueWriteBuffer(buf_coords, CL_TRUE, 0, n_rows*n_cols*sizeof(float), coords);
        buf_pops[iq] = cl::Buffer(context, CL_MEM_WRITE_ONLY, n_rows*sizeof(unsigned int));
      }
//TODO: fix
      // load kernel source
//      std::string kernel_src =
//        #include "kernel/pops.h"
//      ;
      cl::Program::Sources src(1, {kernel_src.c_str(), kernel_src.length()+1});
      cl::Program prog = cl::Program(context, src);
      int err = prog.build(devices);
      if (err != CL_SUCCESS) {
        std::cerr << "error during kernel compilation" << std::endl;
        std::cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        exit(EXIT_FAILURE);
      }
      // initialize pops for every queue to zero
      cl::Kernel krnl_init_pops(prog, "init_pops");
      for (int iq=0; iq < n_queues; ++iq) {
        krnl_init_pops.setArg(0, buf_pops[iq]);
        cl::NDRange full_global_range(n_rows);
        cl::Event event;
        queues[iq].enqueueNDRangeKernel(krnl_init_pops, cl::NullRange, full_global_range, cl::NullRange, NULL, &event);
        // very short and simple kernel: just do a blocking call
        event.wait();
      }
      cl::Kernel krnl_pops(prog, "pops");
      // set kernel args
      float rad2 = radii[0]*radii[0];
      // TODO play with workgroup sizes
      const unsigned int GLOBAL_SIZE = 1024;
      const unsigned int WORKGROUP_SIZE = 128;
      krnl_pops.setArg(2, sizeof(unsigned int), &uint_n_rows);
      krnl_pops.setArg(3, sizeof(unsigned int), &uint_n_cols);
      krnl_pops.setArg(4, buf_coords);
      krnl_pops.setArg(5, sizeof(float), &rad2);
      krnl_pops.setArg(7, sizeof(float)*WORKGROUP_SIZE*n_cols, NULL);
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
        krnl_pops.setArg(1, sizeof(unsigned int), &i_block_ref);
        for (unsigned int i_block=0; i_block < range_length; i_block += n_queues*GLOBAL_SIZE) {
          std::vector<cl::Event> events(n_queues);
          for (unsigned int iq=0; iq < n_queues; ++iq) {
            unsigned int q_block = i_block + iq*GLOBAL_SIZE;
            krnl_pops.setArg(0, sizeof(unsigned int), &q_block);
            krnl_pops.setArg(6, buf_pops[iq]);
            queues[iq].enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &events[iq]);
          }
          cl::Event::waitForEvents(events);
        }
      }
      // retrieve results from all devices
      std::vector<std::size_t> pops(n_rows, 0);
      for (int iq=0; iq < n_queues; ++iq) {
        std::vector<unsigned int> partial_pops(n_rows);
        queues[iq].enqueueReadBuffer(buf_pops[iq], CL_TRUE, 0, n_rows*sizeof(unsigned int), partial_pops.data());
        for (std::size_t i=0; i < n_rows; ++i) {
          pops[i] += partial_pops[i];
        }
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

