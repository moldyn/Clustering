
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
      // has device enough memory for full data set?
      if (n_bytes_per_row * n_rows + (sizeof(int)*n_rows) < n_bytes_global_mem) {
        std::string kernel_src =
            "__kernel void"
            "pops(const int n_rows,"
            "     const int n_cols,"
            "     __constant float* coords"
            "     const float rad2,"
            "     __global unsigned int* pops) {"
            "  float row_i[20];"
            "  int k,j;"
            "  unsigned int pop_i = 0;"
            "  float dist2, tmp_d;"
            "  int i = get_global_id(0);"
            "  for (k=0; k < n_cols; ++k) {"
            "    row_i[k] = coords[i*n_cols+k];"
            "  }"
            "  for (j=0; j < n_rows, ++j) {"
            "    dist2 = 0.0f;"
            "    for (k=0; k < n_cols; ++k) {"
            "      tmp_d = row_i[k] - coords[j*n_cols+k];"
            "      dist2 += tmp_d * tmp_d;"
            "    }"
            "    pop_i += (unsigned int) (dist2 <= rad2);"
            "  }"
            "  pops[i] = pop_i;"
            "}"
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
        // copy data
        cl::Buffer buf_coords = cl::Buffer(context, CL_MEM_READ_ONLY, n_rows*n_cols*sizeof(float));
        cl::Buffer buf_pops = cl::Buffer(context, CL_MEM_WRITE_ONLY, n_rows*sizeof(unsigned int));
        queues[0].enqueueWriteBuffer(buf_coords, CL_TRUE, 0, n_rows*n_cols*sizeof(float), coords);
        // set kernel args
        kernel.setArg(0, n_rows);
        kernel.setArg(1, n_cols);
        kernel.setArg(2, buf_coords);
        kernel.setArg(3, radii[0]*radii[0]);
        kernel.setArg(4, buf_pops);
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
      } else {
        std::cerr << "error: not enough memory on GPU available." << std::endl;
        exit(EXIT_FAILURE);
      }
    } catch(cl::Error error) {
       std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
       exit(EXIT_FAILURE);
    }
  }

} // end namespace OpenCL
} // end namespace Density
} // end namespace Clustering

