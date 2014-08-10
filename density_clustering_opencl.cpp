
#include "density_clustering_opencl.hpp"
#include "tools.hpp"

#define __CL_ENABLE_EXCEPTIONS

#include <utility>
#include <CL/cl.hpp>
#include <CL/opencl.h>

#include <omp.h>

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <tuple>

namespace DC_OpenCL {

std::tuple<cl::Context, cl::Kernel, cl::CommandQueue>
setup_cl(std::string kernel_name, std::string src) {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Program program;
  cl::Context context;
  cl::CommandQueue q;
  cl::Kernel kernel;
  try {
    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    // create context
    context = cl::Context(devices);
    // create cmd queue
    // TODO: if that stuff works, scale to multiple devices
    q = cl::CommandQueue(context, devices[0]);
    // setup kernel source
    cl::Program::Sources source(1, std::make_pair(src.c_str(), src.length() + 1));
    // create program
    program = cl::Program(context, source);
    program.build(devices);
    // load kernel
    kernel = cl::Kernel(program, kernel_name.c_str());
  } catch (cl::Error e) {
    std::cout << e.what() << " :  " << e.err() << std::endl;
    if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
      cl::STRING_CLASS buildlog;
      program.getBuildInfo(devices[0], (cl_program_build_info)CL_PROGRAM_BUILD_LOG, &buildlog);
      std::cerr << buildlog << std::endl;
    }
    exit(EXIT_FAILURE);
  }
  return std::make_tuple(context, kernel, q);
}

std::vector<std::size_t>
calculate_populations(const float* coords,
                      const std::size_t n_rows,
                      const std::size_t n_cols,
                      const float radius) {
  ASSUME_ALIGNED(coords);
  // vector for final results
  std::vector<std::size_t> pops(n_rows);
  // kernel calculates population of frame i
  // by comparing distances to all frames j.
  // population is stored in integer-array of length n_rows.
  // every field of the array gets a zero if j is not in
  // range of i and a one if it is.
  // to get the total population of i, reduce array with a simple
  // sum on the host.
  std::string src =
    "__kernel void pop_i (__global float* coords,\n"
    "                     ulong n_cols,\n"
    "                     float rad2,\n"
    "                     __global uint* in_range,\n"
    "                     ulong i) {\n"
    "  ulong j = get_global_id(0);\n"
    "  float dist = 0.0f;\n"
    "  float d;\n"
    "  ulong k;\n"
    "  for (k=0; k < n_cols; ++k) {\n"
    "    d = coords[i*n_cols+k] - coords[j*n_cols+k];\n"
    "    dist += d*d;\n"
    "  }\n"
    "  in_range[j] = dist < rad2 ? 1 : 0;\n"
    "}\n"
  ;
  try {
    auto cl_context_kernel_queue = setup_cl("pop_i", src);
    cl::Context context = std::get<0>(cl_context_kernel_queue);
    cl::Kernel kernel = std::get<1>(cl_context_kernel_queue);
    cl::CommandQueue q = std::get<2>(cl_context_kernel_queue);
    ulong ulong_n_cols = (ulong) n_cols;
    ulong ulong_n_rows = (ulong) n_rows;
    // setup buffers on device
    cl::Buffer d_coords(context,
                        CL_MEM_READ_ONLY,
                        sizeof(float) * ulong_n_rows * ulong_n_cols,
                        NULL,
                        NULL);
    cl::Buffer d_in_range(context,
                          CL_MEM_WRITE_ONLY,
                          sizeof(uint) * ulong_n_rows,
                          NULL,
                          NULL);
    // transmit coords to device
    q.enqueueWriteBuffer(d_coords,
                         CL_TRUE,
                         0,
                         sizeof(float) * ulong_n_rows * ulong_n_cols,
                         coords,
                         NULL,
                         NULL);
    q.finish();
    // set common arguments
    float rad2 = radius*radius;
    kernel.setArg(0, d_coords);
    kernel.setArg(1, ulong_n_cols);
    kernel.setArg(2, rad2);
    kernel.setArg(3, d_in_range);
    // screen all frames i
    for (ulong i=0; i < ulong_n_rows; ++i) {
      kernel.setArg(4, i);
      // run kernel
      q.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ulong_n_rows), cl::NullRange, NULL, NULL);
      // wait for completion
      q.finish();
      std::vector<uint> buf(ulong_n_rows);
      q.enqueueReadBuffer(d_in_range, CL_TRUE, 0, sizeof(uint) * ulong_n_rows, buf.data(), NULL, NULL);

      //TODO: run reduction on device instead of host
      
      std::size_t j; 
      std::size_t pop_i = 0;
      #pragma omp parallel for default(none) \
                               private(j) \
                               firstprivate(n_rows) \
                               shared(pops,buf) \
                               reduction(+:pop_i)
      for (j=0; j < n_rows; ++j) {
        pop_i += buf[j];
      }
      pops[i] = pop_i;
    }
  } catch (cl::Error e) {
    std::cout << e.what() << " :  " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
  return pops;
}

} // end namespace DC_OpenCL
