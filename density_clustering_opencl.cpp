
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

namespace { // local namespace
  std::map<std::string, cl::Kernel> kernels;
  
  ulong ul_n_rows;
  ulong ul_n_cols;
  ulong ul_n_size;

  // device buffer for coordinates
  cl::Buffer d_coords;
  
  std::vector<cl::Platform> platforms;
  cl::Context context;
  std::vector<cl::Device> devices;
  std::vector<cl::CommandQueue> queues;
  
  void
  setup_context() {
    try {
      // TODO: if that stuff works, scale to multiple devices
      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
      std::cout << "using platform 0: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
      // create context
      context = cl::Context(devices);
      // create cmd queue
      queues.push_back(cl::CommandQueue(context, devices[0]));
    } catch (cl::Error e) {
      std::cerr << "error during OpenCL setup" << std::endl;
      std::cerr << e.what() << " :  " << e.err() << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  void
  upload_coords(const float* coords,
                const std::size_t n_rows,
                const std::size_t n_cols) {
    ul_n_rows = (ulong) n_rows;
    ul_n_cols = (ulong) n_cols;
    ul_n_size = ul_n_rows * ul_n_cols;
    // setup buffers on device
    d_coords = cl::Buffer(context,
                          CL_MEM_READ_ONLY,
                          sizeof(float) * ul_n_size,
                          NULL,
                          NULL);
    // transmit coords to device
    queues[0].enqueueWriteBuffer(d_coords,
                                 CL_TRUE,
                                 0,
                                 sizeof(float) * ul_n_size,
                                 coords,
                                 NULL,
                                 NULL);
    queues[0].finish();
  }
  
  void
  build_kernel(std::string name, std::string src) {
    cl::Program program;
    try {
      // setup kernel source
      cl::Program::Sources source(1, std::make_pair(src.c_str(), src.length() + 1));
      // create program
      program = cl::Program(context, source);
      program.build(devices);
      // load kernel
      kernels[name] = cl::Kernel(program, name.c_str());
    } catch (cl::Error e) {
      std::cout << e.what() << " :  " << e.err() << std::endl;
      if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
        cl::STRING_CLASS buildlog;
        program.getBuildInfo(devices[0], (cl_program_build_info)CL_PROGRAM_BUILD_LOG, &buildlog);
        std::cerr << buildlog << std::endl;
      }
      exit(EXIT_FAILURE);
    }
  }
  
  void
  setup_kernels() {
    std::string src;
  
    //// pop_i
    //
    // kernel calculates population of frame i
    // by comparing distances to all frames j.
    // population is stored in integer-array of length n_rows.
    // every field of the array gets a zero if j is not in
    // range of i and a one if it is.
    src =
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
    build_kernel("pop_i", src);

    //// sum_uints (single task kernel)
    //
    // kernel calculates total sum of unsigned integer array.
    src =
      "__kernel void sum_uints (__global uint* uints,\n"
      "                         ulong n_rows,\n"
      "                         __global ulong* result,\n"
      "                         ulong i_result) {\n"
      "  ulong r=0;\n"
      "  for(ulong i=0; i < n_rows; ++i) {\n"
      "    r += uints[i]\n"
      "  }\n"
      "  result[i_result] = r;\n"
      "}\n"
    ;
    build_kernel("sum_uints", src);
  }
} // end local namespace




void
setup(const float* coords,
      const std::size_t n_rows,
      const std::size_t n_cols) {
  setup_context();
  setup_kernels();
  upload_coords(coords, n_rows, n_cols);
}

std::vector<std::size_t>
calculate_populations(const float radius) {
  // vector for final results
  std::vector<ulong> pops(ul_n_rows);
  const float rad2 = radius * radius;
  try {
    // buffer to hold info per j if it is no
    // range of i or not.
    // values will be equal to one or zero.
    cl::Buffer d_in_range(context,
                          CL_MEM_WRITE_ONLY,
                          sizeof(uint) * ul_n_rows,
                          NULL,
                          NULL);
    // set common arguments
    kernels["pop_i"].setArg(0, d_coords);
    kernels["pop_i"].setArg(1, ul_n_cols);
    kernels["pop_i"].setArg(2, rad2);
    kernels["pop_i"].setArg(3, d_in_range);
    // screen all frames i
    for (ulong i=0; i < ul_n_rows; ++i) {
      kernels["pop_i"].setArg(4, i);
      // run kernel
      queues[0].enqueueNDRangeKernel(kernels["pop_i"], cl::NullRange, cl::NDRange(ul_n_rows), cl::NullRange, NULL, NULL);
      queues[0].finish();
      // run reduction and save result
      queues[0].enqueueTask(kernels["sum_uints"], NULL, NULL);
      queues[0].finish();
    }
  } catch (cl::Error e) {
    std::cout << e.what() << " :  " << e.err() << std::endl;
    exit(EXIT_FAILURE);
  }
  return pops;
}

} // end namespace DC_OpenCL

