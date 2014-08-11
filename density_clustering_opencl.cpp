
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

namespace {
  std::map<std::string, cl::Kernel> kernels;
  
  // device buffer for coordinates
  cl::Buffer d_coords;
  
  std::vector<cl::Platform> platforms;
  cl::Context context;
  std::vector<cl::Device> devices;
  std::vector<cl::CommandQueue> queues;
  
  void
  setup_context() {
    try {
      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
      // create context
      context = cl::Context(devices);
      // create cmd queue
      // TODO: if that stuff works, scale to multiple devices
      qs.push_back(cl::CommandQueue(context, devices[0]));
    catch (cl::Error e) {
      std::cerr << "error during OpenCL setup" << std::endl;
      std::cerr << e.what() << " :  " << e.err() << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  void
  upload_coords(const float* coords,
                const std::size_t n_rows,
                const std::size_t n_cols) {
    // setup buffers on device
    cl::Buffer d_coords(context,
                        CL_MEM_READ_ONLY,
                        sizeof(float) * n_rows * n_cols,
                        NULL,
                        NULL);
    // transmit coords to device
    q.enqueueWriteBuffer(d_coords,
                         CL_TRUE,
                         0,
                         sizeof(float) * n_rows * n_cols,
                         coords,
                         NULL,
                         NULL);
    q.finish();
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
      kernel[name] = cl::Kernel(program, name.c_str());
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
      "                         __global ulong* result) {\n"
      "  ulong r=0;\n"
      "  for(ulong i=0; i < n_rows; ++i) {\n"
      "    r += uints[i]\n"
      "  }\n"
      "  *result = r;\n"
      "}\n"
    build_kernel("sum_uints", src);
  }
} // end local namespace


void
setup(const float* coords,
      const std::size_t n_rows,
      const std::size_t n_cols) {


}





std::vector<std::size_t>
calculate_populations(const float radius) {
  // vector for final results
  std::vector<std::size_t> pops(n_rows);



  try {
    ulong ulong_n_cols = (ulong) n_cols;
    ulong ulong_n_rows = (ulong) n_rows;
    cl::Buffer d_in_range(context,
                          CL_MEM_WRITE_ONLY,
                          sizeof(uint) * ulong_n_rows,
                          NULL,
                          NULL);
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
