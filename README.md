


This software package provides extensive tools for fast, robust and stable
clustering of molecular dynamics trajectories.
The essential functions are:
  - density-based geometric clustering for microstate generation
  - dynamic clustering based on the Most-Probable-Path algorithm (MPP)
  - variable dynamic coring for boundary corrections
  - dynamic noise assignment.

Additionally, the package includes tools to efficiently filter original
coordinates or order parameters based on a discrete state definition
to identify representative structures and variables of clusters.

<!--
Computationally demanding functions are parallelized in a hybrid model using
OpenMP for SMP parallelization on a single node (multithreading)
and MPI over different cluster nodes. MPI support, however, is optional and
for a modern computer with a high number of fast cores or
even multiple CPUs, OpenMP parallelization is sufficiently fast.
-->
Computationally demanding functions are parallelized using CUDA or OpenMP.

# Documentation
All options are well documented and may be viewed by 'clustering -h'.

The 'doc' directory includes an extensive tutorial which describes all technical
details in performing a complete clustering run from trajectory to markov state model.

The source code itself is additionally documented via doxygen. Run 'make doc' in
the build directory (see below for installation intructions) to compile the source
code documentation in html.


# Citations
The underlying methods are based on the following articles:
  - F. Sittel and G. Stock, *Robust Density-Based Clustering to Identify
    Metastable Conformational States of Proteins*,
    J. Chem. Theory Comput., 12, 2426; DOI: 10.1021/acs.jctc.5b01233
  - A. Jain and G. Stock, *Hierarchical folding free energy landscape of HP35
    revealed by most probable path clustering*,
    J. of Phys. Chem. B, 118, 7750 - 7760, 2014; DOI: 10.1021/jp410398a
  - D. Nagel, A. Weber, B. Lickert and G. Stock, *Dynamical coring of Markov state models*, J. Chem. Phys, Accepted

We kindly ask you to cite these articles if you use this software package for
published works.


# Licensing
This project was created by [lettis](http://www.lettis.net) and is currently maintained by [moldyn-nagel](https://github.com/moldyn-nagel).

Copyright (c) 2015-2018, [Florian Sittel](http://www.lettis.net) and Daniel Nagel
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

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



# Installation
## Requirements
 required:
  -  **BOOST >= 1.49**
  -  **cmake >= 2.8**
  -  a **recent** C++ compiler (e.g. GNU g++ >= 4.9, must
     support C++11 standard)

 optional:
  - **CUDA >= 9.1**
  - doxygen (to build the API docs)
<!--  - MPI (for parallelized execution on clusters) -->

## Quick-Start

To quickly get a working (but possibly underperforming) binary

  - unpack the code ...
```
    # tar xfz clustering_VERSION.tar.gz
```
  - create a build folder inside the code directory ...
```
    # cd clustering_VERSION
    # mkdir build
```
  - change to the build directory ...
```
    # cd build
```
  - ... and run cmake
```
    # cmake .. -DCMAKE_INSTALL_PREFIX=/my/installation/path
```
  - then compile and install the package to /my/installation/path (or any other
    path you chose above) by invoking
```
    # make
    # make install 
```

## Optimized Binaries and Custom Build Options

### CUDA
If you have an Nvidia graphic card it can be used to significantly speed up the
*clustering density* method, by setting the following **cmake-option**:
-DUSE_CUDA=1

### Vectorization
If you have a modern computer with vectorizing instruction sets (SSE2, SSE4_2,
AVX, ...), set the following **cmake-option**: -DCPU_ACCELERATION=<OPTION>,
where <OPTION> is one of
  - SSE2
  - SSE4_1
  - SSE4_2
  - AVX

It is important to select an option that is actually supported by your machine.
Otherwise the program will produce erratic results, crash or not compile at all.
On linux systems, you can check your computer's capabilities with
```
    # cat /proc/cpuinfo
```
Check in the *flags:*-block, if a certain instruction set is supported.
If it is not listed, it is not supported.

### Native Compilation
To compile the code with '-march=native' option (specific to the GNU compiler),
add '-DNATIVE_COMPILATION=ON' to your cmake-flags.
Using this option, the GNU compiler will automatically use all available
instruction sets for optimal performance (**attention**: you still
need to set the vectorization option above, even if you use this option).

Unfortunately, the resulting binary will most likely run only on the computer
it was compiled on - do **not** use this option if you want
to distribute the binary, e.g. on a cluster.

<!--
### MPI
For MPI support, build your binary with the additional cmake-flag -DDC_USE_MPI=ON.
Invoke the *clustering_mpi* binary in the following way to run on several nodes with local multithreading via OpenMP:

     # /usr/bin/mpirun -n N_NODES -bind-to-core -bynode -cpus-per-proc N_THREADS_PER_NODE -report-bindings  \
          clustering_mpi density -f COORDS_FILE -r RADIUS -p POPS_OUT -d FE_OUT -n N_THREADS_PER_NODE
-->

