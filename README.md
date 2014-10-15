
# density\_clustering (version 0.5)
... a fast code for density-based clustering of MD data

## installation

### requirements

* __BOOST >= 1.49__ (_program\_options_ package)
* __cmake >= 2.8__
* a __recent C++__ compiler (e.g. GNU g++ 4.9 or Intel icpc 14, must support C++11 standard)

### the fast approach

* unpack the code, e.g.
 
    \# tar xfz density\_clustering\_0.5.tar.gz

* in the code directory, create a build folder and run cmake in there

    \# cd density\_clustering\_0.5  
    \# mkdir build  
    \# cd build  
    \# cmake ..  

* compile the program

    \# make

congratulations! you should now have a functional binary
(if all requirements mentioned above have been met...).


### the approach for fast programs

if you have a modern machine with vectorizing instruction sets (e.g. SSE4_2 or AVX)
and / or multiple cores, regard the following __cmake-options__:

* __-DDC\_CPU\_ACCELERATION__=OPTION sets vectorizing instruction set; possible options:
    * SSE2
    * SSE4\_1
    * SSE4\_2
    * AVX

it is important to select an option that is actually supported by your machine.
otherwise the program will produce erratic results, crash or not compile at all.
on linux, you can check your computer's capabilities with

    # cat /proc/cpuinfo

check in the _flags:_ block, if a certain instruction set is supported.
if it is not listed, it is not.


* __-DDC\_NATIVE\_COMPILATION__=ON uses native compilation, i.e. with _-march=native_.

this option is specific to the GNU compiler and will compile the code with all available
optimization flags supported by the machine the compiler detects during compilation.  
_Attention_: with this option, the compiled binary will most likely run only on the
machine it was compiled on. do not use it, if you want to execute the binary
on several different computers.

* __Parallelization__ is based on OpenMP (i.e. shared-memory only) and always compiled automatically into the code.

tests (on a Core-i7 with six real + six hyperthreaded cores)
have shown that the code runs at highest efficiency when all _real_ cores of a computer are used.
using logic (e.g. hyperthreaded) cores only slows it down.
however, every computer is different and it is worthwhile to do some benchmarking on your own.



### MPI

/usr/bin/mpirun -n 2 -bind-to-core -bynode -cpus-per-proc 2 -report-bindings   clustering_mpi density -f test.dat -r 0.5 -p pop_mpi -d fe_mpi -v -n 2





### get it running with the intel compiler (**ATTENTION**: currently not supported!)

This is currently not supported, but preliminary intel compiler support was implemented
and is currently commented out in CMakeLists.txt.
If you really want to get your hands dirty and/or need support for the intel compiler,
you can use the code and adapt it to your needs.

* be sure to have a BOOST version compiled with your intel compiler
  (i.e. not the standard-lib compiled with gcc)
    * download boost
    * unpack it and run ./bootstrap --prefix=[...]
    * compile it (with ./b2)

* compile with custom boost install:
      cmake .. -DCMAKE\_CXX\_COMPILER=icpc
               -DBoost\_NO\_SYSTEM\_PATHS=ON
               -DBoost\_NO\_BOOST\_CMAKE=ON
               -DBOOST\_INCLUDEDIR=/vol/home/fs172/lib/bmdphi1/include/
               -DBOOST\_LIBRARYDIR=/vol/home/fs172/lib/bmdphi1/lib/

