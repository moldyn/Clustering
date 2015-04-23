
v_dir=$1

if [ ! -d $v_dir ]; then
  echo "$v_dir does not exist. will create it."
  mkdir $v_dir
fi

cp -r clustering.cpp\
      CMakeLists.txt\
      config.hpp.cmake.in\
      density_clustering.cpp\
      density_clustering.hpp\
      density_clustering_mpi.cpp\
      density_clustering_mpi.hpp\
      density_clustering_common.hpp\
      density_clustering_common.cpp\
      FindOpenCL.cmake\
      cl.hpp\
      density_clustering_opencl.hpp\
      density_clustering_opencl.cpp\
      pops.cl\
      generate_header.py\
      logger.cpp\
      logger.hpp\
      mpp.cpp\
      mpp.hpp\
      network_builder.cpp\
      network_builder.hpp\
      README.md\
      state_filter.cpp\
      state_filter.hpp\
      tools.cpp\
      tools.hpp\
      tools.hxx\
      coring.cpp\
      coring.hpp\
      embedded_cytoscape.hpp\
      doc\
      $v_dir/

tar cf $v_dir.tar $v_dir/
gzip $v_dir.tar

