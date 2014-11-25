
v_dir=$1

if [ ! -d $v_dir ]; then
  echo "$v_dir does not exist. will create it."
  mkdir $v_dir
fi

cp clustering.cpp\
   clustering.hpp\
   CMakeLists.txt\
   config.hpp.cmake.in\
   density_clustering.cpp\
   density_clustering.hpp\
   density_clustering_mpi.cpp\
   density_clustering_mpi.hpp\
   density_clustering_common.hpp\
   density_clustering_common.cpp\
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
   embedded_cytoscape.hpp\
   $v_dir/

tar cf $v_dir.tar $v_dir/
gzip $v_dir.tar

