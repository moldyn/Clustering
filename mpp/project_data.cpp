
#include "project_data.h"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

// implement serializer functions for BOOST serialization
namespace boost { namespace serialization {
  template <class Archive>
  void serialize(Archive& ar, MicroStates& ms, const unsigned int version) {
    ar & ms.n_microstates;
    ar & ms.trajectory;
    ar & ms.centers;
    ar & ms.n_data_cols;
    ar & ms.cluster_variance;
  }
  template <class Archive>
  void serialize(Archive& ar, ProjectData& pd, const unsigned int version) {
    ar & pd.coord_file;
    ar & pd.microstates;
  }
} // namespace boost
} // namespace serialization


ProjectData load_project(std::string fname) {
  ProjectData proj;
  std::ifstream ifs(fname);
  boost::archive::binary_iarchive ia(ifs);
  ia >> proj;
  return proj;
}

void save_project(std::string fname, ProjectData proj) {
  std::ofstream ofs(fname);
  boost::archive::binary_oarchive oa(ofs);
  oa << proj;
}

