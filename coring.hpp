#pragma once

#include <list>
#include <map>

#include <boost/program_options.hpp>

namespace Clustering {
namespace Coring {
  //typedef std::map<std::size_t, std::size_t, std::greater<std::size_t>> EtdMap;
  typedef std::map<std::size_t, float> WTDMap;
  
  // TODO doc
  WTDMap
  compute_wtd(std::list<std::size_t> streaks);

  void
  main(boost::program_options::variables_map args);

} // end namespace Coring
} // end namespace Clustering

