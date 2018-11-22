/*
Copyright (c) 2015, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

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
*/
#pragma once

#include <list>
#include <map>

#include <boost/program_options.hpp>

namespace Clustering {
namespace Coring {
  //typedef std::map<std::size_t, std::size_t, std::greater<std::size_t>> EtdMap;
  typedef std::map<std::size_t, float> WTDMap;
  
  /*!
   * controlling function and user interface for boundary corrections
   *
   * *parsed arguments*:\n
   *    - **states**: single column file with state information.
   *    - **windows**: double column file with states and their coring time.
   *    - **output**: file name to store resulting trajectory
   *    - **distribution**: generate and write waiting time distributions to file.
   *    - **cores**: file name to store resulting cores.
   *    - **concat-nframes**: no. of frames per trajectory.
   *    - **concat-limits**: boundaries of trajectories.
   */
  WTDMap
  compute_wtd(std::list<std::size_t> streaks);

  void
  main(boost::program_options::variables_map args);

} // end namespace Coring
} // end namespace Clustering

