/*
Copyright (c) 2015-2019, Florian Sittel (www.lettis.net) and Daniel Nagel
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
/*! \file
 * \brief Dynamical Coring
 *
 * \sa \link Clustering::Coring
 */

namespace Clustering {
  /*!
   * \brief functions related to dynamical coring
   *
   * This module contains all functions for finding the optimal coring time and
   * dynamical coring. The underlaying principal is to define dynamical cores.
   * In contrast to geometrical cores, this algorithm scales linear with the
   * the number of frames. The idea is to require after a transition to stay
   * at least for a certain time (coring time) in the new state, otherwise
   * the transition is identified as intrastate fluctuation and the frames are
   * are assigned to the previous state. A more in detail discussion can be
   * found in Nagel19.
   */
  namespace Coring {
    //! store the waiting time distribution for each state with time vs count.
    typedef std::map<std::size_t, float> WTDMap;

    /*!
     * \brief compute the waiting time distribution for single state
     *
     * \param streaks array with all wainting times
     */    
    WTDMap
    compute_wtd(std::list<std::size_t> streaks);

    /*!
     * \brief controlling function and user interface for boundary corrections
     *
     * \param states single column file with state information.
     * \param windows double column file with states and their coring time
     * \param distribution generate and write waiting time distributions to file.
     * \param output file name to store resulting trajectory.
     * \param concat-nframes no. of frames per trajectory.
     * \param concat-limits length of concated trajectories.
     * \param cores file name to store resulting cores.
     * \return void
     */
    void
    main(boost::program_options::variables_map args);
  } // end namespace Coring
} // end namespace Clustering

