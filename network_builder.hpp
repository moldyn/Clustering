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

#include <map>
#include <iostream>

#include <boost/program_options.hpp>
/*! \file
 * \brief Network Builder
 *
 * \sa \link Clustering::NetworkBuilder
 */
namespace Clustering {
  /*!
   * \brief functions for network creation from free energy screening
   *
   * This module finds local minimas in the free energy landscape. Therefore,
   * it build a network from the previously determined clusters.
   */
  namespace NetworkBuilder {
    /*!
     * controlling function and user interface for network creation
     *
     * \param min min. free energy to take into account
     * \param max max. free energy to take into account
     * \param step free energy stepping
     * \param basename basic input file format
     * \param minpop min. population per microstate, discard states with lower population
     * \param network-html generate an html representation of the network
     */
    void
    main(boost::program_options::variables_map args);
  } // end namespace Clustering
} // end namespace NetworkBuilder

