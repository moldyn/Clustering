#pragma once

#include <boost/program_options.hpp>

//TODO: doc
void density_main(boost::program_options::variables_map args);

/*
 * MPP clustering
 *  args:
 *   input (string)
 *   lagtime (int)
 *   qmin-from (float)
 *   qmin-to (float)
 *   qmin-step (float)
 */
void mpp_main(boost::program_options::variables_map args);

