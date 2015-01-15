#pragma once

#include <list>
#include <map>

//typedef std::map<std::size_t, std::size_t, std::greater<std::size_t>> EtdMap;
typedef std::map<std::size_t, float> WTDMap;

// TODO doc
WTDMap
compute_wtd(std::list<std::size_t> streaks);

