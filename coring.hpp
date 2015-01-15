#pragma once

#include <list>
#include <map>

typedef std::map<std::size_t, std::size_t, std::greater<std::size_t>> EtdMap;

// TODO doc
EtdMap
compute_etd(std::list<std::size_t> streaks);

