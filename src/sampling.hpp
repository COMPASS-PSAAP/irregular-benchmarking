#ifndef IRREGULAR_BENCHMARKING_SAMPLING_HPP
#define IRREGULAR_BENCHMARKING_SAMPLING_HPP

#include <map>

// Draws a key from a normalized map<int, double> distribution, e.g. a
// Pattern's comm_partners or buffer_size table.
int sample_from_map(const std::map<int, double> &dist);

#endif // IRREGULAR_BENCHMARKING_SAMPLING_HPP
