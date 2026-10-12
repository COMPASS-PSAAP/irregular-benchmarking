#ifndef IRREGULAR_BENCHMARKING_SAMPLING_HPP
#define IRREGULAR_BENCHMARKING_SAMPLING_HPP

#include <map>
#include <vector>

// Draws a key from a normalized map<int, double> distribution, e.g. a
// Pattern's comm_partners or buffer_size table.
int sample_from_map(const std::map<int, double> &dist);

// Draws a message size in bytes from a Vernier buffer_size histogram: a bin by
// its count, then a point uniformly inside it. Vernier keys each bin by its lower
// edge, so the next edge closes it; the last bin is taken to be as wide as the
// first, as Vernier's own sampler (vernier_sample.py) assumes. Unlike that
// sampler, a draw in the first bin is spread over the bin too, rather than always
// giving its lower edge. A single bin is a single recorded size and is returned
// exactly.
int sample_interpolated(const std::vector<double> &edges, const std::vector<int> &counts);

#endif // IRREGULAR_BENCHMARKING_SAMPLING_HPP
