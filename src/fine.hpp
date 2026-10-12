#ifndef IRREGULAR_BENCHMARKING_FINE_HPP
#define IRREGULAR_BENCHMARKING_FINE_HPP

#include <map>
#include <string>
#include <vector>

// One rank's own histograms from a Vernier capture of a single pattern
// (vernier-<rank>-0.hist-combined.json, see scripts/vernier-config.py). Vernier
// counts sends and receives together in `neighbors` and `buffer_size`; only
// `call_type` tells them apart.
struct FinePattern {
    std::string name;               // fine_<pattern name Vernier recorded>
    int sends = 0;                  // messages this rank sent (send call types)
    std::map<int, double> neighbors;  // partner rank -> messages, normalized
    std::vector<double> size_edges;   // buffer_size bin lower edges, bytes, ascending
    std::vector<int> size_counts;     // messages per bin
};

// Reads this rank's file from a capture directory. Exits with an error message
// if the file is missing, holds more than one pattern, or was captured at
// another number of ranks than this run has.
FinePattern read_fine_pattern(const std::string &dir);

extern bool fine_input;          // true when the input is a Vernier capture directory
extern FinePattern fine_pattern; // filled by cli::parseArgs() when fine_input

#endif // IRREGULAR_BENCHMARKING_FINE_HPP
