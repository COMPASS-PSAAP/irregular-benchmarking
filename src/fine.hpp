#ifndef IRREGULAR_BENCHMARKING_FINE_HPP
#define IRREGULAR_BENCHMARKING_FINE_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "options.hpp"

// One rank's own records from a Vernier capture of a single pattern
// (vernier-<rank>-0.hist-hierarchical.json, see scripts/vernier-config.py):
// per MPI call type, per partner rank, a histogram of message sizes.
struct SizeBins {
    std::vector<double> edges;  // bin lower edges in bytes, ascending
    std::vector<int> counts;    // messages per bin
};

struct FinePattern {
    std::string name;  // fine_<pattern name Vernier recorded>
    std::map<int, std::map<int, SizeBins>> calls;  // call type -> partner rank -> sizes
};

// Reads this rank's file from a capture directory. Exits with an error message
// if the file is missing, holds more than one pattern, or was captured at
// another number of ranks than this run has.
FinePattern read_fine_pattern(const std::string &dir);

// This rank's messages for one sample, as (partner, bytes): one per recorded
// send (EXPORT, whose lists name what this rank sends) or receive (IMPORT,
// whose lists name what it receives), each to or from its recorded partner,
// with a size drawn within that partner's bins (exact when the partner has one
// bin, as in a capture of one exchange). A capture of k exchanges records k
// messages per partner per exchange, so captures are of one exchange.
std::vector<std::pair<int, int>> fine_messages(const FinePattern &pattern, halo_t type);

// Number of records fine_messages draws for this halo type.
int fine_message_count(const FinePattern &pattern, halo_t type);

extern bool fine_input;          // true when the input is a Vernier capture directory
extern FinePattern fine_pattern; // filled by cli::parseArgs() when fine_input

#endif // IRREGULAR_BENCHMARKING_FINE_HPP
