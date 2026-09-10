#ifndef IRREGULAR_BENCHMARKING_PATTERN_HPP
#define IRREGULAR_BENCHMARKING_PATTERN_HPP

#include <map>
#include <string>

#include <nlohmann/json.hpp>

// A communication pattern loaded from a benchmark config file: how many
// partners each rank talks to, how much data it sends, and how those
// partners are distributed relative to the rank, all normalized into
// probabilities.
struct Pattern {
    int pattern_count;
    int message_count;
    double comm_partners_count;
    std::map<int, double> comm_partners;                      // normalized
    std::map<int, double> buffer_size;                        // fill-forward + normalized
    std::map<int, std::map<int, double>> dist_to_neighbors;   // each inner map normalized
};

// Populates a Pattern from its JSON representation, filling forward zero
// entries and normalizing counts into probabilities. Also widens the
// global data_sent_max/nneighbors_max option bounds (see options.hpp) to
// fit this pattern.
void from_json(const nlohmann::json &j, Pattern &p);

extern std::map<std::string, Pattern> patterns;

#endif // IRREGULAR_BENCHMARKING_PATTERN_HPP
