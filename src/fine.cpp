#include "fine.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>

#include <mpi.h>
#include <nlohmann/json.hpp>

bool fine_input = false;
FinePattern fine_pattern;

namespace {

[[noreturn]] void fail(const std::string &message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "ERROR (rank " << rank << "): " << message << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

// Vernier's codes for MPI_Irsend, MPI_Isend, MPI_Send and MPI_Ssend (the send
// symbols IQ, IS, S, SI of its vernier_sample.py).
const std::set<int> SEND_CALLS = {18, 21, 30, 31};

} // namespace

FinePattern read_fine_pattern(const std::string &dir) {
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string path = dir + "/vernier-" + std::to_string(rank) + "-0.hist-combined.json";
    std::ifstream file(path);
    if (!file.is_open()) {
        fail("could not open " + path);
    }
    // [{"pattern_map": "name:id"}, {"comm_map": "size|..."}, {"pattern_<id>": {...}}]
    nlohmann::json j;
    file >> j;
    const std::string pattern_map = j.at(0).at("pattern_map").get<std::string>();
    const std::size_t colon = pattern_map.rfind(':');
    if (colon == std::string::npos || pattern_map.find(',') != std::string::npos) {
        fail(path + " must hold exactly one pattern, not \"" + pattern_map + "\"");
    }
    const std::string comm_map = j.at(1).at("comm_map").get<std::string>();
    const int captured_size = std::stoi(comm_map.substr(0, comm_map.find('|')));
    if (captured_size != size) {
        fail(path + " was captured on " + std::to_string(captured_size) + " ranks, this run has " +
             std::to_string(size));
    }

    FinePattern p;
    p.name = "fine_" + pattern_map.substr(0, colon);
    const nlohmann::json &h = j.at(2).at("pattern_" + pattern_map.substr(colon + 1));

    for (auto &[code, count] : h.at("call_type").items()) {
        if (SEND_CALLS.count(std::stoi(code))) {
            p.sends += count.get<int>();
        }
    }
    double total = 0.0;
    for (auto &[partner, count] : h.at("neighbors").items()) {
        total += count.get<int>();
    }
    for (auto &[partner, count] : h.at("neighbors").items()) {
        p.neighbors[std::stoi(partner)] = count.get<int>() / total;
    }
    // nlohmann orders keys as strings, so sort the bins by their numeric edge.
    std::map<double, int> bins;
    for (auto &[edge, count] : h.at("buffer_size").items()) {
        bins[std::stod(edge)] = count.get<int>();
    }
    for (const auto &[edge, count] : bins) {
        p.size_edges.push_back(edge);
        p.size_counts.push_back(count);
    }
    return p;
}
