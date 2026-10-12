#include "fine.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include "sampling.hpp"

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

// Vernier's call codes (source/core/codex.hpp at 09d650b): sends MPI_Irsend 18,
// MPI_Issend 19, MPI_Isend 21, MPI_Send 30, MPI_Send_init 31, MPI_Ssend 32;
// receives MPI_Irecv 16, MPI_Recv 24, MPI_Recv_init 25. MPI_Sendrecv (33) is
// left out: one record stands for both a send and a receive.
const std::set<int> SEND_CALLS = {18, 19, 21, 30, 31, 32};
const std::set<int> RECV_CALLS = {16, 24, 25};

const std::set<int> &calls_for(halo_t type) {
    return type == EXPORT ? SEND_CALLS : RECV_CALLS;
}

} // namespace

FinePattern read_fine_pattern(const std::string &dir) {
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string path = dir + "/vernier-" + std::to_string(rank) + "-0.hist-hierarchical.json";
    std::ifstream file(path);
    if (!file.is_open()) {
        fail("could not open " + path);
    }
    // [{"pattern_map": "name:id"}, {"comm_map": "size|..."},
    //  {"pattern_<id>": {"<call type>": {"<partner>": {"<bin lower edge>": count}}}}]
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
    for (auto &[call, partners] : j.at(2).at("pattern_" + pattern_map.substr(colon + 1)).items()) {
        for (auto &[partner, bins] : partners.items()) {
            // nlohmann orders keys as strings, so sort the bins by their numeric edge.
            std::map<double, int> sorted;
            for (auto &[edge, count] : bins.items()) {
                sorted[std::stod(edge)] = count.get<int>();
            }
            SizeBins &sizes = p.calls[std::stoi(call)][std::stoi(partner)];
            for (const auto &[edge, count] : sorted) {
                sizes.edges.push_back(edge);
                sizes.counts.push_back(count);
            }
        }
    }
    return p;
}

std::vector<std::pair<int, int>> fine_messages(const FinePattern &pattern, halo_t type) {
    std::vector<std::pair<int, int>> messages;
    for (const auto &[call, partners] : pattern.calls) {
        if (calls_for(type).count(call) == 0) {
            continue;
        }
        for (const auto &[partner, sizes] : partners) {
            int recorded = 0;
            for (int count : sizes.counts) {
                recorded += count;
            }
            for (int m = 0; m < recorded; ++m) {
                messages.emplace_back(partner, sample_interpolated(sizes.edges, sizes.counts));
            }
        }
    }
    return messages;
}

int fine_message_count(const FinePattern &pattern, halo_t type) {
    int total = 0;
    for (const auto &[call, partners] : pattern.calls) {
        if (calls_for(type).count(call) == 0) {
            continue;
        }
        for (const auto &[partner, sizes] : partners) {
            for (int count : sizes.counts) {
                total += count;
            }
        }
    }
    return total;
}
