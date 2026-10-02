#include "benchmark.hpp"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include "options.hpp"
#include "pattern.hpp"
#include "sampling.hpp"

namespace {

using cabana_datatype = double;
using DataTypes = Cabana::MemberTypes<cabana_datatype>;
using MemorySpace = Kokkos::HostSpace;
using clock_type = std::chrono::high_resolution_clock;

constexpr int VectorLength = 2;

double msSince(const clock_type::time_point &start) {
    std::chrono::duration<double, std::milli> duration = clock_type::now() - start;
    return duration.count();
}

// for cases where buffer sizes are not divisible by sizeof(cabana_datatype)
int bytes_to_elems(int bytes) {
    constexpr int elem_size = static_cast<int>(sizeof(cabana_datatype));
    return (bytes + elem_size - 1) / elem_size;
}

// Index of the NUMA domain that holds every CPU this rank may run on, or -1 when
// its affinity spans more than one domain (an unbound rank has no single domain).
int boundNumaDomain() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        return -1;
    }
    int found = -1;
    for (const auto &entry : std::filesystem::directory_iterator("/sys/devices/system/node")) {
        const std::string dir = entry.path().filename().string();
        if (dir.size() <= 4 || dir.compare(0, 4, "node") != 0 || !std::isdigit(dir[4])) {
            continue;
        }
        std::ifstream file(entry.path() / "cpulist");
        std::string list;
        std::getline(file, list);
        // cpulist looks like "0-55,112-167"; memory-only domains leave it empty.
        bool overlaps = false;
        std::stringstream ranges(list);
        std::string range;
        while (std::getline(ranges, range, ',')) {
            if (range.empty()) {
                continue;
            }
            const std::size_t dash = range.find('-');
            const int lo = std::stoi(range.substr(0, dash));
            const int hi = (dash == std::string::npos) ? lo : std::stoi(range.substr(dash + 1));
            for (int cpu = lo; cpu <= hi && cpu < CPU_SETSIZE; ++cpu) {
                overlaps = overlaps || CPU_ISSET(cpu, &mask);
            }
        }
        if (overlaps) {
            if (found != -1) {
                return -1;
            }
            found = std::stoi(dir.substr(4));
        }
    }
    return found;
}

// The node and NUMA domain of every rank in MPI_COMM_WORLD.
struct RankPlacement {
    std::vector<int> node;   // lowest world rank on that rank's node
    std::vector<int> numa;   // NUMA domain index on its node, -1 if unbound
    int unbound_ranks = 0;   // ranks with no single NUMA domain
};

RankPlacement gatherPlacement(int comm_rank, int comm_size) {
    // Same grouping locality_aware uses for its node-local communicator.
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm_rank, MPI_INFO_NULL, &node_comm);
    int node_id = comm_rank;
    MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm);
    MPI_Comm_free(&node_comm);

    int mine[2] = {node_id, boundNumaDomain()};
    std::vector<int> all(2 * comm_size);
    MPI_Allgather(mine, 2, MPI_INT, all.data(), 2, MPI_INT, MPI_COMM_WORLD);

    RankPlacement placement;
    placement.node.resize(comm_size);
    placement.numa.resize(comm_size);
    for (int r = 0; r < comm_size; ++r) {
        placement.node[r] = all[2 * r];
        placement.numa[r] = all[2 * r + 1];
        if (placement.numa[r] == -1) {
            ++placement.unbound_ranks;
        }
    }
    return placement;
}

// Untimed exchange of 64 B and 128 KiB messages with rank +/- d for each distance
// d, so connections to those ranks are open before any pattern is measured. It
// runs on a duplicate of MPI_COMM_WORLD that is freed afterwards, so nothing
// cached on the communicator carries over into the benchmark.
void runWarmup(warmup_t mode, int comm_rank, int comm_size) {
    std::vector<int> distances;
    if (mode == WARMUP_NEAREST) {
        distances.push_back(1);
        if (comm_size - 1 != 1) {
            distances.push_back(comm_size - 1); // rank - 1
        }
    } else if (mode == WARMUP_ALL) {
        for (int d = 1; d < comm_size; ++d) {
            distances.push_back(d);
        }
    }
    if (distances.empty()) {
        return;
    }

    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    constexpr int sizes[2] = {1 << 6, 1 << 17};
    std::vector<char> send_buf(sizes[1], 1), recv_buf(sizes[1]);
    for (int d : distances) {
        const int to = (comm_rank + d) % comm_size;
        const int from = (comm_rank - d + comm_size) % comm_size;
        for (int bytes : sizes) {
            MPI_Request requests[2];
            MPI_Irecv(recv_buf.data(), bytes, MPI_BYTE, from, 0, comm, &requests[0]);
            MPI_Isend(send_buf.data(), bytes, MPI_BYTE, to, 0, comm, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
    }
    MPI_Comm_free(&comm);
}

struct SampleTimings {
    double halo = -1.0;
    double resize = -1.0;
    double gather = -1.0;
    double apply = -1.0;
    double window = -1.0; // first k applies: startup + q*k
    double timed = -1.0;  // next n applies: q*n
    double nosy_count = 0.0;
};

// Copied and adapted from
// https://github.com/ECP-copa/Cabana/wiki/2-Programming-Guide
//
// BuildType selects Cabana::Export/Import, CommSpace the MPI-Advance or plain
// MPI backend.
template <class BuildType, class CommSpace>
SampleTimings runSample(int num_tuple,
                        const Kokkos::View<int *, MemorySpace> &export_ids,
                        const Kokkos::View<int *, MemorySpace> &export_ranks) {
    SampleTimings timings;

    auto start = clock_type::now();
    Cabana::Halo<MemorySpace, BuildType, CommSpace> halo(MPI_COMM_WORLD, num_tuple, export_ids, export_ranks);
    timings.halo = msSince(start);

    start = clock_type::now();
    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa("my_aosoa", halo.numLocal() + halo.numGhost());
    auto slice_ranks = Cabana::slice<0>(aosoa);
    for (int i = 0; i < num_tuple; ++i) {
        slice_ranks(i) = i;
    }
    timings.resize = msSince(start);

    start = clock_type::now();
    auto gather = Cabana::createGather(halo, aosoa, 1.0);
    timings.gather = msSince(start);

    timings.window = 0.0;
    timings.timed = 0.0;
    // Halo and gather setup leave ranks out of step; start the first gather together.
    if (barrier) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    for (int i = 0; i < nwindow + ntimed; i++) {
        if (nosy_percent > 0 && (std::rand() % 100) < nosy_percent) {
            std::this_thread::sleep_for(std::chrono::milliseconds(nosy_time_ms));
            timings.nosy_count += 1.0;
        }
        start = clock_type::now();
        gather.apply();
        if (barrier) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double elapsed = msSince(start);
        // The backend pays lazy setup in the first few applies, so they are totalled apart.
        if (i < nwindow) {
            timings.window += elapsed;
        } else {
            timings.timed += elapsed;
        }
    }
    timings.apply = timings.window + timings.timed;

    return timings;
}

// Ghost values from one gather, split into the halo's own per-neighbor blocks.
struct GhostBlocks {
    std::vector<int> ranks;                  // source rank of each block, in plan order
    std::vector<std::vector<double>> values; // ghosts received from that rank
};

// One gather on a fresh halo. Local values encode (rank, index) so every ghost
// identifies which rank sent it.
template <class BuildType, class CommSpace>
GhostBlocks gatherGhosts(int num_tuple,
                         const Kokkos::View<int *, MemorySpace> &export_ids,
                         const Kokkos::View<int *, MemorySpace> &export_ranks,
                         int comm_rank) {
    Cabana::Halo<MemorySpace, BuildType, CommSpace> halo(MPI_COMM_WORLD, num_tuple, export_ids, export_ranks);
    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa("verify_aosoa", halo.numLocal() + halo.numGhost());
    auto values = Cabana::slice<0>(aosoa);
    for (std::size_t i = 0; i < halo.numLocal(); ++i) {
        values(i) = comm_rank * 1.0e7 + i;
    }
    for (std::size_t i = halo.numLocal(); i < halo.numLocal() + halo.numGhost(); ++i) {
        values(i) = -1.0;
    }
    auto gather = Cabana::createGather(halo, aosoa, 1.0);
    gather.apply();

    GhostBlocks blocks;
    std::size_t g = halo.numLocal();
    for (int n = 0; n < halo.numNeighbor(); ++n) {
        if (halo.numImport(n) == 0) {
            continue;
        }
        blocks.ranks.push_back(halo.neighborRank(n));
        blocks.values.emplace_back();
        for (std::size_t k = 0; k < halo.numImport(n); ++k) {
            blocks.values.back().push_back(values(g++));
        }
    }
    return blocks;
}

// Ghosts in a block that did not come from the rank the plan assigns to it.
int misplacedGhosts(const GhostBlocks &blocks) {
    int misplaced = 0;
    for (std::size_t b = 0; b < blocks.ranks.size(); ++b) {
        for (double v : blocks.values[b]) {
            if (static_cast<int>(v / 1.0e7) != blocks.ranks[b]) {
                ++misplaced;
            }
        }
    }
    return misplaced;
}

// Check MPI-Advance's gather against plain MPI's for the same export lists:
// every ghost must sit in the block of the rank that sent it, and each sender's
// block must hold the same values in the same order as plain MPI's. The order
// of the blocks themselves is plan-defined and reported for information only.
template <class BuildType>
void verifySample(const std::string &name, int num_tuple,
                  const Kokkos::View<int *, MemorySpace> &export_ids,
                  const Kokkos::View<int *, MemorySpace> &export_ranks,
                  int comm_rank, int comm_size) {
    auto expected = gatherGhosts<BuildType, Cabana::Mpi>(num_tuple, export_ids, export_ranks, comm_rank);
    auto actual = gatherGhosts<BuildType, Cabana::LocalityAware>(num_tuple, export_ids, export_ranks, comm_rank);

    std::map<int, std::vector<double>> expected_by_rank, actual_by_rank;
    for (std::size_t b = 0; b < expected.ranks.size(); ++b) {
        expected_by_rank[expected.ranks[b]] = expected.values[b];
    }
    for (std::size_t b = 0; b < actual.ranks.size(); ++b) {
        actual_by_rank[actual.ranks[b]] = actual.values[b];
    }

    // {misplaced (MPI), misplaced (MPI-Advance), per-sender data differs, block order differs}
    int bad[4] = {misplacedGhosts(expected) > 0 ? 1 : 0,
                  misplacedGhosts(actual) > 0 ? 1 : 0,
                  expected_by_rank != actual_by_rank ? 1 : 0,
                  expected.ranks != actual.ranks ? 1 : 0};
    int total_bad[4];
    MPI_Reduce(bad, total_bad, 4, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (comm_rank == 0) {
        bool pass = total_bad[0] == 0 && total_bad[1] == 0 && total_bad[2] == 0;
        printf("(%s)  verify: %s (ranks with misplaced ghosts: MPI %d, MPI-Advance %d; "
               "ranks whose per-sender data differs: %d; ranks with different block order: %d; of %d)\n",
               name.c_str(), pass ? "PASS" : "FAIL", total_bad[0], total_bad[1], total_bad[2],
               total_bad[3], comm_size);
        fflush(stdout);
    }
}

} // namespace

void run_benchmark() {
    int comm_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    int comm_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    runWarmup(warmup_mode, comm_rank, comm_size);

    const RankPlacement placement = gatherPlacement(comm_rank, comm_size);
    const bool numa_known = placement.unbound_ranks == 0;
    if (comm_rank == 0 && !numa_known) {
        printf("NUMA split unavailable: %d of %d ranks are not bound within one NUMA domain\n",
               placement.unbound_ranks, comm_size);
        fflush(stdout);
    }

    std::vector<std::string> names = selected_patterns;
    if (names.empty()) {
        for (const auto &[name, pattern] : patterns) {
            names.push_back(name);
        }
    }

    for (const std::string &name : names) {
        Pattern &pattern = patterns.at(name);
        for (int sample_iter = 0; sample_iter < nsamples; sample_iter++) {
            std::vector<int> neighbors_data;
            std::vector<int> neighbors;
            std::vector<int> neighbors_bytes;
            std::set<int> seen_neighbors;
            int total_export = 0;
            int total_bytes = 0;

            int nneighborsV = sample_from_map(pattern.comm_partners);

            neighbors_data.reserve(nneighborsV);
            neighbors.reserve(nneighborsV);
            int numberOfmessages = nneighborsV;

            // How many distinct ranks the offset distribution can actually reach.
            // Computed with the same expression used in the loop below so the two
            // agree, and only when the flag is on so the default path is untouched.
            int reachable_targets = 0;
            if (distinct_neighbors) {
                std::set<int> reachable;
                for (const auto &entry : pattern.dist_to_neighbors[nneighborsV]) {
                    reachable.insert((entry.first + comm_rank + comm_size) % comm_size);
                }
                reachable_targets = static_cast<int>(reachable.size());
            }

            for (int i = 0; i < numberOfmessages; ++i) {
                int data_sentV = sample_from_map(pattern.buffer_size);
                int n_export = bytes_to_elems(data_sentV);
                total_export += n_export;
                total_bytes += data_sentV;

                // Drawing with replacement lets a rank pick the same partner twice,
                // so its realised partner count falls short of nneighborsV. With
                // --distinct-neighbors we redraw instead, but only while the
                // distribution still has an unused target, so this cannot spin.
                const bool avoid_duplicates =
                    distinct_neighbors
                    && static_cast<int>(seen_neighbors.size()) < reachable_targets;

                int node = -1;
                do {
                    int distanceToN = sample_from_map(pattern.dist_to_neighbors[nneighborsV]);
                    node = (distanceToN + comm_rank + comm_size) % comm_size;
                } while (avoid_duplicates && seen_neighbors.count(node) > 0);

                seen_neighbors.insert(node);
                neighbors.push_back(node);
                neighbors_data.push_back(n_export);
                neighbors_bytes.push_back(data_sentV);
            }

            // Where each message goes relative to this rank; a self-send counts as local.
            double on_node_msgs = 0, off_node_msgs = 0, on_node_bytes = 0, off_node_bytes = 0;
            double in_numa_bytes = 0, out_numa_bytes = 0;
            for (std::size_t i = 0; i < neighbors.size(); ++i) {
                const int partner = neighbors[i];
                const bool same_node = placement.node[partner] == placement.node[comm_rank];
                if (same_node) {
                    on_node_msgs += 1;
                    on_node_bytes += neighbors_bytes[i];
                } else {
                    off_node_msgs += 1;
                    off_node_bytes += neighbors_bytes[i];
                }
                if (same_node && placement.numa[partner] == placement.numa[comm_rank]) {
                    in_numa_bytes += neighbors_bytes[i];
                } else {
                    out_numa_bytes += neighbors_bytes[i];
                }
            }

            int num_tuple = bytes_to_elems(data_sent_max);

            Kokkos::View<int *, MemorySpace> export_ranks("export_ranks", total_export);
            Kokkos::View<int *, MemorySpace> export_ids("export_ids", total_export);

            for (int i = 0; i < total_export; ++i) {
                export_ids(i) = -1;
                export_ranks(i) = -1;
            }

            int inum = 0;
            auto it_data = neighbors_data.begin();
            auto it_neighbors = neighbors.begin();

            while (it_data != neighbors_data.end() && it_neighbors != neighbors.end()) {
                for (int i = 0; i < *it_data; ++i) {
                    export_ids(inum) = i;
                    export_ranks(inum++) = *it_neighbors;
                }
                ++it_data;
                ++it_neighbors;
            }

            auto TIME_START_HALO = clock_type::now();

            MPI_Barrier(MPI_COMM_WORLD);
            SampleTimings timings;
            if (comm_type == MPIADVANCE) {
                if (halo_type == EXPORT) {
                    timings = runSample<Cabana::Export, Cabana::LocalityAware>(num_tuple, export_ids, export_ranks);
                } else {
                    timings = runSample<Cabana::Import, Cabana::LocalityAware>(num_tuple, export_ids, export_ranks);
                }
            } else {
                if (halo_type == EXPORT) {
                    timings = runSample<Cabana::Export, Cabana::Mpi>(num_tuple, export_ids, export_ranks);
                } else {
                    timings = runSample<Cabana::Import, Cabana::Mpi>(num_tuple, export_ids, export_ranks);
                }
            }

            double halo_gather = msSince(TIME_START_HALO);

            constexpr int DATA_SIZE = 17;
            double local_vals[DATA_SIZE] = {
                timings.halo,
                timings.resize,
                timings.gather,
                timings.apply,
                halo_gather,
                (double)nneighborsV,
                (double)total_bytes,
                (double)numberOfmessages,
                timings.nosy_count,
                timings.window,
                timings.timed,
                on_node_msgs,
                off_node_msgs,
                on_node_bytes,
                off_node_bytes,
                in_numa_bytes,
                out_numa_bytes
            };

            double min_vals[DATA_SIZE];
            double max_vals[DATA_SIZE];
            double sum_vals[DATA_SIZE];

            // Perform reductions
            MPI_Reduce(local_vals, min_vals, DATA_SIZE, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(local_vals, max_vals, DATA_SIZE, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(local_vals, sum_vals, DATA_SIZE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (comm_rank == 0) {
                const char *labels[DATA_SIZE] = {
                    "haloTime",
                    "resizeTime",
                    "gatherTime",
                    "apply",
                    "halo_gather_time",
                    "nneighbors",
                    "data_sent",
                    "numberOfmessages",
                    "NoseNeighbors",
                    "startupWindow",
                    "timedTotal",
                    "onNodeNeighbors",
                    "offNodeNeighbors",
                    "onNodeBytes",
                    "offNodeBytes",
                    "inNumaBytes",
                    "outNumaBytes"
                };
                constexpr int FIRST_NUMA_ROW = 15;

                printf("%-20s %-12s %-12s %-12s\n", "Metric", "Min", "Max", "Average");
                printf("------------------------------------------------------------\n");

                for (int i = 0; i < DATA_SIZE; ++i) {
                    if (i >= FIRST_NUMA_ROW && !numa_known) {
                        continue;
                    }
                    double avg = sum_vals[i] / comm_size;
                    printf("(%s)  %-20s %-.6f     %-.6f     %-.6f\n", name.c_str(), labels[i], min_vals[i], max_vals[i], avg);
                }
                printf("------------------------------------------------------------\n");
                if (calc_budget_s > 0) {
                    // Max over ranks, the time the application pays. All times are in ms.
                    double y = max_vals[9];
                    double x = max_vals[10];
                    double q = x / ntimed;
                    double startup = y - nwindow * q;
                    double n_budget = std::floor((calc_budget_s * 1000.0 - startup) / q - nwindow);
                    printf("(%s)  calc: q %.6f ms  startup %.6f ms  n %.0f  (budget %g s, k %d)\n",
                           name.c_str(), q, startup, n_budget, calc_budget_s, nwindow);
                }
                fflush(stdout);
            }
            if (comm_type == MPIADVANCE) {
                if (verify) {
                    if (halo_type == EXPORT) {
                        verifySample<Cabana::Export>(name, num_tuple, export_ids, export_ranks, comm_rank, comm_size);
                    } else {
                        verifySample<Cabana::Import>(name, num_tuple, export_ids, export_ranks, comm_rank, comm_size);
                    }
                }
            }
        }
    }
}
