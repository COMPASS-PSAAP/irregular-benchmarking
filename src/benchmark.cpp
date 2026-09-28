#include "benchmark.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <thread>
#include <vector>

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

struct SampleTimings {
    double halo = -1.0;
    double resize = -1.0;
    double gather = -1.0;
    double apply = -1.0;
    double first_apply = -1.0;
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

    timings.apply = 0.0;
    for (int i = 0; i < niterations; i++) {
        if (nosy_percent > 0 && (std::rand() % 100) < nosy_percent) {
            std::this_thread::sleep_for(std::chrono::milliseconds(nosy_time_ms));
            timings.nosy_count += 1.0;
        }
        start = clock_type::now();
        gather.apply();
        double elapsed = msSince(start);
        timings.apply += elapsed;
        // The first apply can include one-time lazy setup, so report it on its own.
        if (i == 0) {
            timings.first_apply = elapsed;
        }
        if (barrier) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

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

    for (auto &[name, pattern] : patterns) {
        for (int sample_iter = 0; sample_iter < nsamples; sample_iter++) {
            std::vector<int> neighbors_data;
            std::vector<int> neighbors;
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

            constexpr int DATA_SIZE = 10;
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
                timings.first_apply
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
                    "firstApply"
                };

                printf("%-20s %-12s %-12s %-12s\n", "Metric", "Min", "Max", "Average");
                printf("------------------------------------------------------------\n");

                for (int i = 0; i < DATA_SIZE; ++i) {
                    double avg = sum_vals[i] / comm_size;
                    printf("(%s)  %-20s %-.6f     %-.6f     %-.6f\n", name.c_str(), labels[i], min_vals[i], max_vals[i], avg);
                }
                printf("------------------------------------------------------------\n");
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
