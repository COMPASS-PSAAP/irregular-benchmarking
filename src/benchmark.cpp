#include "benchmark.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <thread>
#include <vector>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include "options.hpp"
#include "pattern.hpp"
#include "sampling.hpp"

// Copied and adapted from
// https://github.com/ECP-copa/Cabana/wiki/2-Programming-Guide
void run_benchmark() {
    auto TIME_START = std::chrono::high_resolution_clock::now();
    auto TIME_END = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = TIME_END - TIME_START;
    int comm_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    int comm_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    using cabana_datatype = double;

    // for cases where buffer sizes are not divisible by sizeof(cabana_datatype)
    auto bytes_to_elems = [](int bytes) {
        return (bytes + sizeof(cabana_datatype) - 1) / sizeof(cabana_datatype);
    };

    for (auto &[name, pattern] : patterns) {
        for (int sample_iter = 0; sample_iter < nsamples; sample_iter++) {
            std::vector<int> neighbors_data;
            std::vector<int> neighbors;
            std::set<int> seen_neighbors;
            int total_export = 0;
            int nneighborsV = -1;

            nneighborsV = sample_from_map(pattern.comm_partners);

            neighbors_data.reserve(nneighborsV);
            neighbors.reserve(nneighborsV);
            double numberOfmessages = nneighborsV;

            for (int i = 0; i < numberOfmessages; ++i) {
                int data_sentV = sample_from_map(pattern.buffer_size);
                int n_export = bytes_to_elems(data_sentV);
                total_export += n_export;

                int distanceToN = sample_from_map(pattern.dist_to_neighbors[nneighborsV]);
                int node = (distanceToN + comm_rank + comm_size) % comm_size;

                seen_neighbors.insert(node);
                neighbors.push_back(node);
                neighbors_data.push_back(n_export);
            }

            double haloTime = -1.0;
            double resizeTime = -1.0;
            double gatherTime = -1.0;
            double apply = -1.0;
            double nosy_count = 0.0; // Tracks nosy iterations for this sample

            using DataTypes = Cabana::MemberTypes<cabana_datatype>;
            const int VectorLength = 2;
            using MemorySpace = Kokkos::HostSpace;

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

            assert(inum == total_export);

            auto TIME_START_HALO = std::chrono::high_resolution_clock::now();

            MPI_Barrier(MPI_COMM_WORLD);
            if (comm_type == MPIADVANCE) {
                if (halo_type == EXPORT) {
                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::Halo<MemorySpace, Cabana::Export, Cabana::LocalityAware> halo(MPI_COMM_WORLD, num_tuple, export_ids, export_ranks);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    haloTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa("my_aosoa", halo.numLocal() + halo.numGhost());
                    auto slice_ranks = Cabana::slice<0>(aosoa);
                    for (int i = 0; i < num_tuple; ++i) {
                        slice_ranks(i) = i;
                    }
                    fflush(stdout);

                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    resizeTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    auto gather = Cabana::createGather(halo, aosoa, 1.0);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    gatherTime = duration.count();

                    apply = 0;
                    for (int i = 0; i < niterations; i++) {
                        if (nosy_percent > 0 && (std::rand() % 100) < nosy_percent) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(nosy_time_ms));
                            nosy_count += 1.0;
                        }
                        TIME_START = std::chrono::high_resolution_clock::now();
                        gather.apply();
                        TIME_END = std::chrono::high_resolution_clock::now();
                        duration = TIME_END - TIME_START;
                        apply = apply + duration.count();
                        if (barrier) {
                            MPI_Barrier(MPI_COMM_WORLD);
                        }
                    }

                } else if (halo_type == IMPORT) {
                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::Halo<MemorySpace, Cabana::Import, Cabana::LocalityAware> halo(MPI_COMM_WORLD, num_tuple, export_ids, export_ranks);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    haloTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa("my_aosoa", halo.numLocal() + halo.numGhost());
                    auto slice_ranks = Cabana::slice<0>(aosoa);
                    for (int i = 0; i < num_tuple; ++i) {
                        slice_ranks(i) = i;
                    }
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    resizeTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    auto gather = Cabana::createGather(halo, aosoa, 1.0);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    gatherTime = duration.count();

                    apply = 0;
                    for (int i = 0; i < niterations; i++) {
                        if (nosy_percent > 0 && (std::rand() % 100) < nosy_percent) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(nosy_time_ms));
                            nosy_count += 1.0;
                        }
                        TIME_START = std::chrono::high_resolution_clock::now();
                        gather.apply();
                        TIME_END = std::chrono::high_resolution_clock::now();
                        duration = TIME_END - TIME_START;
                        apply = apply + duration.count();
                        if (barrier) {
                            MPI_Barrier(MPI_COMM_WORLD);
                        }
                    }
                } else {
                    // error
                }
            } else if (comm_type == MPIS) {
                if (halo_type == EXPORT) {
                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::Halo<MemorySpace, Cabana::Export, Cabana::Mpi> halo(MPI_COMM_WORLD, num_tuple, export_ids, export_ranks);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    haloTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa("my_aosoa", halo.numLocal() + halo.numGhost());
                    auto slice_ranks = Cabana::slice<0>(aosoa);
                    for (int i = 0; i < num_tuple; ++i) {
                        slice_ranks(i) = i;
                    }
                    fflush(stdout);

                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    resizeTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    auto gather = Cabana::createGather(halo, aosoa, 1.0);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    gatherTime = duration.count();

                    apply = 0;
                    for (int i = 0; i < niterations; i++) {
                        if (nosy_percent > 0 && (std::rand() % 100) < nosy_percent) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(nosy_time_ms));
                            nosy_count += 1.0;
                        }
                        TIME_START = std::chrono::high_resolution_clock::now();
                        gather.apply();
                        TIME_END = std::chrono::high_resolution_clock::now();
                        duration = TIME_END - TIME_START;
                        apply = apply + duration.count();
                        if (barrier) {
                            MPI_Barrier(MPI_COMM_WORLD);
                        }
                    }
                } else if (halo_type == IMPORT) {
                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::Halo<MemorySpace, Cabana::Import, Cabana::Mpi> halo(MPI_COMM_WORLD, num_tuple, export_ids, export_ranks);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    haloTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa("my_aosoa", halo.numLocal() + halo.numGhost());
                    auto slice_ranks = Cabana::slice<0>(aosoa);
                    for (int i = 0; i < num_tuple; ++i) {
                        slice_ranks(i) = i;
                    }
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    resizeTime = duration.count();

                    TIME_START = std::chrono::high_resolution_clock::now();
                    auto gather = Cabana::createGather(halo, aosoa, 1.0);
                    TIME_END = std::chrono::high_resolution_clock::now();
                    duration = TIME_END - TIME_START;
                    gatherTime = duration.count();

                    apply = 0;
                    for (int i = 0; i < niterations; i++) {
                        if (nosy_percent > 0 && (std::rand() % 100) < nosy_percent) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(nosy_time_ms));
                            nosy_count += 1.0;
                        }
                        TIME_START = std::chrono::high_resolution_clock::now();
                        gather.apply();
                        TIME_END = std::chrono::high_resolution_clock::now();
                        duration = TIME_END - TIME_START;
                        apply = apply + duration.count();
                        if (barrier) {
                            MPI_Barrier(MPI_COMM_WORLD);
                        }
                    }
                } else {
                    // error
                }
            }
            auto TIME_END_Halo = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> halo_gather_time = TIME_END_Halo - TIME_START_HALO;
            double halo_gather = halo_gather_time.count();

            constexpr int DATA_SIZE = 9;
            double local_vals[DATA_SIZE] = {
                haloTime,
                resizeTime,
                gatherTime,
                apply,
                halo_gather,
                (double)nneighborsV,
                (double)(inum * sizeof(cabana_datatype)),
                numberOfmessages,
                nosy_count
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
                    "NoseNeighbors"
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
        }
    }
}
