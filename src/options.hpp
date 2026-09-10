#ifndef IRREGULAR_BENCHMARKING_OPTIONS_HPP
#define IRREGULAR_BENCHMARKING_OPTIONS_HPP

#include <string>

enum distribution_t {
    GAUSSIAN,
    EMPIRICAL,
    STATIC_VALUE
};

enum halo_t {
    IMPORT,
    EXPORT
};

enum comm_t {
    MPIADVANCE,
    MPIS
};

// Benchmark options: populated by cli::parseArgs() (and, for
// data_sent_max/nneighbors_max, by pattern::from_json()) and read by
// run_benchmark().
extern int nsamples;
extern int niterations;

extern std::string filepath;
extern distribution_t distribution_type;
extern halo_t halo_type;
extern comm_t comm_type;
extern std::string crs;
extern bool barrier;
extern int seed;
extern bool unique_seed;
extern int data_sent_max;
extern int nneighbors_max;

// Noisy-neighbor options
extern int nosy_percent;
extern int nosy_time_ms;

#endif // IRREGULAR_BENCHMARKING_OPTIONS_HPP
