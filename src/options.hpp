#ifndef IRREGULAR_BENCHMARKING_OPTIONS_HPP
#define IRREGULAR_BENCHMARKING_OPTIONS_HPP

#include <string>
#include <vector>

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

enum warmup_t {
    WARMUP_NONE,
    WARMUP_NEAREST, // rank +/- 1
    WARMUP_ALL      // every other rank
};

// Benchmark options: populated by cli::parseArgs() (and, for data_sent_max,
// by pattern::from_json()) and read by run_benchmark().
extern int nsamples;
extern int nwindow;
extern int ntimed;
extern double calc_budget_s;

extern std::string filepath;
extern std::vector<std::string> selected_patterns; // empty: every pattern, in name order
extern warmup_t warmup_mode;
extern distribution_t distribution_type;
extern halo_t halo_type;
extern comm_t comm_type;
extern std::string crs;
extern bool barrier;
extern bool distinct_neighbors;
extern bool verify;
extern int seed;
extern bool unique_seed;
extern int data_sent_max;

// Noisy-neighbor options
extern int nosy_percent;
extern int nosy_time_ms;

#endif // IRREGULAR_BENCHMARKING_OPTIONS_HPP
