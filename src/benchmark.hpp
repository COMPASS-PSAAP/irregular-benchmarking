#ifndef IRREGULAR_BENCHMARKING_BENCHMARK_HPP
#define IRREGULAR_BENCHMARKING_BENCHMARK_HPP

// Runs the halo-exchange benchmark over every pattern in `patterns` (see
// pattern.hpp), sampling nsamples times per pattern per the global options
// (see options.hpp), and reports min/max/average timing statistics across
// ranks.
void run_benchmark();

#endif // IRREGULAR_BENCHMARKING_BENCHMARK_HPP
