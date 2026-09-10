#ifndef IRREGULAR_BENCHMARKING_CLI_HPP
#define IRREGULAR_BENCHMARKING_CLI_HPP

// Parses command-line arguments into the global benchmark options (see
// options.hpp) and, if a pattern file was given, loads it into `patterns`
// (see pattern.hpp). Exits the process on invalid input.
void parseArgs(int argc, char **argv);

#endif // IRREGULAR_BENCHMARKING_CLI_HPP
