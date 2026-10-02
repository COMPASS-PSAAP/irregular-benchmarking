#include "cli.hpp"

#include <climits>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include <Cabana_Core.hpp>
#include <tclap/CmdLine.h>

#include "options.hpp"
#include "pattern.hpp"

namespace {

// Prints an error message and exits the program with an error code.
void exitError(const std::string &error_message) {
    std::cerr << error_message << std::flush;
    std::exit(-1);
}

// Sets `value` based on a command-line argument and checks its validity.
void setAndCheckValue(int &value, TCLAP::ValueArg<int> &arg,
                       const char *errorMessage, int minValue = 0, int maxValue = INT_MAX) {
    int tempValue = arg.getValue();
    // If the value from the argument is not -1 (indicating it was set by
    // the user), update 'value' with the new value.
    if (tempValue != -1) {
        value = tempValue;
    }

    if (value < minValue || value > maxValue) {
        exitError(errorMessage);
    }
}

} // namespace

void parseArgs(int argc, char **argv) {
    try {
        TCLAP::CmdLine cmd("\nNOTE: TODO", ' ', "1.0");

        TCLAP::ValueArg<std::string> filepathArg("f", "filepath", "Path to the BENCHMARK_CONFIG file", true, "", "string");
        TCLAP::ValueArg<int> samplesArg("I", "samples", "Number of random samples to generate", false, 25, "int");
        TCLAP::ValueArg<int> windowArg("k", "window", "Number of startup-window gathers per sample, timed as one total; these absorb the backend's lazy setup", false, nwindow, "int");
        TCLAP::ValueArg<int> timedArg("n", "timed-iterations", "Number of gathers timed after the startup window", false, ntimed, "int");
        TCLAP::ValueArg<double> calcArg("", "calc", "Budget in seconds; per sample, solve window = startup + q*k and timed = q*n for q and startup, and print the n that makes startup + q*(k+n) fill the budget", false, calc_budget_s, "seconds");
        TCLAP::ValueArg<int> seedArg("S", "seed", "Positive integer to be used as seed for random number generation", false, -1, "int");
        TCLAP::SwitchArg useedArg("q", "unique-seed", "unique seed per rank", false);
        TCLAP::SwitchArg persistentArg("p", "persistent", "will use the persistent mpi-advance", false);
        TCLAP::SwitchArg barrierArg("b", "barrier", "time each gather barrier to barrier, so the wait for the slowest rank is included; without it ranks can drift out of sync", false);
        TCLAP::SwitchArg verifyArg("V", "verify", "after each MPI-Advance sample, gather once through plain MPI and once through MPI-Advance on the same halo and report ranks whose ghost data differ", false);
        TCLAP::ValueArg<std::string> patternsArg("P", "patterns", "Comma-separated pattern names to run, in this order (default: every pattern in the config, in name order)", false, "", "names");
        TCLAP::ValueArg<std::string> warmupArg("W", "warmup", "Untimed point-to-point warmup of 64 B and 128 KiB messages before any pattern: none (default), nearest (rank +/- 1) or all (every other rank)", false, "none", "mode");
        TCLAP::SwitchArg distinctArg("U", "distinct-neighbors", "draw each rank's neighbors without replacement, so a rank never draws the same partner twice", false);

        TCLAP::SwitchArg reportParamsArg("r", "report-params", "Enables parameter reporting for use with analysis scripts", false);
        TCLAP::ValueArg<std::string> distributionArg("d", "distribution", "Choose from: gaussian (default), empirical or static", false, "gaussian", "string");
        TCLAP::ValueArg<std::string> splitArg("s", "split-type", "Choose from: SOCKET|S|s (default), NUMA|U|u or NODE|N|n", false, "SOCKET", "string");
        TCLAP::ValueArg<std::string> commArg("c", "comm", "Choose from: MPIA|A|a (default) or MPI|M|m", false, "MPIA", "string");
        TCLAP::ValueArg<std::string> INorOUTArg("x", "type", "Choose from: EXPORT|E|e (default) or IMPORT|I|i", false, "EXPORT", "string");
        TCLAP::ValueArg<std::string> ALLTOALLV("a", "alltoallv", "Choose from: STANDARD|S|s (default) or LOCALITY|L|l", false, "STANDARD", "string");
        TCLAP::ValueArg<std::string> CRS(
            "C",
            "crs",
            "Choose CRS method: default| nonblocking | personalized | personalized_loc | nonblocking_loc |rma",
            false,
            "default",
            "string"
        );

        // Nosy Neighbor args
        TCLAP::ValueArg<int> nosyPercentArg("N", "nosy-percent", "Percentage probability (0-100) to act as a noisy neighbor", false, 0, "int");
        TCLAP::ValueArg<int> nosyTimeArg("T", "nosy-time", "Time to wait if chosen as nosy neighbor (in milliseconds)", false, 0, "int");

        cmd.add(filepathArg);
        cmd.add(samplesArg);
        cmd.add(windowArg);
        cmd.add(timedArg);
        cmd.add(calcArg);
        cmd.add(seedArg);
        cmd.add(useedArg);
        cmd.add(distributionArg);
        cmd.add(reportParamsArg);
        cmd.add(commArg);
        cmd.add(INorOUTArg);
        cmd.add(ALLTOALLV);
        cmd.add(splitArg);
        cmd.add(CRS);
        cmd.add(persistentArg);
        cmd.add(barrierArg);
        cmd.add(distinctArg);
        cmd.add(patternsArg);
        cmd.add(warmupArg);
        cmd.add(verifyArg);
        cmd.add(nosyPercentArg);
        cmd.add(nosyTimeArg);

        cmd.parse(argc, argv);

        filepath = filepathArg.getValue();
        bool persistent = persistentArg.getValue();
        barrier = barrierArg.getValue();
        distinct_neighbors = distinctArg.getValue();
        verify = verifyArg.getValue();

        nosy_percent = nosyPercentArg.getValue();
        nosy_time_ms = nosyTimeArg.getValue();

        if (filepath.empty()) {
            exitError("ERROR: Filepath is empty!\n");
        }

        try {
            std::filesystem::path p(filepath);

            if (std::filesystem::exists(p)) {
                std::ifstream file(filepath);
                if (!file.is_open()) {
                    exitError("Error: Could not open file. ");
                }

                nlohmann::json j;
                file >> j;

                for (auto &[pattern_name, pattern_json] : j.items()) {
                    patterns[pattern_name] = pattern_json.get<Pattern>();
                }
            } else {
                exitError("The file does not exist.");
            }
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        std::string pattern_list = patternsArg.getValue();
        if (!pattern_list.empty()) {
            std::stringstream names(pattern_list);
            std::string name;
            std::set<std::string> seen;
            while (std::getline(names, name, ',')) {
                if (patterns.count(name) == 0) {
                    std::string available;
                    for (const auto &[known, pattern] : patterns) {
                        available += " " + known;
                    }
                    exitError("ERROR: Unknown pattern '" + name + "'; the config has:" + available + "\n");
                }
                if (!seen.insert(name).second) {
                    exitError("ERROR: Pattern '" + name + "' is listed twice\n");
                }
                selected_patterns.push_back(name);
            }
        }

        std::string warmup = warmupArg.getValue();
        if (warmup == "none") {
            warmup_mode = WARMUP_NONE;
        } else if (warmup == "nearest") {
            warmup_mode = WARMUP_NEAREST;
        } else if (warmup == "all") {
            warmup_mode = WARMUP_ALL;
        } else {
            exitError("ERROR: Invalid warmup choice [none,nearest,all]\n");
        }

        unique_seed = useedArg.getValue();

        std::string distribution = distributionArg.getValue();
        // For nsamples, no specific range, only non-negative check
        setAndCheckValue(nsamples, samplesArg, "ERROR: Invalid number of samples\n", 0);

        setAndCheckValue(nwindow, windowArg, "ERROR: Invalid startup window length\n", 0);
        // At least one timed gather, since --calc divides by n.
        setAndCheckValue(ntimed, timedArg, "ERROR: Invalid number of timed iterations\n", 1);
        calc_budget_s = calcArg.getValue();

        if (distribution == "gaussian" || distribution == "g") {
            distribution_type = GAUSSIAN;
        } else if (distribution == "empirical" || distribution == "e") {
            distribution_type = EMPIRICAL;
        } else if (distribution == "static" || distribution == "s") {
            distribution_type = STATIC_VALUE;
        } else {
            exitError("ERROR: Invalid distribution choice [empirical,gaussian]\n");
        }

        std::string comm = commArg.getValue();
        if (comm == "A" || comm == "a" || comm == "MPIA") {
            comm_type = MPIADVANCE;
            comm = "MPIADVANCE";
        } else if (comm == "M" || comm == "m" || comm == "MPI") {
            comm_type = MPIS;
            comm = "MPI";
        } else {
            exitError("ERROR: Invalid Backend choice [MPIA,MPIA]\n");
        }

        std::string split = splitArg.getValue();
        if (split == "U" || split == "u" || split == "NUMA") {
            MPIL_Set_split(NUMA);
            split = "NUMA";
        } else if (split == "N" || split == "n" || split == "NODE") {
            split = "NODE";
            MPIL_Set_split(NODE);
        } else if (split == "S" || split == "s" || split == "SOCKET") {
            split = "SOCKET";
            MPIL_Set_split(SOCKET);
        } else {
            exitError("ERROR: Invalid split choice [SOCKET,NODE,NUMA]\n");
        }

        std::string discovery = CRS.getValue();
        if (discovery == "rma") {
            MPIL_Set_alltoall_crs(ALLTOALL_CRS_RMA);
        } else if (discovery == "default") {
            // no-op: use MPI_Advance's default CRS discovery method
        } else if (discovery == "nonblocking") {
            MPIL_Set_alltoall_crs(ALLTOALL_CRS_NONBLOCKING);
        } else if (discovery == "personalized") {
            MPIL_Set_alltoall_crs(ALLTOALL_CRS_PERSONALIZED);
        } else if (discovery == "personalized_loc") {
            MPIL_Set_alltoall_crs(ALLTOALL_CRS_PERSONALIZED_LOC);
        } else if (discovery == "nonblocking_loc") {
            MPIL_Set_alltoall_crs(ALLTOALL_CRS_NONBLOCKING_LOC);
        } else {
            exitError("ERROR: Invalid pattern choice [default,nonblocking, personalized, personalized_loc, nonblocking_loc]\n");
        }

        crs = discovery;

        std::string type = INorOUTArg.getValue();
        if (type == "E" || type == "e" || type == "EXPORT") {
            halo_type = EXPORT;
            type = "EXPORT";
        } else if (type == "I" || type == "i" || type == "IMPORT") {
            halo_type = IMPORT;
            type = "IMPORT";
        } else {
            exitError("ERROR: Invalid Patern choice [EXPORT,IMPORT]\n");
        }

        std::string alltoallv = ALLTOALLV.getValue();
        if (alltoallv == "S" || alltoallv == "s" || alltoallv == "STANDARD") {
            mpil_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_STANDARD;
            alltoallv = "NEIGHBOR_ALLTOALLV_INIT_STANDARD";
        } else if (alltoallv == "L" || alltoallv == "l" || alltoallv == "LOCALITY") {
            mpil_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_LOCALITY;
            alltoallv = "NEIGHBOR_ALLTOALLV_INIT_LOCALITY";
        } else {
            exitError("ERROR: Invalid alltoallv choice [LOCALITY,STANDARD]\n");
        }

        int seedholder = seedArg.getValue();
        seed = (seedholder != -1) ? seedholder : static_cast<int>(time(NULL));

        int comm_rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

        // Resolved in place so sample_from_map()'s RNG picks up the same seed.
        if (unique_seed) {
            seed += comm_rank;
        }
        srand(seed);

        if (comm_rank == 0) {
            if (reportParamsArg.getValue()) {
                char mpi_library[MPI_MAX_LIBRARY_VERSION_STRING];
                int mpi_library_len = 0;
                MPI_Get_library_version(mpi_library, &mpi_library_len);
                std::string mpi_library_line(mpi_library, mpi_library_len);
                for (char &c : mpi_library_line) {
                    if (c == '\n') {
                        c = ' ';
                    }
                }

                printf("------------------------------------------------------------\n");
                printf("-MPI: %s\n", comm.c_str());
                printf("-halotype: %s\n", type.c_str());
                printf("-File: %s\n", filepath.c_str());
                printf("-samples: %i\n", nsamples);
                printf("-patterns: %s\n", pattern_list.empty() ? "all" : pattern_list.c_str());
                printf("-warmup: %s\n", warmup.c_str());
                printf("-window (k): %i\n", nwindow);
                printf("-timed iterations (n): %i\n", ntimed);
                if (calc_budget_s > 0) {
                    printf("-calc budget (s): %g\n", calc_budget_s);
                } else {
                    printf("-calc: off\n");
                }
                printf("-CRS: %s\n", crs.c_str());
                printf("-alltoallv: %s\n", alltoallv.c_str());
                printf("-split: %s\n", split.c_str());
                printf("-persistent: %s\n", persistent ? "true" : "false");
                printf("-barrier: %s\n", barrier ? "true" : "false");
                printf("-distinct neighbors: %s\n", distinct_neighbors ? "true" : "false");
                printf("-nosy percent: %i\n", nosy_percent);
                printf("-nosy time (ms): %i\n", nosy_time_ms);
                // Rank 0 added rank 0 above, so this is the base seed either way.
                printf("-seed: %i%s\n", seed, unique_seed ? " (per rank: seed + rank)" : "");
                printf("-MPI library: %s\n", mpi_library_line.c_str());
                printf("-compiler: %s\n", __VERSION__);
                printf("------------------------------------------------------------\n");
            } else {
                printf("------------------------------------------------------------\n");
            }
        }

    } catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
        exit(-1);
    }
}
