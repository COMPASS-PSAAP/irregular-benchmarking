#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "benchmark.hpp"
#include "cli.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    {
        // Parse command-line arguments into the global benchmark options
        parseArgs(argc, argv);

        Kokkos::ScopeGuard scope_guard(argc, argv);
        // Run the benchmark
        run_benchmark();
    }

    MPI_Finalize();

    return 0;
}
