#ifndef IRREGULAR_BENCHMARKING_MATRIX_HPP
#define IRREGULAR_BENCHMARKING_MATRIX_HPP

#include <string>
#include <vector>

// The halo of a distributed sparse matrix-vector product y = A x, read from a
// PETSc binary matrix (.pm, as written by ABCG's test_data/convert_petsc.py).
// Rows and x are split into contiguous blocks exactly as ABCG's readParMatrix
// does, so the halo is the one ABCG's SpMV exchanges: each rank needs the x
// entries in its rows' off-process columns, one double each.
struct MatrixHalo {
    std::string name;  // file name without extension, used as the pattern name
    int global_rows = 0;
    int global_cols = 0;
    long global_nnz = 0;
    int local_cols = 0;  // x entries this rank owns
    // Owned x entries that other ranks need: (local index, requesting rank).
    std::vector<int> export_ids;
    std::vector<int> export_ranks;
    // x entries this rank needs from others: (owner's local index, owner rank).
    std::vector<int> import_ids;
    std::vector<int> import_ranks;
};

// Collective over MPI_COMM_WORLD. Exits with an error message if the file is
// not a PETSc binary matrix or has fewer rows or columns than ranks.
MatrixHalo read_matrix_halo(const std::string &path);

extern bool matrix_input;       // true when the input file is a .pm matrix
extern MatrixHalo matrix_halo;  // filled by cli::parseArgs() when matrix_input

#endif // IRREGULAR_BENCHMARKING_MATRIX_HPP
