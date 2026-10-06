#include "matrix.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <mpi.h>

bool matrix_input = false;
MatrixHalo matrix_halo;

namespace {

// First int32 of a PETSc binary matrix file.
constexpr int32_t PETSC_MAT_CODE = 1211216;

[[noreturn]] void fail(const std::string &message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "ERROR (rank " << rank << "): " << message << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

int32_t byteswap(int32_t v) {
    return static_cast<int32_t>(__builtin_bswap32(static_cast<uint32_t>(v)));
}

// Reads count int32 values starting offset int32s into the file.
std::vector<int32_t> read_int32s(std::ifstream &file, long long offset, long long count, bool swap,
                                 const char *what) {
    std::vector<int32_t> values(count);
    file.seekg(offset * static_cast<long long>(sizeof(int32_t)));
    file.read(reinterpret_cast<char *>(values.data()), count * static_cast<long long>(sizeof(int32_t)));
    if (!file) {
        fail(std::string("could not read the matrix ") + what);
    }
    if (swap) {
        for (int32_t &v : values) {
            v = byteswap(v);
        }
    }
    return values;
}

// ABCG's block split of n items over num_procs ranks: the first n % num_procs
// ranks get one extra item.
void block(int n, int num_procs, int rank, int &count, int &first) {
    count = n / num_procs;
    const int extra = n % num_procs;
    first = count * rank;
    if (extra > rank) {
        ++count;
        first += rank;
    } else {
        first += extra;
    }
}

} // namespace

MatrixHalo read_matrix_halo(const std::string &path) {
    int rank = 0;
    int num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MatrixHalo halo;
    halo.name = std::filesystem::path(path).stem().string();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fail("cannot open " + path);
    }

    // Header: code, rows, cols, nnz. PETSc writes big-endian, so swap when the
    // code does not match as read.
    std::vector<int32_t> header = read_int32s(file, 0, 4, false, "header");
    bool swap = false;
    if (header[0] != PETSC_MAT_CODE) {
        swap = true;
        for (int32_t &v : header) {
            v = byteswap(v);
        }
        if (header[0] != PETSC_MAT_CODE) {
            fail(path + " is not a PETSc binary matrix");
        }
    }
    halo.global_rows = header[1];
    halo.global_cols = header[2];
    halo.global_nnz = header[3];
    // ABCG has a separate split for this case; it does not arise for the
    // matrices and rank counts this benchmark is run with.
    if (halo.global_rows < num_procs || halo.global_cols < num_procs) {
        fail(path + " has fewer rows or columns than there are ranks");
    }

    int local_rows = 0, first_row = 0, first_col = 0;
    block(halo.global_rows, num_procs, rank, local_rows, first_row);
    block(halo.global_cols, num_procs, rank, halo.local_cols, first_col);

    // Layout after the header: global_rows row sizes, then global_nnz column
    // indices, then the values (not needed here).
    std::vector<int32_t> row_sizes = read_int32s(file, 4LL + first_row, local_rows, swap, "row sizes");
    long long local_nnz = 0;
    for (int32_t s : row_sizes) {
        local_nnz += s;
    }
    long long first_nnz = 0;
    MPI_Exscan(&local_nnz, &first_nnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        first_nnz = 0;  // MPI_Exscan leaves rank 0's result undefined
    }
    std::vector<int32_t> cols =
        read_int32s(file, 4LL + halo.global_rows + first_nnz, local_nnz, swap, "column indices");

    // Off-process columns of this rank's rows, sorted and without repeats.
    std::vector<int> needed;
    for (int32_t c : cols) {
        if (c < first_col || c >= first_col + halo.local_cols) {
            needed.push_back(c);
        }
    }
    std::sort(needed.begin(), needed.end());
    needed.erase(std::unique(needed.begin(), needed.end()), needed.end());

    std::vector<int> first_cols(num_procs + 1);
    MPI_Allgather(&first_col, 1, MPI_INT, first_cols.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_cols[num_procs] = halo.global_cols;

    // needed is sorted and every rank owns one contiguous range, so the
    // requests come out grouped by owner in rank order, as Alltoallv expects.
    std::vector<int> request_counts(num_procs, 0);
    for (int c : needed) {
        const int owner = static_cast<int>(
            std::upper_bound(first_cols.begin(), first_cols.end(), c) - first_cols.begin()) - 1;
        halo.import_ranks.push_back(owner);
        halo.import_ids.push_back(c - first_cols[owner]);
        ++request_counts[owner];
    }

    // Tell each owner which of its columns this rank needs.
    std::vector<int> requested_counts(num_procs);
    MPI_Alltoall(request_counts.data(), 1, MPI_INT, requested_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> request_displs(num_procs, 0), requested_displs(num_procs, 0);
    for (int p = 1; p < num_procs; ++p) {
        request_displs[p] = request_displs[p - 1] + request_counts[p - 1];
        requested_displs[p] = requested_displs[p - 1] + requested_counts[p - 1];
    }
    std::vector<int> requested(requested_displs[num_procs - 1] + requested_counts[num_procs - 1]);
    MPI_Alltoallv(needed.data(), request_counts.data(), request_displs.data(), MPI_INT,
                  requested.data(), requested_counts.data(), requested_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    for (int p = 0; p < num_procs; ++p) {
        for (int i = requested_displs[p]; i < requested_displs[p] + requested_counts[p]; ++i) {
            halo.export_ids.push_back(requested[i] - first_col);
            halo.export_ranks.push_back(p);
        }
    }
    return halo;
}
