# irregular-benchmarking

Benchmarking and test harness for irregular, sparse MPI communication
patterns, built on [Cabana](https://github.com/ECP-copa/Cabana) and
[Kokkos](https://github.com/kokkos/kokkos), with support for locality-aware
MPI extensions (MPI_Advance), NuMesh, and split-communicator strategies for
reducing per-rank memory overhead.

## Dependencies

- CMake >= 3.18
- MPI
- [Cabana](https://github.com/ECP-copa/Cabana) (built with MPI and Grid/Cajita support)
- Kokkos (via Cabana)
- MPI_Advance
- NuMesh
- [TCLAP](https://github.com/mirror/tclap) (vendored as a git submodule)
- [nlohmann/json](https://github.com/nlohmann/json)
- [BLT](https://github.com/LLNL/blt) (vendored as a git submodule, used for the CMake build/test scaffolding)

TCLAP, nlohmann/json, and vernier are located via `find_path`/`find_library`
against `${SPACK_PREFIX}`, so point `SPACK_PREFIX` (or your compiler's default
search paths) at an environment providing them. `spack-envs/llnl/` contains
example Spack environments for LLNL systems (Dane, Tioga, Tuolumne).

## Building

This repository uses git submodules (`blt`, `tclap`). Clone with:

```bash
git clone --recurse-submodules <repo-url>
```

or, if already cloned:

```bash
git submodule update --init --recursive
```

Then configure and build with CMake, pointing at your MPI_Advance and NuMesh
installs:

```bash
mkdir build && cd build
cmake -DMPI_Advance_PREFIX=/path/to/mpi_advance \
      -DNuMesh_PREFIX=/path/to/numesh \
      ..
make
```

See `run_cmake.sh` for an example invocation.

## Usage

```bash
./irregular-benchmarking --filepath pattern_config.json [options]
```

| Flag | Long form | Description | Default |
|---|---|---|---|
| `-f` | `--filepath` | Path to the pattern config JSON file (required) | - |
| `-I` | `--samples` | Number of random samples per pattern | `25` |
| `-i` | `--iterations` | Number of gather iterations per sample | `1` |
| `-c` | `--comm` | Communication backend: `MPIA`\|`A`\|`a` (MPI_Advance, default) or `MPI`\|`M`\|`m` | `MPIA` |
| `-x` | `--type` | Halo direction: `EXPORT`\|`E`\|`e` (default) or `IMPORT`\|`I`\|`i` | `EXPORT` |
| `-s` | `--split-type` | Communicator split: `SOCKET`\|`S`\|`s` (default), `NUMA`\|`U`\|`u`, or `NODE`\|`N`\|`n` | `SOCKET` |
| `-a` | `--alltoallv` | Neighbor alltoallv init: `STANDARD`\|`S`\|`s` (default) or `LOCALITY`\|`L`\|`l` | `STANDARD` |
| `-C` | `--crs` | CRS discovery method: `default`, `nonblocking`, `personalized`, `personalized_loc`, `nonblocking_loc`, `rma` | `default` |
| `-S` | `--seed` | Seed for random sampling | none |
| `-q` | `--unique-seed` | Use a distinct seed per rank (`seed + rank`) | off |
| `-b` | `--barrier` | Insert an `MPI_Barrier` between gather iterations (excluded from timing) | off |
| `-r` | `--report-params` | Print the resolved run configuration before benchmarking | off |
| `-N` | `--nosy-percent` | Percent chance (0-100) a rank sleeps before each gather, simulating a noisy neighbor | `0` |
| `-T` | `--nosy-time` | Sleep duration in ms when acting as a noisy neighbor | `0` |
| `-d` | `--distribution` | Accepted for `gaussian`\|`empirical`\|`static` but not currently consumed by the benchmark loop | `gaussian` |

For each pattern in the config file, the benchmark draws `--samples` random
draws of a neighbor count, per-neighbor buffer sizes, and neighbor
placement from that pattern's distributions, builds a `Cabana::Halo`, and
times `--iterations` gather calls. Min/max/average timings (halo
construction, AoSoA resize, gather setup, gather apply) are reduced across
ranks and printed on rank 0.

### Pattern config format

The `--filepath` argument points to a JSON file mapping pattern names to
their communication statistics:

```json
{
  "my_pattern": {
    "comm_partners": { "4": 10, "8": 3 },
    "buffer_size": { "64": 5, "128": 2 },
    "dist_to_neighbors": { "4": { "1": 2, "-1": 2 }, "8": { "2": 1, "-2": 2 } },
    "pattern_count": 1,
    "message_count": 13
  }
}
```

- `comm_partners`: histogram of neighbor counts observed (key = number of
  neighbors, value = number of occurrences).
- `buffer_size`: histogram of per-message buffer sizes in bytes.
- `dist_to_neighbors`: for each neighbor count, a histogram of neighbor
  rank offsets relative to the sending rank.
- `pattern_count`, `message_count`: optional; `message_count` defaults to
  half the sum of all `dist_to_neighbors` values.

Each histogram is fill-forwarded (zero entries take the last non-zero
value) and normalized into a probability distribution before sampling.

## Layout

- `src/` - benchmark source: `main.cpp` wires together CLI parsing
  (`cli.*`), pattern loading (`pattern.*`), sampling (`sampling.*`), and
  the benchmark loop (`benchmark.*`); options shared across those are in
  `options.*`
- `tests/` - unit tests (via BLT + GoogleTest)
- `cmake/` - project-local CMake find modules
- `spack-envs/` - Spack environment files for supported systems
- `blt/`, `tclap/` - vendored dependencies (git submodules)

## Contributing

See the issue and pull request templates under `.github/` for the expected
format when filing bugs, feature requests, or PRs. Code is formatted with
`clang-format` (see `.clang-format`); a CI check enforces this on pull
requests.
