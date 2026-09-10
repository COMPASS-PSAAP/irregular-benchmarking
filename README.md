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

## Layout

- `src/` - core library code
- `examples/` - example/benchmark drivers
- `tests/` - unit tests (via BLT + GoogleTest)
- `cmake/` - project-local CMake find modules
- `spack-envs/` - Spack environment files for supported systems
- `blt/`, `tclap/` - vendored dependencies (git submodules)

## Contributing

See the issue and pull request templates under `.github/` for the expected
format when filing bugs, feature requests, or PRs. Code is formatted with
`clang-format` (see `.clang-format`); a CI check enforces this on pull
requests.
