# irregular-benchmarking

Benchmarking and test harness for irregular, sparse MPI communication
patterns, built on [Cabana](https://github.com/CUP-ECS/Cabana/tree/mpiadvane-commspace-restructure)
and [Kokkos](https://github.com/kokkos/kokkos), with support for
locality-aware MPI extensions (MPI_Advance) and split-communicator
strategies for reducing per-rank memory overhead.

## Dependencies

- CMake >= 3.18
- MPI
- [Cabana](https://github.com/CUP-ECS/Cabana/tree/mpiadvane-commspace-restructure)
  (the CUP-ECS fork, `mpiadvane-commspace-restructure` branch - stock
  [ECP-copa/Cabana](https://github.com/ECP-copa/Cabana) doesn't have
  locality-aware support; built with MPI, Grid/Cajita, and
  locality-aware support)
- Kokkos (via Cabana)
- [locality_aware](https://github.com/mpi-advance/locality_aware) (a.k.a. MPI_Advance; provides the locality-aware MPI extensions used for halo exchange)
- [TCLAP](https://github.com/mirror/tclap)
- [nlohmann/json](https://github.com/nlohmann/json)

### Spack environment

`spack-envs/llnl/dane/spack.yaml` is a working Spack environment for Dane
(`spack-envs/llnl/tioga/` and `spack-envs/llnl/tuolumne/` are still empty
placeholders). It pulls `cabana` (with the `+locality_aware` variant) and
`localityaware` from a custom repo
([CUP-ECS/spack-packages](https://github.com/CUP-ECS/spack-packages),
`cabana-locality-aware` branch), plus `tclap`, `nlohmann-json`, and
`caliper` from builtin Spack.

The `cabana` package's `git` URL is CUP-ECS/Cabana, and its `master`
branch is unrelated to (260 commits ahead, 65 behind) the
`mpiadvane-commspace-restructure` branch this project actually needs -
double check the environment's `cabana` spec/`develop:` override points
at `mpiadvane-commspace-restructure`, not `master`.

To build the environment:

```bash
spack env create irregular-benchmarking spack-envs/llnl/dane/spack.yaml
spack env activate irregular-benchmarking
spack install
```

Then use the environment's view (or `spack location -i <package>` for
individual prefixes) as `SPACK_PREFIX` below.

## Building

```bash
mkdir build && cd build
cmake -DMPI_Advance_PREFIX=/path/to/locality_aware \
      -DSPACK_PREFIX=/path/to/spack/env/view \
      ..
make
```

## Usage

```bash
./irregular-benchmarking --filepath pattern_config.json [options]
```

| Flag | Long form | Description | Default |
|---|---|---|---|
| `-f` | `--filepath` | Path to the pattern config JSON file (required) | - |
| `-I` | `--samples` | Number of random samples per pattern | `25` |
| `-k` | `--window` | Startup-window gathers per sample, timed as one total | `10` |
| `-n` | `--timed-iterations` | Gathers timed after the startup window (at least 1) | `1000` |
| | `--calc` | Budget in seconds; print the `n` that fills it (see below) | off |
| `-c` | `--comm` | Communication backend: `MPIA`\|`A`\|`a` (MPI_Advance, default) or `MPI`\|`M`\|`m` | `MPIA` |
| `-x` | `--type` | Halo direction: `EXPORT`\|`E`\|`e` (default) or `IMPORT`\|`I`\|`i` | `EXPORT` |
| `-s` | `--split-type` | Communicator split: `SOCKET`\|`S`\|`s` (default), `NUMA`\|`U`\|`u`, or `NODE`\|`N`\|`n` | `SOCKET` |
| `-a` | `--alltoallv` | Neighbor alltoallv init: `STANDARD`\|`S`\|`s` (default) or `LOCALITY`\|`L`\|`l` | `STANDARD` |
| `-C` | `--crs` | CRS discovery method: `default`, `nonblocking`, `personalized`, `personalized_loc`, `nonblocking_loc`, `rma` | `default` |
| `-S` | `--seed` | Integer seed for random sampling; the same seed reproduces the same draws | current time |
| `-q` | `--unique-seed` | Use a distinct seed per rank (`seed + rank`) | off |
| `-b` | `--barrier` | Time each gather barrier to barrier, so the wait for the slowest rank is included; without it ranks can drift out of sync | off |
| `-P` | `--patterns` | Comma-separated pattern names to run, in the order given; unknown or repeated names are an error | every pattern, in name order |
| `-W` | `--warmup` | Untimed point-to-point warmup before any pattern: `none`, `nearest` (rank ± 1), or `all` (every other rank) | `none` |
| `-V` | `--verify` | After each MPI_Advance sample, gather once through plain MPI and once through MPI_Advance on the same halo and report ranks whose ghost data differ | off |
| `-U` | `--distinct-neighbors` | Draw each rank's neighbors without replacement, so a rank never draws the same partner twice | off |
| `-r` | `--report-params` | Print the resolved run configuration before benchmarking | off |
| `-N` | `--nosy-percent` | Percent chance (0-100) a rank sleeps before each gather, simulating a noisy neighbor | `0` |
| `-T` | `--nosy-time` | Sleep duration in ms when acting as a noisy neighbor | `0` |
| `-d` | `--distribution` | Accepted for `gaussian`\|`empirical`\|`static` but not currently consumed by the benchmark loop | `gaussian` |

For each pattern in the config file, the benchmark draws `--samples` random
draws of a neighbor count, per-neighbor buffer sizes, and neighbor
placement from that pattern's distributions, builds a `Cabana::Halo`, and
runs `k + n` gather calls. Min/max/average timings (halo construction, AoSoA
resize, gather setup, gather apply) are reduced across ranks and printed on
rank 0. The backend pays lazy setup in its first few applies, so the first
`k` are totalled as `startupWindow` (startup + q·k) and the next `n` as
`timedTotal` (q·n), where q is the steady-state time per gather.

With `--calc <seconds>`, rank 0 also solves those two equations from the
max-over-ranks totals for q and startup, and prints the `n` for which
startup + q·(k + n) fills the budget. The budget covers one sample, and an
invocation runs `--samples` of them for every pattern it runs, so divide a launch's
total budget by the number of samples it will run.

Each sample also reports where its messages go: `onNodeNeighbors` and
`offNodeNeighbors` count messages to ranks on the same node or another one
(the same `MPI_COMM_TYPE_SHARED` grouping locality_aware uses), with their
bytes in `onNodeBytes` and `offNodeBytes`. `inNumaBytes` and `outNumaBytes`
split the bytes by whether the partner shares this rank's NUMA domain, read
from the rank's CPU affinity; they are omitted, with a note at startup, when
any rank is not bound within a single NUMA domain. A self-send counts as
local.

`--warmup` exchanges one 64 B and one 128 KiB message with each warmup
partner (rank ± 1, or every other rank) on a throwaway duplicate of
`MPI_COMM_WORLD`, so those connections are open before the first pattern
while no plan or communicator cache exists yet. `all` puts every sample and
backend in the same state: if the MPI transport connects lazily, whichever
sample first reaches a partner otherwise pays to connect to it. Its cost grows
with the square of the rank count. On Dane, connection setup lands in the halo
build (`haloTime`, roughly 150-300 ms at 2-16 nodes without `all`), not in
`startupWindow`.

Draws come from one random stream in pattern order, so with the same seed a
pattern selected by `--patterns` gets different draws than it does in a run
of every pattern.

`--report-params` also prints the seed, the MPI library version, and
the compiler version.

### Recording provenance

`scripts/provenance.sh` records a launch's provenance from inside the sbatch
job: job id, start time, node list, tasks per node, per-rank CPU binding, raw
fabric queries for switch placement (Slurm on Dane has no topology), the
compiler, MPI library, and build flags, the benchmark commit with its
uncommitted diff, the Spack builds of localityaware, cabana, and kokkos and
the state of their dev checkouts (with `PROV_SPACK_ENV` set), the pattern
config's hash and pattern names, and the MPI- and Slurm-related environment.
Pass it the benchmark's own srun geometry so the binding it records matches:

```bash
PROV_SPACK_ENV=~/spackenvs/spack-k5 scripts/provenance.sh OUTDIR build/src/irregular-benchmarking \
    pattern_config.json -- --nodes=2 --ntasks-per-node=112
```

`scripts/pattern-names.py CONFIG.json` prints a config's pattern names, one
per line.

### Running a campaign on Slurm

Three scripts run the data collection: one pilot per node count fixes `k` and
`n`, then any number of launches at that node count reuse them.

- `scripts/pilot.sbatch NODES INPUT_FILE TARGET_SECONDS [WARMUP]` records
  provenance, runs every pattern once on STANDARD with `k = 25`, `n = 100`, and
  `--calc TARGET_SECONDS`, and writes `<jobid>-pilot-<nodes>/targets` (one
  `pattern k n` line per pattern). The job fails unless every pattern gets
  `n >= 1`.
- `scripts/gatherdata.sbatch NODES INPUT_FILE TARGETS_FILE [WARMUP]` is one
  launch. It records provenance, copies `targets` into
  `<jobid>-gatherdata-<nodes>/`, shuffles pattern × {MPI, STANDARD, LOCALITY} ×
  `SAMPLES` with a seed it saves to `shuffle_seed` (set `SHUFFLE_SEED` to repeat
  an order), and runs one `srun` per sample with `-I 1 -b -q -r -W WARMUP`.
  `manifest.csv` gets one row per run: pattern, arm, seed, `k`, `n`, start and
  end time, and exit status.
- `scripts/submit.sh NODES INPUT_FILE TARGET_SECONDS LAUNCHES [WARMUP]` submits
  the pilot and then `LAUNCHES` gatherdata jobs with
  `--dependency=afterok:<pilot>` and `--kill-on-invalid-dep=yes`, so the
  launches start only once the pilot has succeeded and are cancelled if it
  fails. Run it from the directory the results should go to.

`WARMUP` is the benchmark's `-W` and defaults to `nearest`. Both sbatch scripts
take `NODES` as an argument but cannot change their allocation, so they check
it against `sbatch -N`; `submit.sh` passes both. sbatch runs a spooled copy of
each script, so the checkout path (`REPO`), `BINARY`, and `PROV_SPACK_ENV` are
set at the top of `pilot.sbatch` and `gatherdata.sbatch`; edit them for your
own checkout. For example, 40 launches at each of four node counts:

```bash
mkdir -p ~/campaign && cd ~/campaign
for n in 2 4 8 16; do
  /path/to/irregular-benchmarking/scripts/submit.sh $n /path/to/pattern_config.json 10 40
done
```

Each `srun` costs roughly 10-15 s to start, so a launch takes about
patterns × arms × (TARGET_SECONDS + 15 s), longer where an arm is slower
than STANDARD (every arm uses the `n` the STANDARD pilot chose).

Neighbor placement is drawn with replacement by default, so a rank can draw
the same partner twice and its realised partner count then falls short of the
neighbor count it drew. `--distinct-neighbors` redraws on a collision instead,
which makes the realised count match the drawn one; it stops redrawing once
the offset distribution has no unused target left, so it cannot spin.

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
- `scripts/` - provenance recording and the Slurm campaign scripts (see
  "Running a campaign on Slurm")
- `cmake/` - project-local CMake find modules
- `spack-envs/` - Spack environment files for supported systems

## Contributing

See the issue and pull request templates under `.github/` for the expected
format when filing bugs, feature requests, or PRs. Code is formatted with
`clang-format` (see `.clang-format`); a CI check enforces this on pull
requests.
