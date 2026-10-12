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
./irregular-benchmarking --filepath matrix.pm [options]
./irregular-benchmarking --filepath capture_dir/ [options]
```

| Flag | Long form | Description | Default |
|---|---|---|---|
| `-f` | `--filepath` | A JSON pattern config, a PETSc binary matrix (`.pm`) whose SpMV halo is exchanged, or a Vernier capture directory whose per-rank histograms each rank samples (required) | - |
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
`timedTotal` (q·n), where q is the steady-state time per gather. With
`--barrier`, each gather is timed from the end of the previous barrier
through the apply and the barrier after it, so these totals include the wait
for the slowest rank. `windowApplyOnly` and `timedApplyOnly` are the same two
totals with each gather's timer stopped before its barrier: this rank's apply
alone. Their Max column is the largest per-rank total, not the sum of each
gather's slowest apply. The barrier still keeps ranks in step. Without
`--barrier` the two pairs are equal.

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

### Matrix input

Given a `.pm` file (the PETSc binary format ABCG reads; ABCG's
`test_data/convert_petsc.py` converts SuiteSparse Matrix Market files to it),
the benchmark exchanges that matrix's SpMV halo instead of sampling a pattern.
Rows and the x vector are split into contiguous blocks exactly as ABCG's
`readParMatrix` splits them, and each rank sends every x entry another rank's
rows reference, one double each. The neighbors, message sizes, and entries are
therefore the same as ABCG's SpMV exchange at the same rank count.

- The matrix is one pattern, named after the file (`nd12k.pm` is `nd12k`), so
  `--patterns`, the pilot, and `targets` work as for a JSON config. Nothing is
  random, so every sample is the same exchange.
- With `--type IMPORT` the halo is built from what each rank receives (owner
  rank and the owner's local index), and `data_sent` and the per-message rows
  then count incoming bytes.
- Building the lists exchanges request counts with every rank (one
  `MPI_Alltoall` plus an `MPI_Alltoallv`) before anything is timed, so
  connections to most partners are already open when the first sample starts.
- The matrix needs at least as many rows and columns as there are ranks.

`-V` checks the gathered ghost data against plain MPI. On Dane, with both the
current and the old locality_aware, EXPORT halos pass and IMPORT halos report
differing per-sender data on every rank, for JSON patterns as well as
matrices, so IMPORT results through MPI_Advance should not be trusted yet.

### Vernier capture input (fine patterns)

Given a directory, the benchmark reads a Vernier capture of one pattern: one
`vernier-<rank>-0.hist-hierarchical.json` per rank, as written by ABCG's
`benchmark_spmv` built with `ABCG_VERNIER` (see "Running a campaign on Slurm").
Each rank reads only its own file, which holds, per MPI call type and per
partner rank, a histogram of message sizes.

- The pattern is named `fine_<name>`, where `<name>` is the pattern Vernier
  recorded in the files (ABCG names it after the matrix), so the name travels
  with the data rather than with the directory.
- With `--type EXPORT` (the default) each rank replays its recorded sends,
  since Cabana works out the receives from the export lists; with `--type
  IMPORT` it replays its recorded receives. Every message goes to its recorded
  partner, with a size drawn within that partner's own size bins. A capture of
  one exchange (as `pilot.sbatch` makes) has one bin per partner, so the lists
  are exactly the captured ones. A capture of several exchanges would replay
  every exchange's messages at once, so capture one.
- Partners are not drawn, so `--distinct-neighbors` does not apply.
- The capture must come from the same number of ranks as the run; otherwise
  the run stops with an error.

### Recording provenance

`scripts/provenance.sh` records a launch's provenance from inside the sbatch
job: job id, start time, node list, tasks per node, per-rank CPU binding, raw
fabric queries for switch placement (Slurm on Dane has no topology), the
compiler, MPI library, and build flags, the benchmark commit with its
uncommitted diff, the Spack builds of localityaware, cabana, and kokkos and
the state of their dev checkouts (with `PROV_SPACK_ENV` set), each input's
hash and pattern names (for a Vernier capture directory, its per-rank files
hashed together and the capture's `vernier_settings`), and the MPI- and
Slurm-related environment.
Pass it the benchmark's own srun geometry so the binding it records matches:

```bash
PROV_SPACK_ENV=~/spackenvs/spack-k5 scripts/provenance.sh OUTDIR build/src/irregular-benchmarking \
    pattern_config.json -- --nodes=2 --ntasks-per-node=112
```

`scripts/pattern-names.py CONFIG.json|MATRIX.pm|CAPTURE_DIR` prints the
input's pattern names, one per line.

### Running a campaign on Slurm

Three scripts run the data collection: one pilot per node count fixes `k` and
`n`, then any number of launches at that node count reuse them.

- `scripts/pilot.sbatch NODES INPUTS TARGET_SECONDS [WARMUP]` records
  provenance, runs every pattern once on STANDARD with `k = 25`, `n = 100`, and
  `--calc TARGET_SECONDS`, and writes `<jobid>-pilot-<nodes>/targets` (one
  `pattern k n` line per pattern). The job fails unless every pattern gets
  `n >= 1` and no pattern name comes from two input files.
- `scripts/gatherdata.sbatch NODES INPUTS TARGETS_FILE [WARMUP]` is one
  launch. It records provenance, copies `targets` into
  `<jobid>-gatherdata-<nodes>/`, shuffles pattern × {MPI, STANDARD, LOCALITY} ×
  `SAMPLES` with a seed it saves to `shuffle_seed` (set `SHUFFLE_SEED` to repeat
  an order), and runs one `srun` per sample with `-I 1 -b -q -r -W WARMUP`.
  `manifest.csv` gets one row per run: pattern, arm, seed, `k`, `n`, start and
  end time, and exit status.
- `scripts/submit.sh NODES INPUTS TARGET_SECONDS LAUNCHES [WARMUP]` submits
  the pilot and then `LAUNCHES` gatherdata jobs with
  `--dependency=afterok:<pilot>` and `--kill-on-invalid-dep=yes`, so the
  launches start only once the pilot has succeeded and are cancelled if it
  fails. Run it from the directory the results should go to.

`INPUTS` is one input file or several, comma-separated, mixing JSON pattern
configs and `.pm` matrices; every pattern from every file is pilot-sized and
shuffled into each launch, and each run is given the file its pattern came from.
`WARMUP` is the benchmark's `-W` and defaults to `nearest`. Both sbatch scripts
take `NODES` as an argument but cannot change their allocation, so they check
it against `sbatch -N`; `submit.sh` passes both. sbatch runs a spooled copy of
each script, so the checkout path (`REPO`), `BINARY`, and `PROV_SPACK_ENV` are
set at the top of `pilot.sbatch` and `gatherdata.sbatch`; edit them for your
own checkout. Both scripts also take `BINARY` and `EXTRA_ARGS` from the
environment at submission, e.g. to run a build against another locality_aware
with `-s NUMA`, and record both in `launch_settings`. When `BINARY` was built
from a git archive rather than a checkout, also set `BENCH_COMMIT`, since
provenance.sh then has no repository to read the commit from.

`submit.sh` takes two more settings from the environment. `MAX_RUNNING`, if
set, makes launch i also wait (`afterany`) for launch i - `MAX_RUNNING`, so at
most that many of one call's launches run at once and the rest are spread over
time. `EXCLUDE_NODES`, if set, is passed to every sbatch as `--exclude`.

`ABCG_SPMV` and `ABCG_CG`, if set in the environment, are ABCG's
`benchmark_spmv` (the SpMV halo exchange alone) and `benchmark_cg_pilot` (CG
iterations), built from ABCG's `pilots` branch. Each `.pm` input then also
gets the patterns `abcgspmv_<matrix>` and `abcgcg_<matrix>`: `pilot.sbatch`
sizes each with that program's own STANDARD pilot (same `k`, `n`, warmup, and
budget), and `gatherdata.sbatch` shuffles them into each launch on the same
arms, running the matching program with `-c ARM -P PATTERN -W WARMUP -k K -n N`.
Their output rows use this benchmark's format, so the analysis reads them as
further patterns. ABCG's MPI arm is ABCG's own Isend/Irecv exchange and
`MPI_Allreduce`, not Cabana's; set `ABCG_SPMV_MPI` and `ABCG_CG_MPI` to the same
programs built without MPI Advance and the MPI arm runs those. The STANDARD and
LOCALITY arms use MPI Advance's neighbor alltoallv (persistent by default) and
its allreduce with `ALLREDUCE_PMPI`. `ABCG_PERSISTENT` (`on`, the default,
`off`, or `on off`) picks the MPI Advance neighbor alltoallv: persistent
(init once, then start/wait) as `abcgspmv_`/`abcgcg_` patterns, one call per
exchange as `abcgspmvnp_`/`abcgcgnp_` patterns, each with its own pilot.
`ABCG_EXTRA_ARGS` reaches every ABCG run.

`VERNIER_SPMV`, if set, is ABCG's `benchmark_spmv` built with `ABCG_VERNIER`
(MPI arm only, no MPI Advance). Before the pilots, `pilot.sbatch` captures one
SpMV exchange of each `.pm` input with it, one `srun` each in `INPUTS` order,
into `<pilot dir>/capture/NN_<matrix>/`. `capture/order.csv` records that order
with each matrix's path, sha256, directory, and exit status, and
`capture/vernier_settings` the capture program, libvernier, Vernier commit,
command, and environment. `scripts/vernier-config.py` then checks every capture
and writes `coarse.json` (one `coarse_<matrix>` pattern per matrix: rank 0's
aggregate histograms, the same for every rank, sends and receives together)
and `generated_inputs` (that file and the capture directories, whose patterns
are `fine_<matrix>`: each rank's own recorded sends). The pilot sizes
those like any other input, and `gatherdata.sbatch` adds `generated_inputs` from
next to the targets file, so every launch runs `<matrix>`, `coarse_<matrix>`, and
`fine_<matrix>` side by side. Captures run with `VERNIER_NO_REQUESTS=1`, because
Vernier 09d650b otherwise aborts every rank but 0 in `MPI_Finalize`.

`scripts/vernier-to-pm.py` writes a PETSc binary matrix (`.pm`) whose SpMV halo
is a Vernier pattern, so ABCG (which reads only `.pm`) and this benchmark's `.pm`
mode can run samples as well as real matrices:

```bash
scripts/vernier-to-pm.py fine CAPTURE_DIR out.pm [--symmetric]
scripts/vernier-to-pm.py coarse coarse.json coarse_<matrix> RANKS out.pm --seed S [--distinct-neighbors] [--symmetric]
```

`fine` takes each rank's recorded sends from a capture directory; `coarse` draws
one sample of an aggregate pattern the way the benchmark draws it. Rank p
sending k doubles to rank r becomes k distinct columns of p's block referenced
by r's rows; each rank owns as many rows and columns as its largest message,
with only the diagonal locally. The matrix has to run on exactly as many ranks
as the pattern has. By default it is nonsymmetric with a dominant diagonal, so
CG runs without an indefinite-matrix abort but is not true CG; `--symmetric`
pairs every p->r message with r->p into a symmetric positive definite matrix
(true CG), and stops if some message has no reverse, as in coarse samples.
`out.pm.json` records the source, settings, and every rank's sends. Checked
locally: ABCG captured on a `fine` matrix sends exactly the original capture's
messages, on a `coarse` one exactly the sidecar's, and symmetric matrices take
the same CG iteration count as ABCG's own CG.

For example, 40 launches at each of four node counts:

```bash
mkdir -p ~/campaign && cd ~/campaign
for n in 2 4 8 16; do
  /path/to/irregular-benchmarking/scripts/submit.sh $n /path/to/pattern_config.json 10 40
done
```

or, capturing the matrices first so each launch runs every matrix beside its
coarse and fine samples:

```bash
export VERNIER_SPMV=/path/to/ABCG/build-vernier/benchmarks/benchmark_spmv
for n in 2 4 8 16; do
  /path/to/irregular-benchmarking/scripts/submit.sh $n /path/to/a.pm,/path/to/b.pm 10 40
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
    "dist_to_neighbors": { "4": { "1": 2, "-1": 2 }, "8": { "2": 1, "-2": 2 } }
  }
}
```

- `comm_partners`: histogram of neighbor counts observed (key = number of
  neighbors, value = number of occurrences).
- `buffer_size`: histogram of per-message buffer sizes in bytes. A drawn key
  is sent as that many bytes, unless the pattern has `"buffer_size_bins":
  "lower_edge"`: then the keys are the lower edges of equal-width bins (as in
  Vernier's histograms) and each size is drawn uniformly within its bin, the
  last bin taken to be as wide as the first. `scripts/vernier-config.py` marks
  `coarse.json` this way. Sent at their lower edges, the SuiteSparse SpMV
  captures at 2 nodes came out 8-16% short of the real mean message size;
  drawn within the bins, 0-1%.
- `dist_to_neighbors`: for each neighbor count, a histogram of neighbor
  rank offsets relative to the sending rank.

Other keys (older configs carry `pattern_count` and `message_count`) are
ignored. Each histogram is fill-forwarded (zero entries take the last non-zero
value) and normalized into a probability distribution before sampling.

## Layout

- `src/` - benchmark source: `main.cpp` wires together CLI parsing
  (`cli.*`), pattern loading (`pattern.*`), matrix input (`matrix.*`),
  Vernier capture input (`fine.*`), sampling (`sampling.*`), and the
  benchmark loop (`benchmark.*`); options shared across those are in
  `options.*`
- `scripts/` - provenance recording, the Slurm campaign scripts, and the
  Vernier tools `vernier-config.py` and `vernier-to-pm.py` (see "Running a
  campaign on Slurm")
- `scripts/dane/` - the helpers these campaigns were built and checked with on
  Dane, kept as they ran there (paths are `/g/g20/bacon4` and Dane's
  `/usr/tce` toolchain): `build_step_bench.sh` builds the benchmark against one
  locality_aware commit (with `cli-nosplit.patch` for libraries without
  `MPIL_Set_split`), `build_abcg_pilots.sh` builds ABCG's pilot programs
  (`build-mpi`, `build-old`, `build-vernier`), and `build_vernier.sh` builds
  Vernier for the captures. The rest (`build_step.sh`, `build_candidate.sh`,
  `submit_*`, `summ_*`, `bindcheck.sh`, `countsr.c`, and the `.patch` files) is
  from the 2026-09-28 old-to-new locality_aware slowdown walk and the first
  ABCG builds.
- `cmake/` - project-local CMake find modules
- `spack-envs/` - Spack environment files for supported systems

## Contributing

See the issue and pull request templates under `.github/` for the expected
format when filing bugs, feature requests, or PRs. Code is formatted with
`clang-format` (see `.clang-format`); a CI check enforces this on pull
requests.
