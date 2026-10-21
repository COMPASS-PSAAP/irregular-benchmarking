# Task: statistical analysis of the 2026-10-13 five-way campaign

You are analyzing a benchmark campaign run on LLNL's Dane cluster. Your job is to
read the data, run the statistics the research plan calls for, and produce figures
and a written summary of what the data does and does not show. Work carefully: say
"the data cannot answer this" when that is the truth, and never present a number
without the sample size and interval behind it.

## 1. Read these first

1. The research plan repository: https://github.com/COMPASS-PSAAP/irregular-comm-tradeoff-plan
   (private; clone with `git clone git@github.com:COMPASS-PSAAP/irregular-comm-tradeoff-plan.git`).
   Read `README.md` in full, then in `RESEARCH-PLAN.md`: §1 (what the study must
   produce; worst-case regret), §4 (hypotheses, especially H0, H8, H13, H17, H18),
   §5.4 (measurement protocol and statistics), §5.7 (the pattern-replay benchmark and
   the fidelity ladder), §6.3 (Phase 1C: does pattern replay predict the
   application?), and §13 (scope limits).
2. The proposal synthesis page:
   https://claude.ai/code/artifact/a5415204-7293-4b8f-81eb-e2355b567c03?sk=X-k853RzRvm1HntldAWw0g
   It summarizes the same plan for reviewers. Where it and the repository disagree,
   the repository wins.
3. `data/README.txt` and `data/campaign.sh` in this folder: how the campaign was
   built and run, the exact code commits, and its pass criteria.

**What this dataset is, in the plan's terms:** a Phase 1C-shaped experiment (§6.3) on
Dane. It tests whether a Vernier characterization of an application's halo exchange,
replayed through the pattern-replay benchmark, ranks the communication arms the way
the application does (H17), and whether the cheap aggregate Vernier data manager is
enough or the per-rank hierarchical one is needed (H18, the §5.7 fidelity ladder).
It can also speak to H0 (does the best arm change across matrices and scales), H8
(setup amortization), and H13 (median-best vs. spread-best), within the limits in §6
below.

## 2. The experiment

Six SuiteSparse matrices: crankseg_2, nd12k, x104, crankseg_1, m_t1, nd24k. Four node
counts: 2, 4, 8, 16 (112 MPI ranks per node: 224, 448, 896, 1792 ranks). The matrix
rows and columns are split into one contiguous block per rank; the SpMV halo is the
off-block x entries each rank needs.

For each node count, a **pilot** job first captured each matrix's SpMV halo exchange
with Vernier (one exchange, plain MPI) at that node count, then sized every pattern's
iteration count `n` to about 10 s on the STANDARD arm. Then **40 launches** (separate
Slurm allocations, at most 4 running at once, spread over about a day) each ran all
126 runs below in their own random order, one `srun` per run.

**Seven kinds** (the `pattern` column; `<m>` is the matrix name):

| Kind | Program | What it exchanges |
|---|---|---|
| `<m>` | irregular-benchmarking (the replay benchmark) reading the matrix `.pm` | The matrix's exact SpMV halo, through Cabana's gather |
| `fine_<m>` | irregular-benchmarking reading the Vernier **hierarchical** capture | Each rank replays its own recorded sends, exact partner and size |
| `coarse_<m>` | irregular-benchmarking reading the Vernier **aggregate** histograms | Every rank draws a neighbor count, partners (by neighbor distance), and sizes from one machine-wide histogram |
| `abcgspmv_<m>` | ABCG `benchmark_spmv`, persistent | The application's own halo exchange: pack, exchange, barrier; no multiply |
| `abcgspmvnp_<m>` | the same, non-persistent | One MPI Advance call per exchange instead of a persistent request |
| `abcgcg_<m>` | ABCG `benchmark_cg_pilot`, persistent | Full CG iterations: SpMV + 2 allreduces + vector updates (no barrier) |
| `abcgcgnp_<m>` | the same, non-persistent | |

ABCG is the application. `abcgcg` is the closest thing to solver-level time
(`t_iter`); everything else is exchange-level (`t_exchange`). The plan's scope rule
applies: never report an exchange-level result as a solver-level one.

**Three arms** (the `arm` column), which are the plan's "regimes":

- `MPI`: plain MPI point-to-point. No aggregation.
- `STANDARD`: MPI Advance (locality_aware 752a19b) neighbor alltoallv, no
  aggregation.
- `LOCALITY`: MPI Advance locality-aware neighbor alltoallv, which aggregates
  messages through NUMA-region leaders (`-s NUMA`). This is the one aggregating arm.

So each launch is 6 matrices x 7 kinds x 3 arms = 126 runs; each node count has
40 x 126 = 5,040 runs.

**Fixed settings:** Open MPI 4.1.2, gcc 13.3.1, pbatch (shared queue, so the
"shared" interference condition in §5.4), nodes dane261 and dane764 excluded,
`-W nearest` untimed connection warmup, window `k = 25`, and every benchmark gather
closed by a barrier (`-b`).

## 3. Where the data is and how to read it

The data is in `data/campaign/`:

- `<jobid>-pilot-<N>/`: the pilot for N nodes.
  - `targets` has one line per pattern: `pattern k n`.
  - `capture/order.csv` lists the Vernier captures in order.
  - `capture/NN_<matrix>/` holds the per-rank Vernier files
    (`vernier-<rank>-0.hist-hierarchical.json`) and traces.
  - `coarse.json` holds the aggregate histograms the `coarse_` kind samples from.
- `<jobid>-gatherdata-<N>/`: one launch.
  - `manifest.csv` has one row per run: `run,pattern,arm,seed,k,n,start,end,exit,file`.
    `run` is the position in this launch's random order, and `start` and `end` are
    wall-clock timestamps.
  - `run_NNNN_<pattern>_<arm>_<time>.txt` holds each run's output.
  - `provenance_*.txt` (with `.binding.txt` and `.fabric.txt`) record the node list,
    CPU binding, fabric, builds, and input hashes. `schedule` and `shuffle_seed`
    record the order.
- `data/README.txt` documents all of this.

**Run file format.** A header of `-key: value` lines (`-MPI`, `-File`, `-patterns`,
`-window (k)`, `-timed iterations (n)`, `-alltoallv`, `-persistent`, ...), then a table
with one row per metric:

```
(<pattern>)  <metric>   <min over ranks>   <max over ranks>   <average over ranks>
```

All times are in **milliseconds**, summed over the iterations stated. Benchmark
STANDARD/LOCALITY runs end with a `verify: PASS` line.

**Metrics: irregular-benchmarking** (`<m>`, `fine_`, `coarse_`):

| Metric | Meaning |
|---|---|
| `timedTotal` | Sum over the n timed gathers, each timed from its start through the apply and the barrier after it. Includes waiting for the slowest rank. **Time per exchange = timedTotal / n.** |
| `startupWindow` | The same over the first k = 25 gathers, which absorb lazy setup. |
| `timedApplyOnly`, `windowApplyOnly` | The same totals with each timer stopped *before* the barrier: this rank's own apply time. Min/max/avg across ranks are the per-rank spread, the "time each rank ahead of the synchronization point" quantity in §5.4. |
| `haloTime`, `gatherTime`, `resizeTime` | One-time setup: halo construction (includes MPI connection setup), gather plan, buffer. |
| `numberOfmessages`, `data_sent` | Per rank, messages and bytes sent per exchange. |
| `onNode*`/`offNode*` | The same, split by on-node vs. off-node partner. |
| `inNumaBytes`/`outNumaBytes` | Bytes split by same-NUMA vs. other-NUMA partner. |
| `apply`, `halo_gather_time` | Totals including the window, plus setup. |

**Metrics: ABCG** (`abcg*`):

| Metric | Meaning |
|---|---|
| `timedTotal` | Sum over exactly n timed iterations. **Time per iteration = timedTotal / n.** |
| `startupWindow` | The k = 25 iterations before the timed ones. |
| `setupTime` | Communication-package setup. |
| `numberOfmessages`, `data_sent` | As above. |
| `cgSolves`, `cgItersPerSolve`, `cgReferenceIters` | CG only. The n iterations are spread over `cgSolves` solves, each restarting from x = 0. All six matrices hit the 500-iteration cap without converging (checked at 2 and 16 nodes), so these are fixed-iteration-count timings, and the CG correctness check (iterations equal ABCG's own CG()) is vacuous. `cgItersPerSolve` = 0 means no solve finished within n (e.g. `abcgcg_crankseg_2` at 16 nodes, where n < 500). |
| `verifyErrors` | SpMV only. |

Do not compare raw `timedTotal` values: n differs by pattern and by node count.
Always divide by `n`. Within one node count, n is the same for a pattern across
arms and launches.

## 4. Facts already established (do not re-derive; you may spot-check)

- Every pass check held at every node count: all captures complete, every run exit
  0, every `verify: PASS`, no ABCG `check FAILED`. All 40 launches are complete at
  every node count (8013856 and 8013860 were pulled on 2026-10-14 after they finished).
- **The round trip closes exactly for `fine_`.** Per-rank message counts and bytes
  (min/max/avg) of every `fine_<m>` run equal those of `<m>` at all four node counts.
  So `fine_` vs. `<m>` differences are not reconstruction bugs, which §5.7 says must be
  settled before H17 can be read. `fine_` still sends the same *count* of elements to
  each partner, not the same x entries, so its memory access pattern differs from the
  matrix's.
- `coarse_` matched the matrices' mean bytes per rank within about 0-5% in a 2-node
  test. This has not been checked at 4-16 nodes; check it. Its partner structure is
  drawn, not recorded, so it differs per rank by design.
- **Known timing anomaly.** Runs lose whole multiples of about 0.160 s inside the
  timed loop, mostly on the MPI and STANDARD arms, rarely on LOCALITY.
  - The longest ones in this campaign run 15 minutes or more: six
    `abcgcgnp_crankseg_1` runs at 8 nodes (39.7-77.3 min, as MPI+STANDARD pairs in
    launches 8013844, 8013850, and 8013852) and one `fine_nd24k` MPI run at 16 nodes
    (24.4 min, launch 8013891).
  - Far more runs are 5-35x their cell median: 375 by that rule. They are mostly
    crankseg, and also nd12k and nd24k at 4-16 nodes. They occur in the replay
    benchmark (`<m>`, `fine_`) as well as in ABCG, and almost never in `coarse_`.
    On crankseg at 4 nodes, the stalls inflate even the cell medians.
  - In the earlier one-launch check: `abcgcg_crankseg_2` at 4 nodes.
  - In an older campaign, single launches of crankseg MPI/STANDARD ranged from 0.01x
    to 5.7x their median.

  The cause is unknown. Do not silently drop these runs: report them, use robust
  statistics, and show any result with and without them. `results/` (analysis 0)
  documents the 0.160 s quantum.
- The benchmark's MPI Advance arms run through Cabana. Its LOCALITY times were
  1.6-5x slower than ABCG's own LOCALITY exchange on the same halo in earlier tests,
  so `<m>` vs. `abcgspmv` on LOCALITY compares code paths as well as patterns.

## 5. The analyses to run

The unit of replication is the **launch** (allocation): 40 per node count. Runs
inside one launch share an allocation and are paired; they are not independent
samples. Bootstrap by resampling launches (the block bootstrap of §5.4). Use log
ratios for comparisons, because arms and matrices differ by 10x or more in scale and
the distributions are skewed.

In priority order:

1. **H17: does replay reproduce the application's arm ordering?**
   - For each node count x matrix x launch, rank the three arms by time per
     iteration under each kind.
   - Compare each stand-in's ordering (`<m>`, `fine_`, `coarse_`) with the
     application's (`abcgspmv`, and separately `abcgcg`).
   - Report agreement rates with Wilson intervals across launches.
   - Report **worst-case regret**: the application's time on the stand-in's chosen
     arm, divided by the application's best arm, minus 1. Give the median and the
     maximum over launches.
   - Report agreement by position (matrix, node count), never as a single number.
   - Test the plan's pre-stated prediction: ordering holds for MPI vs. STANDARD
     (non-aggregating) and degrades where LOCALITY (aggregating) is involved.
2. **H18: the fidelity ladder.** Does `coarse_` (aggregate manager) pick the same arm
   as `fine_` (hierarchical manager) and as the application? Where they disagree, use
   the per-rank structure metrics (messages, bytes, busiest rank, on/off-node,
   in/out-NUMA) to say what the aggregate dropped.
3. **Paired fidelity ratios.** Compute log(stand-in time / reference time) per launch
   for:
   - `fine_` vs. `<m>`
   - `coarse_` vs. `<m>`
   - `<m>` vs. `abcgspmv`
   - `abcgspmv` vs. `abcgspmvnp`
   - `abcgcg` vs. `abcgcgnp`

   Report the median as "x times", with a launch-bootstrap BCa 95% interval. Run an
   equivalence test (TOST) at ±10%, stating the margin before looking.
4. **H0: does the argmin change?** Using the application kinds, does the best arm
   differ across matrices at fixed node count, and across node counts at fixed
   matrix? Compare per-workload selection against the best single fixed arm, with
   intervals.
5. **H13: median-best vs. spread-best.** Per matrix x node count x kind, which arm
   has the lowest median across launches, and which the lowest spread (robust: MAD or
   p90/p10 of log time)? Where they differ, report it with intervals on both. Spread
   here is between-allocation spread (`d_alloc`), not per-iteration spread, which
   this data does not have.
6. **H8: setup amortization.** Fit cost(N) = setup + q·N per arm, where:
   - setup is `haloTime` + `gatherTime` + (`startupWindow` − k·q) for the benchmark,
     or `setupTime` + (`startupWindow` − k·q) for ABCG;
   - q is time per iteration.

   Report the break-even N* at which LOCALITY overtakes MPI/STANDARD, with intervals.
   The 2023 paper found 22 and 40 iterations.
7. **Order and drift effects.** Regress log time on position in the launch (`run`)
   and on the previous run's kind, with launch as a random effect (§5.4 requires
   reporting the position coefficient). Check whether the slow crankseg runs follow
   particular predecessors or particular node sets (`provenance_*.txt` has node lists).
8. **Variance decomposition**, if time allows: log(time) ~ kind x arm x matrix +
   (1 | launch), per node count. How much spread is between launches vs. within?

Existing scripts you may reuse or ignore: `compare_fiveway.py` in this folder (per-launch
fidelity table for a single launch), and the scripts on the `analysis` branch of
https://github.com/COMPASS-PSAAP/irregular-benchmarking (`analysis/python.py` for the N*
and bounds method, `qq.py`, `independence.py`, `compare_patterns.py`). Read them before
reusing; they were written for earlier campaign layouts.

## 6. What this data cannot show (state these in the write-up)

- **No per-iteration times.** Each run reports only totals over n iterations. So
  there is no p99 or p99.9 within a run, no GEV fits (H15), and no per-iteration
  dispersion. All spread is across ranks (min/max/avg) or across launches.
- **The benchmark is barrier-closed and has no computation.** No claim about
  accumulation across iterations or overlap can come from the `<m>`/`fine_`/`coarse_`
  kinds (§5.7).
- **One machine and one software stack:** one MPI implementation, one MPI Advance
  version, a shared queue, and launches over about one day rather than weeks apart.
  §5.4 asks for allocations spread over calendar time, so the between-allocation
  spread here likely underestimates `d_alloc`.
- **Two of the plan's arms are missing.** It runs three arms, not the plan's five,
  and no one-sided arm. The combined Vernier manager is not tested, only hierarchical
  (`fine_`) and aggregate (`coarse_`).
- **Interleaving.** Arms run as whole runs in random order within a launch, not
  interleaved per iteration, so this is the plan's dispersion design, not its
  location design.

## 7. Output rules

- **One graph per image.** Every PNG contains exactly one plot (a single set of
  axes). Never use subplots, facets, small multiples, or grids of panels. If you want
  to show something per node count or per matrix, make a separate image for each.
- Name files descriptively, e.g. `regret_fine_vs_abcgcg_8nodes.png`.
- Label axes with quantity and unit, put the sample size (launches) in the title or a
  caption line, and draw intervals wherever there is an estimate.
- Use log scales for times and ratios spanning more than about 3x.
- Write everything to `results/`: figures, CSV tables of every number in the
  write-up, and a `SUMMARY.md` that ranks findings by importance. For every claim,
  `SUMMARY.md` states the n, the interval, and which hypothesis it bears on, and it
  lists what you could not determine.
- Keep analysis code in `analysis/` in this folder so every figure can be
  regenerated.
- Python: use the project venv
  (`C:\Users\Nicholas\PycharmProjects\PythonProject3\.venv`).
