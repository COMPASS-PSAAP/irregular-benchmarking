# Five-way campaign, 2026-10-13: what the data shows

Data: 20,160 runs. That is 40 complete launches at each of 2, 4, 8, and 16 nodes.
Each launch runs 6 matrices x 7 kinds x 3 arms once. The unit of replication is the
launch. Every interval below resamples launches (10,000 resamples, fixed seed) unless
stated otherwise. All times are q = `timedTotal` (max over ranks) / n, in ms per
iteration. Regret is a fraction: 0.30 means the application runs 30% slower on the
chosen arm than on its best arm. Every number here is a row in `tables/`. Every
figure is in `figures/`, one plot per image.

Updated 2026-10-21 to agree with `report/report.tex` (the stand-in ladder, audited
2026-10-15) and `paper/paper.tex` (the statistics, analyses 11-14). Readings the audit
overturned are corrected in place: the MPI-vs-STANDARD disagreement is a systematic bias,
not stall noise (point 2), and H0 holds only at 4 and 8 nodes (point 4). The cause of
the 0.160 s stalls was found after the campaign (point 1). Where this summary and the two
documents differ, the documents are right.

Findings are ranked by how much they change what the data can be used for.

## 1. Most MPI/STANDARD variation is whole 0.160 s stalls, and they decide many orderings

Not a pre-stated analysis: found while auditing. Sources: `tables/stall_quantum.csv`,
`tables/slow_runs.csv`, `figures/stall_quantum_*.png`, `figures/stall_steps_*.png`.

- **What it is.** In 61 of the 447 (nodes, pattern, arm) cells whose `timedTotal`
  spans at least 0.3 s across launches, the run-to-run differences are integer
  multiples of one constant. The best period is a median of 0.1600 s (IQR
  0.1600-0.1605), with phase coherence up to 0.995 against a typical 0.05-0.15. 22 of
  the 61 cells are of the non-persistent ABCG kinds. In the cleanest cell, 16-node
  `fine_crankseg_2` STANDARD, some launches lose 56 stalls in 57 timed iterations. At
  4-node crankseg_2, `<m>`'s MPI and STANDARD runs lose up to 432 stalls in 440
  iterations, and its LOCALITY runs none.
- **Where it happens.** It hits MPI and STANDARD runs. Over all kinds, 375 runs are
  over 5x their cell median, and 12 of them are LOCALITY, all non-persistent ABCG.
  Among the kinds the report studies (`<m>`, `fine_`, `coarse_`, persistent ABCG SpMV
  and CG), 266 of 9,600 MPI and STANDARD runs are flagged: `<m>` 76, `fine_` 98,
  persistent ABCG 90, and **`coarse_` 2**; no LOCALITY run of these kinds is flagged.
  They fall on crankseg_1 and crankseg_2 at every node count, nd24k at 4-16 nodes, and
  nd12k at 8 and 16. Affected launches: 6 of 40 (2 nodes), 20 of 40 (4), 17 of 40 (8),
  and 18 of 40 (16).
- **Why it matters.** The pilot sized n on a STANDARD run, and on crankseg the pilot
  was itself stalled, so n is small: 57 timed iterations for `fine_crankseg_2` at 16
  nodes (pilot 142.8 ms per iteration), where the fastest MPI run takes 0.41 ms per
  iteration and each stall adds 2.8 ms. The pilot's time per iteration was over 2x the
  fastest later STANDARD run's in 47 of 168 (node count, pattern) pairs. On 4-node
  crankseg_2, the median MPI/STANDARD q of `<m>`, `fine_`, and `abcgspmv` is 4-23x its
  fastest launch. In those cells the median is the stall count.
- **The pre-stated 5x flag** does not remove this where stalls inflate the median
  itself. Every "slow runs dropped" result below is therefore only a partial check.
  Analysis 9 (point 3) and the stall-removed medians (point 9) give the other views.
- **What does not predict a stall.** Launch start time (Spearman |rho| <= 0.12,
  p >= 0.45 at every node count), the previous run's kind, and individual nodes (124
  nodes tested; smallest Bonferroni p = 0.73).
- **Cause, found after the campaign (not from these tables).** The stall is libpsm2's
  error-check timer, `PSM2_ERRCHK_TIMEOUT` (default 160:640:2 ms). Open MPI's default
  transport on Dane (pml cm with mtl psm2, as found in the 2026-10-19 transport test)
  and MVAPICH2 both use libpsm2. In job 8065849 (2026-10-19, 4 nodes, `<m>` crankseg_2
  MPI and MVAPICH2 ABCG SpMV) the stall length followed the setting exactly: 160, 20,
  and 5 ms. Per-iteration timing shows single iterations of 160 ms inside the exchange,
  in bursts, on the ranks with the most off-node senders. Whether a packet is actually
  lost, or the timer fires with nothing lost, is not yet known. Details:
  `~/Claude/10-19-2026.txt` and `10-20-2026.txt`.

## 2. H17: which stand-in chooses CG's arm; every LOCALITY miss is a missed LOCALITY win

Sources: `tables/ladder_regret.csv`, `tables/h17_design_points.csv`,
`tables/h17_pairwise.csv`, `figures/ladder_*`, `figures/regret_*`,
`figures/worst_case_regret.png`.

Decision regret: the stand-in's lowest-median arm, scored on the reference's medians.
There are 24 design points (6 matrices x 4 node counts) with 40 launches each. Slow
runs are kept. The ladder scores each stand-in against ABCG CG (the true run) and
against the next step up: CG <- ABCG SpMV <- `<m>` <- `fine_` <- `coarse_`.

| Stand-in | Scored on | Worst case at | Worst-case regret (95% CI) | Points over 10% |
|---|---|---|---|---|
| ABCG SpMV | CG | 2 nodes, x104 | 0.042 (0.031-0.051) | 0 of 24 |
| `<m>` | CG | 4 nodes, nd12k | 0.21 (0.19-0.26) | 1 of 24 |
| `fine_` | CG | 2 nodes, crankseg_1 | 1.01 (0.00-1.33) | 2 of 24 |
| `coarse_` | CG | 4 nodes, crankseg_2 | 31.3 (17.4-73.6) | 6 of 24 |
| `<m>` | ABCG SpMV | 4 nodes, nd12k | 0.30 (0.27-0.33) | 1 of 24 |
| `fine_` | `<m>` | 2 nodes, crankseg_1 | 0.014 (0.00-0.35) | 0 of 24 |
| `coarse_` | `fine_` | 4 nodes, crankseg_2 | 17.2 (15.3-45.6) | 4 of 24 |

- **Only ABCG's SpMV exchange chooses within 10% of CG's best everywhere.** Dropping the
  5x-flagged runs changes none of the over-10% counts; `coarse_`'s worst case drops to
  18.5. The 10% usefulness line was chosen after the regrets were seen; only the 10%
  TOST margin (point 5) was fixed in advance.
- **Every choice that costs CG more than 5% (12 of 12) picks STANDARD where CG's best
  is LOCALITY**: 7 by `coarse_`, 3 by `fine_`, 2 by `<m>`. No stand-in picks LOCALITY
  where CG is fastest without it. This is the half of the plan's prediction that held
  (§6.3). Its stated reason, a missing rank-to-region mapping, does not explain `<m>`'s
  misses, because `<m>` sends the exact pattern.
- **`<m>` misses exactly where CG's LOCALITY gain is smallest**: 1.016x, 1.071x, and
  1.213x at 16-node x104, 16-node m_t1, and 4-node nd12k. It picks LOCALITY at all 13
  points where CG's gain is 2.0x or more, and no point falls between. At 4-node nd12k,
  `<m>`'s LOCALITY takes 1.58x ABCG's while its MPI and STANDARD match ABCG's within
  3.1%: the Cabana LOCALITY path (point 5) overturns a 21% margin. On crankseg, part of
  `<m>`'s correct choice rests on its own stalls: at 4-node crankseg_2 its fastest
  STANDARD launch (0.260 ms) beats its fastest LOCALITY launch (0.365 ms).
- **`fine_`'s worst case is a near-tie.** It differs from `<m>` only at 2-node
  crankseg_1, where `<m>`'s STANDARD and LOCALITY medians are within 1.4%; `<m>`'s
  choice flips in 43% of bootstrap resamples and `fine_`'s in 23%.
- **MPI vs. STANDARD is a systematic bias, not stall noise.** CG is fastest on MPI at 8
  points (2-node nd12k, x104, m_t1, nd24k; 4- and 8-node x104 and m_t1), and at all 8
  every stand-in, ABCG SpMV included, picks STANDARD. There `<m>` puts STANDARD ahead in
  90-100% of launches, `fine_` in 82-100%, ABCG SpMV in 50-88%, and CG in 0-60%. None of
  the 8 points has a slow run. It costs CG 0.1-4.2%. Cause unknown (point 10 gives two
  candidate mechanisms). The 10-14 reading, that the MPI-STANDARD disagreement came
  from stalls, was wrong.
- The original H17 table against ABCG SpMV and CG (analysis 1, `h17_design_points.csv`)
  is unchanged; its worst cases are `<m>` 0.30 and 0.21, `fine_` 1.09 and 1.01,
  `coarse_` 23.0 and 31.3 (abcgspmv and abcgcg).

## 3. H18: the aggregate capture is not enough at 2-4 nodes; its structural distortions do not explain why

Sources: `tables/h18_coarse_vs_fine.csv`, `tables/h18_structure.csv`,
`tables/h18_disagreements.csv`, `tables/h17_p10.csv`,
`figures/regret_coarse_vs_fine_*`, `figures/coarse_structure_ratio_*`.

- **`coarse_` vs. `fine_`.** They agree on the arm at every 8- and 16-node point (12
  of 12). They disagree at 4 of 12 points at 2-4 nodes: crankseg_2 at 2 nodes, and
  crankseg_2, crankseg_1, and nd24k at 4 nodes. At 4-node crankseg_2, CG takes 32.3x as
  long per iteration on `coarse_`'s choice (STANDARD) as on LOCALITY. `coarse_`'s other
  two misses over 10% on CG (2-node crankseg_1, 4-node nd12k) are shared with `fine_`.
- **Not a stall artifact at 4 nodes:** CG's fastest STANDARD launch at 4-node
  crankseg_2 (0.951 ms) is still 3.3x its fastest LOCALITY launch (0.288 ms).
- **Coarse mean bytes per rank:** within 0.7% below to 3.1% above `<m>` at every node
  count and matrix. Mean messages per rank are within 1%.
- **What the aggregate changes**, at every node count (median over launches, coarse /
  exact):
  - same-NUMA bytes 0.54-0.69x on crankseg and nd24k;
  - on-node bytes 0.73-0.91x on crankseg and nd24k;
  - off-node bytes 1.05-1.83x on crankseg and nd24k;
  - the busiest rank's bytes 1.17-2.06x on every matrix (up, not down).
  These distortions are as large at 8-16 nodes, where coarse agrees, so they do not
  explain the 2-4 node disagreements by themselves.
- **`coarse_` also does not stall** (point 1). Its residual run-to-run spread is about
  5% at 2-4 nodes, against 39-90% for the exact-pattern kinds (`tables/variance.csv`).
- **Fastest-10% view (analysis 9, not pre-stated).** Scoring each arm on its
  10th-percentile launch, the 2-node misses of `coarse_` and `fine_` on CG shrink to at
  most 2.6%: there CG's least-stalled best arm is MPI, not LOCALITY. The 4-node misses of
  `coarse_` shrink but remain.
- **Reading.** At 2 nodes, the aggregate capture "fails" because the application's
  LOCALITY advantage there comes from avoiding stalls, which the aggregate pattern does
  not trigger. At 4 nodes something beyond the stalls remains; this data cannot say
  what. Spy plots of generated matrices (`data-10-15-2026-vernier-to-pm`) show that
  coarse partners spread evenly over all ranks, where the real halo has long-range
  bands and blocks.
- **H18 as written cannot be fully tested.** It names the *combined* manager, and
  this campaign has only hierarchical (`fine_`) and aggregate (`coarse_`).

## 4. H0: the application's best arm changes across matrices at 4 and 8 nodes

Sources: `tables/h0_best_arm.csv`, `tables/h0_selection_gain.csv`, `tables/h17_p10.csv`,
`figures/h0_best_arm_*`, `figures/h0_selection_gain_*`.

- **Best arm by lowest median.**
  - crankseg_1 and crankseg_2: LOCALITY at every node count.
  - nd12k and nd24k: LOCALITY from 4 nodes.
  - m_t1 and x104: LOCALITY only at 16 nodes.
  - Everywhere else, STANDARD for abcgspmv and MPI for abcgcg.
- **Per-matrix selection vs. the best single fixed arm** (cross-fitted: chosen on half
  the launches, scored on the other half, 2,000 bootstrap splits). Gain is the
  geometric mean of q(fixed) / q(selected) - 1:

| Nodes | abcgspmv gain (95%) | abcgcg gain (95%) | Best fixed arm |
|---|---|---|---|
| 2 | 0.29 (0.08-0.52) | 0.28 (0.09-0.47) | STANDARD or MPI |
| 4 | 0.44 (0.435-0.446) | 0.29 (0.290-0.298) | LOCALITY |
| 8 | 0.15 (0.14-0.16) | 0.11 (0.10-0.12) | LOCALITY |
| 16 | 0.00 (0.00-0.00) | 0.00 (-0.01-0.00) | LOCALITY |

- **H0 holds at 4 and 8 nodes only.** At 16 nodes LOCALITY wins all six matrices. At 2
  nodes, on each arm's fastest 10% of launches CG's best arm is MPI or STANDARD for
  every matrix, so the 2-node gain comes from stalls, not from the pattern. At 4 and 8
  nodes the fastest-10% best arm is unchanged at every point. The 10-14 reading (H0
  holds at 2-8 nodes) was wrong at 2 nodes.

## 5. Paired fidelity ratios

Source: `tables/ratios.csv`, `figures/ratio_*`. Median per-launch ratio with BCa 95%.
The TOST margin of +/-10% was fixed before computing, using a 90% BCa interval.

- **`fine_` / `<m>`.** Equivalent on LOCALITY at all 24 points (0.993-1.008x). On MPI
  and STANDARD, 15 of 24 are equivalent each; the other 18 (0.86-1.42x) are not shown
  equivalent, and 13 of them contain a slow run.
- **`<m>` / abcgspmv on LOCALITY: 1.49-1.99x, never equivalent (0 of 24).** Same
  messages and bytes, so this is the code path: the replay drives MPI Advance's
  locality-aware exchange through Cabana's gather. On MPI and STANDARD the ratio is
  0.95-1.14 on m_t1 and x104 but 0.95-3.79 on the crankseg matrices, nd12k, and nd24k
  (equivalent at 10 and 11 of 24 overall).
- **Persistent / non-persistent ABCG on LOCALITY:** 0.019-3.2x (abcgspmv) and
  0.024-2.3x (abcgcg); equivalent at 1 of 24. They are different regimes, not one arm
  run two ways. On MPI and STANDARD, 13-15 of 24 are equivalent.
- **`coarse_` / `fine_`:** 0.59-1.17x on LOCALITY and 0.018-1.39x on MPI/STANDARD; the
  low end is where `fine_` stalls and `coarse_` does not.
- **To CG:** ABCG SpMV takes 0.62-0.75 of a CG iteration on LOCALITY and 0.39-0.97 on
  MPI/STANDARD. No stand-in's time is a fixed multiple of CG's (`<m>` MPI 0.44-3.26x,
  `coarse_` MPI 0.035-0.80x; `tables/ladder_ratio.csv`), so stand-ins can serve only
  for choosing an arm, not for estimating CG's time.

## 6. H13: on the simpler matrices, the fastest-median arm is not the steadiest one

Source: `tables/h13.csv`, `figures/h13_median_vs_spread_*`.

- In 36 of 168 (kind, node count, matrix) cells, the median-best and spread-best arms
  differ with bootstrap probability >= 0.9 for both. Spread is the MAD of ln q across
  launches. With p90/p10 as the spread measure, the count is 41.
- **Every one of the 36 has the same form:** STANDARD or MPI has the lowest median, and
  LOCALITY has the lowest spread. They fall mostly on nd12k, x104, m_t1, and nd24k at 2-8
  nodes. `coarse_` adds 4-node crankseg_1 and crankseg_2, 8-node x104, and 16-node x104 and m_t1.
- They include the application:
  - abcgspmv: 2-node nd12k, x104, and m_t1, and 4-node x104;
  - abcgcg: 2-node x104 and nd24k, and 4-node x104 and m_t1.
- The count is the same with slow runs dropped. In the 2-node abcgspmv cells, LOCALITY's MAD
  of ln q is about 0.01-0.02 against about 0.03-0.05 for MPI/STANDARD.
- This supports H13 here. It is also the reverse of the plan's ABCG anecdote, which
  had the variance reduction only on *complex* patterns. Stated limit: this is spread
  across allocations about one day apart in a shared queue (`d_alloc`-like), not
  per-iteration or per-solve dispersion, which this data does not have.

## 7. H8: LOCALITY's break-even reuse count

Source: `tables/h8.csv`, `figures/h8_nstar_*`. N* is computed from medians over
launches, with a 95% launch-bootstrap interval, LOCALITY vs. MPI. Setup is
(`setupTime` or `haloTime` + `gatherTime`) + `startupWindow` - k*q. The untimed
`-W nearest` warmup is outside it, so these N* are lower bounds on a cold start.

| | crankseg_2 | crankseg_1 | nd24k | nd12k | m_t1, x104 |
|---|---|---|---|---|---|
| abcgspmv, 2 nodes | 95 (58-246) | 161 (86-908) | never | never | never |
| abcgspmv, 4 nodes | 23 (13-33) | 40 (24-54) | 185 (126-259) | 1,178 (1,032-1,257) | never |
| abcgspmv, 8 nodes | 35 (10-42) | 21 (4-37) | 93 (73-116) | 130 (93-159) | never |
| abcgspmv, 16 nodes | 9 (3-15) | 22 (10-31) | 68 (38-89) | 570 (433-643) | 4,340 / 4,625 |

- On persistent ABCG, the crankseg break-even at 4-16 nodes (9-40 iterations) is the
  same order as the 2023 paper's 22 and 40.
- **abcgcg (the true run):** 7-31 iterations on crankseg at 4-16 nodes and 86 and 103 at
  2 nodes; 26-83 on nd24k and 70-778 on nd12k at 4-16 nodes, never at 2; m_t1 and x104
  never at 2-8 nodes and 2,449 (1,968-5,599) and 8,518 (3,389-65,722) at 16.
- On the less communication-bound matrices it is hundreds to thousands, or never.
  That fits H8's form: the gate moves with the pattern.
- The non-persistent ABCG kinds are mostly "never" or "until", because their
  LOCALITY pays setup on every call.
- `fine_` and `coarse_` are excluded. Their `haloTime` includes reading the Vernier
  files (230-960 ms on every arm).

## 8. Order effects and variance

Sources: `tables/order_model.csv`, `tables/slow_by_*.csv`, `tables/variance.csv`,
`tables/exchangeability.csv`, `figures/position_effect_by_nodes.png`,
`figures/variance_share_*`.

- **Position in launch** (mixed model with cell fixed effects, a launch random
  intercept, and the previous run's kind), as ln q per 100 positions:
  - 2 nodes: +0.018 (-0.010 to 0.045);
  - 4 nodes: -0.024 (-0.064 to 0.017);
  - **8 nodes: +0.070 (0.025 to 0.116)**, about +9% from the first run to the last,
    and +0.065 (0.029 to 0.101) with slow runs dropped;
  - 16 nodes: +0.001 (-0.035 to 0.038).
  Only 8 nodes drifts. The cause is not determined.
- **Paired comparisons do not drift.** The per-launch arm log ratio d shows no trend
  with launch start time, position in the launch, or the number of concurrent launches
  in any of the 504 comparisons (Spearman, BH q < 0.05), and a bootstrap over blocks of
  five consecutive launches gives intervals as wide as the ordinary launch bootstrap
  (median ratio 1.01). For paired comparisons the launches behave as exchangeable.
- **Previous run's kind:** no consistent effect on the stall rate (`slow_by_predecessor.csv`).
- **Variance decomposition** (ln q ~ cell + (1 | launch)): the share of variance that
  is a whole-launch shift (ICC) is 0.09-0.23 overall and 0.06-0.36 by kind. Most
  variance is per-run, which this design cannot split into repeat noise and launch x
  cell interaction (each cell runs once per launch). `coarse_` has about 5% residual
  spread at 2-4 nodes, against 39-90% for the exact-pattern kinds.

## 9. Statistics: which tests can be trusted on this data

Sources: `tables/normality.csv`, `tables/estimands.csv`, `tables/calibration_*.csv`,
`tables/sample_size.csv`, `tables/replication.csv`, `tables/verdicts.csv`,
`tables/verdict_disagreements.csv`, `tables/best_arm_tests.csv`. Analyses 11-13; the
paper's sec. 5-6. A comparison is two arms of one kind, matrix, and node count, with
d = ln(q_A / q_B) per launch: 504 comparisons, 321 with no stalled run, 101 with stalls
in at most half of either arm's launches (minority), 82 with more (majority).

- **Normality.** Shapiro-Wilk rejects at 5% in 86% of the 504 cells on q, 62% on ln q,
  and 61% of the paired d. Median skewness 1.24 on q, 0.70 on ln q. The stall mixture is
  the mechanism.
- **Estimands.** The median of d, the mean of d, and the ratio of mean times agree in
  sign in 448 of 504 comparisons; where they disagree the estimates sit near zero, and no
  pair of 95% intervals excludes zero on opposite sides. Removing whole stalls changes
  the median's winner in 23 of the 82 majority comparisons (at 4-node crankseg_2, `<m>`'s
  STANDARD/LOCALITY median goes from 16.6x to 0.74x).
- **False positives** (each test in a bootstrap world where its own null holds). The
  paired t-test on ln q has a median rate of 5.1-6.2% at every launch count and regime.
  The t-test on raw q reaches 10-14% at 5 launches with stalls. The sign test never
  exceeds 5%; Wilcoxon is 5.0-5.2% from 10 launches on. Run where the typical launch is a
  tie, the t-test on ln q names a winner in 10-11% of draws at 40 launches: it answers a
  question about the mean, not the typical launch.
- **Power.** Without stalls every test detects a true 10% difference with probability
  >= 0.92 from 10 launches on. With stalls, none exceeds 0.22 even at 40 launches.
- **Coverage.** The t interval for the mean of d covers 94.0-94.6% in the median
  comparison, 89-93% in the worst tenth. The order-statistic interval for the median
  covers 95.8-97.8% at 10-40 launches; BCa for the median 93.0, 93.9, 94.4% at 10, 20, 40.
- **Sample size.** The normal-theory count that chose 40 launches asks for more than 40
  in 187 of 504 cells (up to 1,111), and at its own count the t interval covers under
  93% in 172.
- **Replication.** Each test confirms its own significant verdicts on the other half of
  the launches 92.3-93.3% of the time; first 20 vs. last 20 launches never reverse a
  verdict, and 0.03% of verdicts reverse over 500 random splits.
- **Verdicts on all 40 launches** (BH within each kind): the t-test on ln q and the sign
  test agree on 472 of 504 and never reach opposite verdicts; they name a different
  best arm in 16 of 168 cells. The t-test on raw q disagrees with the sign test on 46.

## 10. Whether each stand-in reproduces the verdicts of the run above it

Sources: `tables/tracking.csv`, `figures/tracking_*`. Analysis 14; the paper's sec. 7-9.
A stand-in reaches the reference's verdict on an arm pair if the sign test gives the
same outcome on both (A faster, B faster, or none). It reproduces the ratio if the
per-launch difference of log arm ratios has a 90% order-statistic interval inside
+/-10%. 72 arm pairs per comparison (24 points x 3 pairs).

| Stand-in vs. reference | Same verdict | Ratio reproduced | Ratio shifted |
|---|---|---|---|
| ABCG SpMV vs. CG | 62 of 72 | 25 of 72 | |
| `<m>` vs. ABCG SpMV | 53 of 72 | 0 of 24 per LOCALITY pair | 16 of 24 per LOCALITY pair |
| `<m>` vs. CG | 49 of 72 | | |
| `fine_` vs. `<m>` | 64 of 72 | 38 of 72 | 2 of 72 |
| `coarse_` vs. `fine_` | 52 of 72 | 16 of 72 | 49 of 72 |

- **ABCG SpMV orders the arms like CG but does not predict by how much.** It
  reproduces CG's ratio at only 7 of 24 points for each pair involving LOCALITY, where
  its arm ratio is 0.58-1.41x CG's. Eight of its ten verdict disagreements are MPI vs.
  STANDARD at 2-4 nodes; at four (2-node nd12k and x104, 4-node x104 and m_t1) the two
  reach opposite significant verdicts. Candidate mechanisms for the MPI-vs-STANDARD
  bias, not separated by this campaign: CG overlaps the exchange with the on-process
  SpMV, which nonblocking point-to-point can progress and a collective may not; and the
  SpMV exchange ends each iteration with a barrier, CG does not.
- **`<m>`'s disagreements come from its code path**, since it sends ABCG's exact
  messages; the gap appears only on LOCALITY (point 5).
- **`coarse_` disagrees with `fine_` mostly at 2 and 4 nodes** (14 of 20 disagreements).

## What this data cannot show

- **No per-iteration times in this campaign.** No p99 or p99.9 within a run, no GEV
  fits (H15), and no per-iteration dispersion. All spread is across ranks (min/max/avg)
  or launches. (Per-iteration recording exists since 2026-10-15, after this campaign.)
- **The benchmark kinds are barrier-closed with no computation.** No claim about
  accumulation across iterations or overlap comes from `<m>`, `fine_`, or `coarse_`.
  Only `abcgcg` is solver-level (`t_iter`); everything else is `t_exchange`.
- **One machine and stack:** Dane (Omni-Path), Open MPI 4.1.2, locality_aware 752a19b,
  a shared pbatch queue. Launches span 23.7 hours, not weeks, so `d_alloc` is likely
  underestimated.
- **Three arms, not five; no one-sided arm; no combined Vernier manager.**
- **CG never converged** within ABCG's 500-iteration cap on these matrices, so CG time
  is time per fixed iteration, not time to solution.
- **The design is dispersion, not location:** arms run as whole runs in random order,
  not interleaved per iteration.

## What I could not determine

- **Whether the PSM2 timer fires on a lost packet or with nothing lost**, and why the
  aggregate pattern and persistent LOCALITY avoid it. In the 2026-10-19 stall test the
  waiting ranks were those with the most off-node senders (crankseg_2 at 4 nodes:
  Spearman +0.49), which `coarse_`'s random partners and LOCALITY's aggregation would
  both reduce; untested.
- **The cause of the MPI-vs-STANDARD bias** (point 10).
- **Whether the 4-node `coarse_` failures** come from the lost partner structure, from
  stall exposure, or from both.
- **Why only 8 nodes shows a position drift.**
- **A clean stall-free arm ordering.** The p10 view and the stall-removed medians are
  lower bounds, not stall-free times. Where most launches stall (4-node crankseg), even
  they are inflated.

## Definitions, fixed before computing, and deviations

- q = `timedTotal` max over ranks / n.
  - Correction to the plan text: across ranks, min and max differ by at most 0.74%
    (median 0.001%), not "well under 0.1%" everywhere. The barrier inside every timed
    replay and SpMV iteration produces this agreement.
- **Slow run:** q > 5x the median of its (nodes, pattern, arm) cell. Results are shown
  kept and dropped.
- **Stalled run** (analyses 11-14): a slow run, or a run whose timed total exceeds its
  cell's fastest launch by at least one whole 0.160 s step in a cell with 0.160 s
  coherence above 0.5.
- **Tie band:** +/-10%, the prompt's TOST margin.
- **Not pre-stated** (added after the audit, and labeled where used): the stall-quantum
  analysis (`a0_audit.stall_quantum`), the p10 scoring (`a9_low_quantile.py`), and the
  10% usefulness criterion for decision regret (point 2).
- PROMPT.md §4's slow-run paragraph was wrong (six runs at 60-77 min; ABCG only) and
  has been corrected.

## Regenerating

From `analysis/`: `..\..\.venv\Scripts\python.exe run_all.py`. It rewrites
`runs.csv`, `launches.csv`, `tables/`, `figures/`, and `five-way-campaign-report.pdf`
(this summary with its figures) from `data/campaign/`, running analyses a0-a14 and then
make_pdf. The PDF needs reportlab in the venv. The last full run's log is `run_all.log`.
