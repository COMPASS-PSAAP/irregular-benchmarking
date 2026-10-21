"""Analysis 9 (exploratory, not pre-stated): H17 decision regret scored on each arm's
10th-percentile launch instead of its median.

Motivation, found in analysis 0: MPI/STANDARD runs of the exact-pattern kinds lose
whole multiples of 0.160 s, which inflates their medians on crankseg (and nd12k, nd24k
at 8 nodes) by up to two orders of magnitude, while coarse_ almost never does. Scoring
the p10 launch asks whether the stand-ins rank the arms the way the application does
when both are close to stall-free; comparing it with the median result separates
"the stand-in misses aggregation" from "the stand-in misses the stalls". p10 of 40
launches lies between the 4th- and 5th-lowest values: a low quantile, not a clean
stall-free time.

Writes tables/h17_p10.csv and prints the design points whose choice differs between
the median and p10 scoring.
"""
import itertools

import numpy as np
import pandas as pd

from common import ARMS, B, MATRICES, NODES, STAND_INS, launch_resamples, load_runs, percentile_interval, q_table, write_table

REFS = ["abcgspmv", "abcgcg"]


def main():
    runs = load_runs()
    rows = []
    for stand_in, reference in list(itertools.product(STAND_INS, REFS)) + [("coarse", "fine")]:
        for nodes in NODES:
            for matrix in MATRICES:
                T = q_table(runs, nodes, matrix, [stand_in, reference])
                S, R = T[stand_in][ARMS].to_numpy(), T[reference][ARMS].to_numpy()
                ps, pr = np.quantile(S, 0.1, axis=0), np.quantile(R, 0.1, axis=0)
                choice = int(ps.argmin())
                idx = launch_resamples(len(S))
                bs, br = np.quantile(S[idx], 0.1, axis=1), np.quantile(R[idx], 0.1, axis=1)
                bc = bs.argmin(axis=1)
                breg = br[np.arange(B), bc] / br.min(axis=1) - 1
                lo, hi = percentile_interval(breg)
                rows.append({"stand_in": stand_in, "reference": reference, "nodes": nodes, "matrix": matrix,
                             "launches": len(S), "p10_choice": ARMS[choice], "p10_reference_best": ARMS[int(pr.argmin())],
                             "p10_regret": pr[choice] / pr.min() - 1, "p10_regret_lo": lo, "p10_regret_hi": hi,
                             "p_choice_changes": float(np.mean(bc != choice)),
                             **{f"p10_q_ms_{s}_{a}": v for s, arr in (("stand_in", ps), ("reference", pr))
                                for a, v in zip(ARMS, arr)}})
    table = pd.DataFrame(rows)
    write_table(table, "h17_p10.csv")
    h17 = pd.read_csv(__import__("common").TABLES / "h17_design_points.csv")
    h17 = h17[h17.slow_runs == "kept"][["stand_in", "reference", "nodes", "matrix", "stand_in_choice",
                                        "reference_best", "decision_regret"]]
    h18 = pd.read_csv(__import__("common").TABLES / "h18_coarse_vs_fine.csv")
    h18 = h18[h18.slow_runs == "kept"][["stand_in", "reference", "nodes", "matrix", "stand_in_choice",
                                        "reference_best", "decision_regret"]]
    both = table.merge(pd.concat([h17, h18]), on=["stand_in", "reference", "nodes", "matrix"])
    print(both.groupby(["stand_in", "reference"]).agg(
        worst_median_regret=("decision_regret", "max"), worst_p10_regret=("p10_regret", "max"),
        points_regret_gt5pct_median=("decision_regret", lambda s: int((s > 0.05).sum())),
        points_regret_gt5pct_p10=("p10_regret", lambda s: int((s > 0.05).sum()))).round(3).to_string())
    diff = both[(both.decision_regret > 0.05) | (both.p10_regret > 0.05)]
    print(diff[["stand_in", "reference", "nodes", "matrix", "stand_in_choice", "reference_best", "decision_regret",
                "p10_choice", "p10_reference_best", "p10_regret", "p10_regret_lo", "p10_regret_hi"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
