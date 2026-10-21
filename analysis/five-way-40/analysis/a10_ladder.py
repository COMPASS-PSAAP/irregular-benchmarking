"""Analysis 10: the stand-in ladder, scored against the true run.

The five kinds form a chain, each one step further from the run we care about:
  ABCG CG (abcgcg, the true run) <- ABCG SpMV exchange (abcgspmv) <- the matrix replay
  (pm) <- the fine Vernier replay (fine) <- the coarse Vernier replay (coarse).
For each stand-in this scores (1) its arm choice on ABCG CG and (2) its arm choice on
the next step up, with a1_h17.ordering_stats (decision regret from medians over
launches, launch-bootstrap interval), and gives per-arm time ratios to CG and to the
next step (median of per-launch ratios, BCa 95%). Persistent ABCG kinds only.

Writes tables/ladder_regret.csv, tables/ladder_ratio.csv, and figures
ladder_worst_regret_vs_cg.png, ladder_regret_vs_cg.png, ladder_time_ratio_vs_cg.png.
"""
import numpy as np
import pandas as pd

from a1_h17 import ordering_stats
from common import (ARM_COLOR, ARMS, HUMAN, MATRICES, NODES, bca, load_runs, new_fig, q_table, save_fig,
                    write_table)

CHAIN = ["coarse", "fine", "pm", "abcgspmv", "abcgcg"]
NEXT = dict(zip(CHAIN[:-1], CHAIN[1:]))
SHORT = {"coarse": "coarse replay", "fine": "fine replay", "pm": ".pm replay", "abcgspmv": "ABCG SpMV",
         "abcgcg": "ABCG CG"}
COLOR = {"coarse": "#d62728", "fine": "#ff7f0e", "pm": "#2ca02c", "abcgspmv": "#1f77b4"}
CRITERION = 0.10


def main():
    runs = load_runs()
    regret, ratio = [], []
    for stand_in in CHAIN[:-1]:
        for reference, relation in (("abcgcg", "vs_cg"), (NEXT[stand_in], "vs_next")):
            if relation == "vs_next" and reference == "abcgcg":
                continue                                  # abcgspmv's next step is CG itself
            for nodes in NODES:
                for matrix in MATRICES:
                    for drop in (False, True):
                        T = q_table(runs, nodes, matrix, [stand_in, reference], drop_slow=drop)
                        S, R = T[stand_in][ARMS].to_numpy(), T[reference][ARMS].to_numpy()
                        key = {"stand_in": stand_in, "reference": reference, "relation": relation,
                               "nodes": nodes, "matrix": matrix, "slow_runs": "dropped" if drop else "kept"}
                        regret.append({**key, **ordering_stats(S, R)})
                        if drop:
                            continue
                        for a, arm in enumerate(ARMS):
                            lr = np.log(S[:, a] / R[:, a])
                            lo, hi = bca(lr)
                            ratio.append({**key, "arm": arm, "launches": len(lr), "median_ratio": np.exp(np.median(lr)),
                                          "lo95": np.exp(lo), "hi95": np.exp(hi)})
    regret, ratio = pd.DataFrame(regret), pd.DataFrame(ratio)
    write_table(regret, "ladder_regret.csv")
    write_table(ratio, "ladder_ratio.csv")

    kept = regret[(regret.slow_runs == "kept") & (regret.relation == "vs_cg")]
    order = CHAIN[:-1]
    worst = kept.groupby("stand_in").decision_regret.max().reindex(order)
    over = kept.groupby("stand_in").decision_regret.apply(lambda s: int((s > CRITERION).sum())).reindex(order)
    L = f"{kept.launches.min()}-{kept.launches.max()}"
    fig, ax = new_fig("Worst-case cost of each stand-in's arm choice for ABCG's CG",
                      "stand-in (left = furthest from the true run)", "worst slowdown of CG vs. its best arm (%)",
                      caption=f"For each stand-in, the largest decision regret over the 24 combinations of matrix and "
                              f"node count: how much slower ABCG's CG runs on the arm the stand-in ranks fastest (by "
                              f"median over launches) than on CG's own fastest arm. The label above each bar counts the "
                              f"combinations where that regret exceeds {CRITERION:.0%} (dashed line). {L} launches per "
                              f"combination; slow runs kept.")
    x = np.arange(len(order))
    ax.bar(x, worst * 100, color=[COLOR[s] for s in order])
    for xi, w, n in zip(x, worst * 100, over):
        ax.text(xi, max(w, 0.5) * 1.15, f"{n} of 24 over {CRITERION:.0%}", ha="center", fontsize=8)
    ax.axhline(CRITERION * 100, color="grey", ls="--", lw=0.8)
    ax.set_xticks(x, [SHORT[s] for s in order])
    ax.set_yscale("log")
    ax.set_ylim(1, worst.max() * 100 * 4)
    save_fig(fig, "ladder_worst_regret_vs_cg.png")

    points = [(n, m) for n in NODES for m in MATRICES]
    fig, ax = new_fig("Cost of letting each stand-in pick the arm for ABCG's CG", "node count / matrix",
                      "slowdown of CG vs. its best arm (%)", width=11, height=5.2,
                      caption=f"Decision regret on ABCG's CG at each combination of node count and matrix, one color "
                              f"per stand-in, with 95% launch-bootstrap intervals; 0% means the stand-in picked CG's "
                              f"fastest arm. Dashed line: the {CRITERION:.0%} usefulness criterion. {L} launches per "
                              f"point; slow runs kept.")
    for k, s in enumerate(order):
        t = kept[kept.stand_in == s].set_index(["nodes", "matrix"]).reindex(points)
        xs = np.arange(len(points)) + (k - 1.5) * 0.18
        ax.errorbar(xs, t.decision_regret * 100,
                    yerr=np.clip(np.vstack([t.decision_regret - t.decision_regret_lo,
                                            t.decision_regret_hi - t.decision_regret]) * 100, 0, None),
                    fmt="o", ms=3.5, capsize=2, color=COLOR[s], label=SHORT[s])
    ax.axhline(CRITERION * 100, color="grey", ls="--", lw=0.8)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(len(points)), [f"{n}/{m}" for n, m in points], rotation=70, fontsize=7)
    ax.legend(fontsize=8)
    save_fig(fig, "ladder_regret_vs_cg.png")

    r = ratio[ratio.relation == "vs_cg"]
    fig, ax = new_fig("Time per iteration of each stand-in relative to ABCG's CG", "stand-in",
                      "stand-in time / CG time (per iteration)",
                      caption=f"Each point is one combination of node count and matrix: the median over launches of "
                              f"the per-launch ratio of the stand-in's time per iteration to CG's, on the same arm "
                              f"(color). A CG iteration includes the SpMV exchange, the local multiply, two allreduces, "
                              f"and vector updates, so ratios below 1 are expected. {L} launches per point; slow runs "
                              f"kept.")
    for k, s in enumerate(order):
        for j, arm in enumerate(ARMS):
            v = r[(r.stand_in == s) & (r.arm == arm)].median_ratio
            jitter = np.linspace(-0.05, 0.05, len(v))
            ax.scatter(k + (j - 1) * 0.22 + jitter, v, s=12, color=ARM_COLOR[arm], label=arm if k == 0 else None)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(order)), [SHORT[s] for s in order])
    ax.legend(fontsize=8)
    save_fig(fig, "ladder_time_ratio_vs_cg.png")

    print(kept.groupby("stand_in").agg(worst=("decision_regret", "max"), over10=(
        "decision_regret", lambda s: int((s > CRITERION).sum())), argmin_agree=("argmin_agree", "median"))
        .reindex(order).round(3).to_string())
    nxt = regret[(regret.slow_runs == "kept") & (regret.relation == "vs_next")]
    print(nxt.groupby(["stand_in", "reference"]).agg(worst=("decision_regret", "max"), over10=(
        "decision_regret", lambda s: int((s > CRITERION).sum()))).round(3).to_string())
    print(r.groupby(["stand_in", "arm"]).median_ratio.agg(["min", "median", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
