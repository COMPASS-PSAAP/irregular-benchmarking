"""Analysis 5 (H13): is the arm with the lowest median the arm with the lowest spread?

Per node count x matrix x kind: for each arm, the median of log q across launches and
two robust spreads of log q across launches, MAD (scaled by 1.4826) and log(p90/p10).
This spread is between-allocation (d_alloc-like), over launches about one day apart;
the data has no per-iteration times. Launch bootstrap gives intervals on each arm's
median and spread and the probability that each arm is median-best and spread-best.
"differ" = the point estimates pick different arms; "differ_confident" = each pick
has bootstrap probability >= 0.9. Slow runs kept and dropped.

Writes tables/h13.csv and figures h13_median_vs_spread_<kind>_<N>nodes.png for the
application kinds abcgspmv and abcgcg.
"""
import numpy as np
import pandas as pd

from common import (HUMAN, ARM_COLOR, ARMS, B, KIND_LABEL, MATRICES, NODES, launch_resamples, load_runs, new_fig,
                    percentile_interval, q_table, save_fig, write_table)

KINDS = ["pm", "fine", "coarse", "abcgspmv", "abcgspmvnp", "abcgcg", "abcgcgnp"]


def mad(x, axis):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def p90p10(x, axis):
    return np.quantile(x, 0.9, axis=axis) - np.quantile(x, 0.1, axis=axis)


def main():
    runs = load_runs()
    rows = []
    for kind in KINDS:
        for nodes in NODES:
            for matrix in MATRICES:
                for drop in (False, True):
                    T = q_table(runs, nodes, matrix, [kind], drop_slow=drop)
                    X = np.log(T[kind][ARMS].to_numpy())
                    idx = launch_resamples(len(X))
                    Xb = X[idx]                                           # (B, L, 3)
                    med, bmed = np.median(X, axis=0), np.median(Xb, axis=1)
                    row = {"kind": kind, "nodes": nodes, "matrix": matrix,
                           "slow_runs": "dropped" if drop else "kept", "launches": len(X)}
                    for name, f in (("mad", mad), ("p90p10", p90p10)):
                        s, bs = f(X, 0), f(Xb, 1)
                        row[f"{name}_best"] = ARMS[int(s.argmin())]
                        pbest = np.bincount(bs.argmin(axis=1), minlength=3) / B
                        for a, arm in enumerate(ARMS):
                            row[f"{name}_{arm}"] = s[a]
                            row[f"{name}_{arm}_lo"], row[f"{name}_{arm}_hi"] = percentile_interval(bs[:, a])
                            row[f"p_{name}_best_{arm}"] = pbest[a]
                    row["median_best"] = ARMS[int(med.argmin())]
                    pmed = np.bincount(bmed.argmin(axis=1), minlength=3) / B
                    for a, arm in enumerate(ARMS):
                        row[f"median_q_ms_{arm}"] = np.exp(med[a])
                        lo, hi = percentile_interval(bmed[:, a])
                        row[f"median_q_ms_{arm}_lo"], row[f"median_q_ms_{arm}_hi"] = np.exp(lo), np.exp(hi)
                        row[f"p_median_best_{arm}"] = pmed[a]
                    for name in ("mad", "p90p10"):
                        row[f"differ_{name}"] = row["median_best"] != row[f"{name}_best"]
                        row[f"differ_{name}_confident"] = bool(
                            row[f"differ_{name}"] and row[f"p_median_best_{row['median_best']}"] >= 0.9
                            and row[f"p_{name}_best_{row[f'{name}_best']}"] >= 0.9)
                    rows.append(row)
    table = pd.DataFrame(rows)
    write_table(table, "h13.csv")

    kept = table[table.slow_runs == "kept"]
    for kind in ["abcgspmv", "abcgcg"]:
        for nodes in NODES:
            t = kept[(kept.kind == kind) & (kept.nodes == nodes)].set_index("matrix").reindex(MATRICES)
            fig, ax = new_fig(f"Fastest arm vs. steadiest arm, {HUMAN[kind]}, {nodes} nodes",
                              "median time per iteration across launches (ms)",
                              "variation between launches (MAD of ln time)", height=5.6,
                              caption=f"Each marker is one arm on one matrix (color = arm, shape = matrix). Further "
                                      f"right is slower; higher means more variation from launch to launch (0.1 is "
                                      f"about 10%). Bars are 95% launch-bootstrap intervals. Where a matrix's "
                                      f"left-most and lowest markers are different colors, the fastest arm is not the "
                                      f"steadiest. Launches per matrix: {', '.join(str(v) for v in t.launches)}; slow "
                                      f"runs kept.")
            for k, matrix in enumerate(MATRICES):
                r = t.loc[matrix]
                for arm in ARMS:
                    m, s = r[f"median_q_ms_{arm}"], r[f"mad_{arm}"]
                    ax.errorbar(m, s, xerr=[[m - r[f"median_q_ms_{arm}_lo"]], [r[f"median_q_ms_{arm}_hi"] - m]],
                                yerr=[[s - r[f"mad_{arm}_lo"]], [r[f"mad_{arm}_hi"] - s]],
                                fmt="osD^vP"[k], ms=5, capsize=2, color=ARM_COLOR[arm],
                                label=f"{matrix} {arm}" if arm == "MPI" else None)
            handles = [plt_line(ARM_COLOR[a], "o", a) for a in ARMS] + \
                      [plt_line("black", "osD^vP"[k], m) for k, m in enumerate(MATRICES)]
            ax.legend(handles=handles, fontsize=7, ncol=3)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylim(bottom=0)
            save_fig(fig, f"h13_median_vs_spread_{kind}_{nodes}nodes.png")

    cols = ["kind", "nodes", "matrix", "median_best", "mad_best", "p90p10_best", "differ_mad_confident",
            "differ_p90p10_confident"]
    print(kept[kept.differ_mad | kept.differ_p90p10][cols].to_string(index=False))
    print("\nconfident differences (kept / dropped):",
          int(kept.differ_mad_confident.sum()), "/",
          int(table[table.slow_runs == "dropped"].differ_mad_confident.sum()), "of", len(kept), "cells (MAD)")


def plt_line(color, marker, label):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=color, marker=marker, ls="none", label=label)


if __name__ == "__main__":
    main()
