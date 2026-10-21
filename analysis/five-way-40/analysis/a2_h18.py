"""Analysis 2 (H18): the fidelity ladder. Does coarse_ (aggregate Vernier manager) pick
the arm fine_ (hierarchical manager) picks, and the application's? Where it does not,
what did the aggregate drop?

Ordering and regret reuse a1_h17.ordering_stats with fine_ as the reference (coarse_
against the application is in h17_design_points.csv). Structure compares the per-rank
counts of coarse_ against <m> (fine_ equals <m> exactly; see audit.csv) from the MPI-arm
runs: coarse_ draws a new pattern per run, so its value is the median over launches
with the min and max beside it. The mean-bytes check the prompt asks for is the
data_sent_avg row.

Writes tables/h18_coarse_vs_fine.csv, tables/h18_structure.csv,
tables/h18_disagreements.csv, and figures regret_coarse_vs_fine_<N>nodes.png and
coarse_structure_ratio_<N>nodes.png.
"""
import numpy as np
import pandas as pd

from a1_h17 import compare, plot_regret
from common import MATRICES, NODES, RESULTS, load_runs, new_fig, save_fig, write_table

STRUCTURE = ["numberOfmessages_avg", "numberOfmessages_max", "data_sent_avg", "data_sent_max",
             "onNodeNeighbors_avg", "offNodeNeighbors_avg", "onNodeBytes_avg", "offNodeBytes_avg",
             "inNumaBytes_avg", "outNumaBytes_avg"]


def structure(runs):
    rows = []
    mpi = runs[runs.arm == "MPI"]
    for nodes in NODES:
        for matrix in MATRICES:
            pm = mpi[(mpi.nodes == nodes) & (mpi.kind == "pm") & (mpi.matrix == matrix)]
            co = mpi[(mpi.nodes == nodes) & (mpi.kind == "coarse") & (mpi.matrix == matrix)]
            for metric in STRUCTURE:
                ref = pm[metric].iloc[0]
                assert (pm[metric] == ref).all(), (nodes, matrix, metric)   # deterministic for <m>
                rows.append({"nodes": nodes, "matrix": matrix, "metric": metric, "pm": ref,
                             "coarse_median": co[metric].median(), "coarse_min": co[metric].min(),
                             "coarse_max": co[metric].max(), "coarse_runs": len(co),
                             "ratio_median": co[metric].median() / ref if ref else np.nan,
                             "ratio_min": co[metric].min() / ref if ref else np.nan,
                             "ratio_max": co[metric].max() / ref if ref else np.nan})
    return pd.DataFrame(rows)


def plot_structure(table, nodes):
    t = table[table.nodes == nodes]
    fig, ax = new_fig(f"What the coarse Vernier capture changes about the pattern, {nodes} nodes",
                      "per-rank quantity", "coarse value / exact value", width=10, height=5.4,
                      caption=f"Each point divides the coarse replay's value by the matrix's exact value (the fine "
                              f"replay matches the exact pattern exactly); 1 means preserved. Marker = median over "
                              f"launches, bar = range over {int(t.coarse_runs.min())}-{int(t.coarse_runs.max())} "
                              f"launches, since the coarse replay draws a new pattern every run. From the MPI-arm "
                              f"runs.")
    x = np.arange(len(STRUCTURE))
    for k, matrix in enumerate(MATRICES):
        s = t[t.matrix == matrix].set_index("metric").reindex(STRUCTURE)
        off = (k - 2.5) * 0.12
        ax.errorbar(x + off, s.ratio_median, yerr=np.clip(np.vstack([s.ratio_median - s.ratio_min,
                                                                     s.ratio_max - s.ratio_median]), 0, None),
                    fmt="o", ms=3.5, capsize=2, label=matrix)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x, [m.replace("_avg", " (avg)").replace("_max", " (max rank)") for m in STRUCTURE],
                  rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=3)
    save_fig(fig, f"coarse_structure_ratio_{nodes}nodes.png")


def main():
    runs = load_runs()
    points = []
    for drop in (False, True):
        p, _, _ = compare(runs, "coarse", "fine", drop)
        points += p
    points = pd.DataFrame(points)
    write_table(points, "h18_coarse_vs_fine.csv")
    for nodes in NODES:
        plot_regret(points, "coarse", "fine", nodes)

    table = structure(runs)
    write_table(table, "h18_structure.csv")
    for nodes in NODES:
        plot_structure(table, nodes)

    h17 = pd.read_csv(RESULTS / "tables" / "h17_design_points.csv")
    h17 = h17[(h17.slow_runs == "kept") & h17.reference.isin(["abcgspmv", "abcgcg"])]
    wide = h17.pivot_table(index=["nodes", "matrix", "reference"], columns="stand_in",
                           values=["stand_in_choice", "decision_regret"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    vs_fine = points[points.slow_runs == "kept"].set_index(["nodes", "matrix"])
    wide["coarse_vs_fine_regret"] = [vs_fine.loc[(n, m), "decision_regret"] for n, m in zip(wide.nodes, wide.matrix)]
    ratios = table.pivot_table(index=["nodes", "matrix"], columns="metric", values="ratio_median")
    disagree = wide[(wide.decision_regret_coarse > 0.05) | (wide.coarse_vs_fine_regret > 0.05)]
    disagree = disagree.join(ratios, on=["nodes", "matrix"])
    write_table(disagree, "h18_disagreements.csv")

    print(points[points.slow_runs == "kept"][["nodes", "matrix", "stand_in_choice", "reference_best",
                                              "decision_regret", "decision_regret_lo", "decision_regret_hi",
                                              "argmin_agree"]].round(3).to_string(index=False))
    print("\ncoarse/<m> mean bytes per rank (data_sent_avg), median over launches:")
    print(table[table.metric == "data_sent_avg"].pivot(index="matrix", columns="nodes",
                                                        values="ratio_median").round(3).to_string())


if __name__ == "__main__":
    main()
