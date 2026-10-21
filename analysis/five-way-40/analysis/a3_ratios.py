"""Analysis 3: paired fidelity ratios.

Per node count x matrix x arm, per launch: log(stand-in q / reference q) for
  fine vs pm, coarse vs pm, pm vs abcgspmv, abcgspmv vs abcgspmvnp, abcgcg vs abcgcgnp.
Reported as the median ratio ("x times") with a BCa 95% interval over launches, and a
TOST equivalence verdict at +/-10% stated before computing: equivalent when the 90% BCa
interval of the median log ratio lies inside [ln 0.9, ln 1.1]. Slow runs kept and
dropped. Writes tables/ratios.csv and one figure per comparison and node count,
ratio_<a>_vs_<b>_<N>nodes.png.
"""
import numpy as np
import pandas as pd

from common import HUMAN, ARM_COLOR, ARMS, KIND_LABEL, MATRICES, NODES, bca, load_runs, new_fig, q_table, save_fig, write_table

COMPARISONS = [("fine", "pm"), ("coarse", "pm"), ("pm", "abcgspmv"), ("abcgspmv", "abcgspmvnp"),
               ("abcgcg", "abcgcgnp")]
LO, HI = np.log(0.9), np.log(1.1)


def main():
    runs = load_runs()
    rows = []
    for a, b in COMPARISONS:
        for nodes in NODES:
            for matrix in MATRICES:
                for drop in (False, True):
                    T = q_table(runs, nodes, matrix, [a, b], drop_slow=drop)
                    for arm in ARMS:
                        lr = np.log(T[(a, arm)].to_numpy() / T[(b, arm)].to_numpy())
                        lo95, hi95 = bca(lr)
                        lo90, hi90 = bca(lr, level=0.90)
                        rows.append({"stand_in": a, "reference": b, "nodes": nodes, "matrix": matrix, "arm": arm,
                                     "slow_runs": "dropped" if drop else "kept", "launches": len(lr),
                                     "median_ratio": np.exp(np.median(lr)), "lo95": np.exp(lo95), "hi95": np.exp(hi95),
                                     "lo90": np.exp(lo90), "hi90": np.exp(hi90),
                                     "tost_equivalent_10pct": bool(lo90 > LO and hi90 < HI)})
    table = pd.DataFrame(rows)
    write_table(table, "ratios.csv")

    kept = table[table.slow_runs == "kept"]
    for a, b in COMPARISONS:
        for nodes in NODES:
            t = kept[(kept.stand_in == a) & (kept.reference == b) & (kept.nodes == nodes)]
            fig, ax = new_fig(f"Time of {HUMAN[a]} relative to {HUMAN[b]}, {nodes} nodes",
                              "matrix", "time ratio (per iteration)",
                              caption=f"Within each launch, the time per iteration of {HUMAN[a]} divided by that of "
                                      f"{HUMAN[b]} on the same arm; points are medians over launches with 95% BCa "
                                      f"intervals, one color per arm. Above 1 means {HUMAN[a]} is slower. The grey "
                                      f"band is the 10% equivalence margin. {t.launches.min()}-{t.launches.max()} "
                                      f"launches per point; slow runs kept.")
            x = np.arange(len(MATRICES))
            for k, arm in enumerate(ARMS):
                s = t[t.arm == arm].set_index("matrix").reindex(MATRICES)
                ax.errorbar(x + (k - 1) * 0.2, s.median_ratio,
                            yerr=np.vstack([s.median_ratio - s.lo95, s.hi95 - s.median_ratio]),
                            fmt="o", capsize=3, color=ARM_COLOR[arm], label=arm)
            ax.axhspan(0.9, 1.1, color="grey", alpha=0.15)
            ax.axhline(1, color="grey", lw=0.8)
            ax.set_yscale("log")
            ax.set_xticks(x, MATRICES)
            ax.legend(fontsize=8)
            save_fig(fig, f"ratio_{a}_vs_{b}_{nodes}nodes.png")

    summary = kept.groupby(["stand_in", "reference", "arm"]).agg(
        points=("median_ratio", "size"), equivalent=("tost_equivalent_10pct", "sum"),
        min_ratio=("median_ratio", "min"), max_ratio=("median_ratio", "max"))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
