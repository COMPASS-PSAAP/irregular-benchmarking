"""Analysis 6 (H8): setup amortization. At what reuse count N does LOCALITY overtake
MPI and STANDARD?

cost(N) = setup + q N per arm, with
  setup = haloTime_max + gatherTime_max + (startupWindow_max - k q)   benchmark <m>
  setup = setupTime_max + (startupWindow_max - k q)                   ABCG kinds
fine_ and coarse_ are excluded: their haloTime includes reading the Vernier files
(230-960 ms on every arm, against ~1 ms for <m>), so it is not communication setup.
The -W nearest connection warmup is untimed, so connection setup is outside N*.

From the medians over launches of setup and q per arm:
  always   LOCALITY has lower setup and lower q
  never    LOCALITY has higher setup and higher q
  after    LOCALITY has higher setup, lower q: wins for N > N* = d_setup / d_q
  until    LOCALITY has lower setup, higher q: wins for N < N*
Interval: launch bootstrap of the same computation (N* percentiles over resamples whose
category matches the point estimate; p_category is that share). Slow runs kept and
dropped. Writes tables/h8.csv and figures h8_nstar_<kind>_LOCALITY_vs_<arm>.png.
"""
import numpy as np
import pandas as pd

from common import HUMAN, B, KIND_LABEL, MATRICES, NODES, launch_resamples, load_runs, new_fig, save_fig, write_table

KINDS = ["pm", "abcgspmv", "abcgspmvnp", "abcgcg", "abcgcgnp"]
NODE_MARK = {2: "o", 4: "s", 8: "D", 16: "^"}


def classify(setup_l, q_l, setup_x, q_x):
    """Vectorized: returns (category codes 0..3, N*)."""
    d_setup, d_q = setup_l - setup_x, q_x - q_l
    with np.errstate(divide="ignore", invalid="ignore"):
        nstar = d_setup / d_q
    cat = np.where((d_setup <= 0) & (d_q >= 0), 0, np.where((d_setup >= 0) & (d_q <= 0), 1,
                                                            np.where(d_q > 0, 2, 3)))
    return cat, np.where(cat >= 2, nstar, np.nan)


CATS = ["always", "never", "after", "until"]


def main():
    runs = load_runs()
    runs = runs[runs.kind.isin(KINDS)].copy()
    fixed = np.where(runs.kind == "pm", runs.haloTime_max + runs.gatherTime_max, runs.setupTime_max)
    runs["setup_ms"] = fixed + runs.startupWindow_max - runs.k * runs.q_ms
    rows = []
    for kind in KINDS:
        for nodes in NODES:
            for matrix in MATRICES:
                for drop in (False, True):
                    sub = runs[(runs.kind == kind) & (runs.nodes == nodes) & (runs.matrix == matrix)]
                    if drop:
                        sub = sub[~sub.slow]
                    w = sub.pivot_table(index="launch", columns="arm", values=["setup_ms", "q_ms"]).dropna()
                    S, Q = w["setup_ms"], w["q_ms"]
                    idx = launch_resamples(len(w))
                    for other in ("MPI", "STANDARD"):
                        args = [np.median(S["LOCALITY"]), np.median(Q["LOCALITY"]),
                                np.median(S[other]), np.median(Q[other])]
                        cat, nstar = classify(*[np.array([a]) for a in args])
                        bargs = [np.median(v.to_numpy()[idx], axis=1) for v in
                                 (S["LOCALITY"], Q["LOCALITY"], S[other], Q[other])]
                        bcat, bnstar = classify(*bargs)
                        same = bcat == cat[0]
                        lo, hi = (np.quantile(bnstar[same], [0.025, 0.975]) if cat[0] >= 2 and same.any()
                                  else (np.nan, np.nan))
                        rows.append({"kind": kind, "nodes": nodes, "matrix": matrix, "versus": other,
                                     "slow_runs": "dropped" if drop else "kept", "launches": len(w),
                                     "setup_ms_LOCALITY": args[0], "q_ms_LOCALITY": args[1],
                                     f"setup_ms_other": args[2], "q_ms_other": args[3],
                                     "category": CATS[cat[0]], "p_category": float(same.mean()),
                                     "nstar": nstar[0], "nstar_lo": lo, "nstar_hi": hi,
                                     **{f"p_{c}": float(np.mean(bcat == k)) for k, c in enumerate(CATS)}})
    table = pd.DataFrame(rows)
    write_table(table, "h8.csv")

    kept = table[table.slow_runs == "kept"]
    for kind in KINDS:
        for other in ("MPI", "STANDARD"):
            t = kept[(kept.kind == kind) & (kept.versus == other)]
            fig, ax = new_fig(f"Iterations before LOCALITY beats {other}, {HUMAN[kind]}",
                              "matrix", "break-even iterations (N*)", width=8.5, height=6.4,
                              caption=f"N* is the number of iterations after which LOCALITY's extra setup cost has "
                                      f"been paid back by its faster iterations, from medians over launches with 95% "
                                      f"launch-bootstrap intervals. Black: LOCALITY wins after N* (plotted at 1 if it "
                                      f"wins from the start). Red at the top: LOCALITY never wins. Red elsewhere: "
                                      f"LOCALITY wins only below N*. Dotted lines: the 2023 paper's 22 and 40. "
                                      f"{t.launches.min()}-{t.launches.max()} launches per point; slow runs kept.")
            top = 1e8
            for k, nodes in enumerate(NODES):
                s = t[t.nodes == nodes].set_index("matrix").reindex(MATRICES)
                x = np.arange(len(MATRICES)) + (k - 1.5) * 0.15
                y = np.where(s.category == "always", 1, np.where(s.category == "never", top, s.nstar))
                filled = s.category.isin(["after", "always"]).to_numpy()
                lo = np.where(s.category.isin(["after", "until"]), s.nstar_lo, y)
                hi = np.where(s.category.isin(["after", "until"]), s.nstar_hi, y)
                ax.errorbar(x, y, yerr=np.clip(np.vstack([y - lo, hi - y]), 0, None), fmt="none",
                            ecolor="grey", capsize=2)
                ax.scatter(x[filled], y[filled], marker=NODE_MARK[nodes], color="black", label=f"{nodes} nodes")
                ax.scatter(x[~filled], y[~filled], marker=NODE_MARK[nodes], facecolors="none", edgecolors="red",
                           label=f"{nodes} nodes (never / until)")
            ax.axhline(22, color="#1f77b4", ls=":", lw=1)
            ax.axhline(40, color="#1f77b4", ls=":", lw=1)
            ax.text(len(MATRICES) - 0.5, 30, "2023 paper: 22, 40", fontsize=7, color="#1f77b4", ha="right")
            ax.set_yscale("log")
            ax.set_ylim(0.5, top * 3)
            ax.set_xticks(np.arange(len(MATRICES)), MATRICES)
            ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            save_fig(fig, f"h8_nstar_{kind}_LOCALITY_vs_{other}.png")

    print(kept.pivot_table(index=["kind", "versus", "nodes"], columns="matrix",
                           values="nstar", aggfunc="first").round(0).to_string())
    print(pd.crosstab([kept.kind, kept.versus], kept.category).to_string())


if __name__ == "__main__":
    main()
