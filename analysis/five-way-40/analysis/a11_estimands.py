"""Analysis 11 (paper): which arm is faster depends on what "faster" means.

For every arm pair of every (nodes, kind, matrix), from per-launch paired
d = ln(q_A / q_B) over 40 launches (paired.py states the estimands):
  median_lr      median of d, with the distribution-free order-statistic 95% interval
  mean_lr        mean of d (paired t on ln q), with its t interval
  rom_lr         ln(mean q_A / mean q_B), expected cost, launch-bootstrap percentile 95%
  hl_lr          Hodges-Lehmann pseudo-median (Wilcoxon's estimand)
  clean_median_lr  median of d after removing whole 0.160 s stalls (coherent cells only)
  stall shares of each arm and an exact McNemar test on the paired stall indicators.
A "flip" is a comparison where two estimands disagree on which arm is faster; a
"confident flip" is one where both intervals exclude 0 on opposite sides.

Also, as the usual normality diagnostic, Shapiro-Wilk on q and on ln q per (nodes, kind,
matrix, arm) cell and on d per comparison, with skewness and excess kurtosis.

Writes tables/estimands.csv, tables/normality.csv, and figures estimand_median_vs_rom.png
and stall_steps_<kind>_crankseg_2_4nodes.png.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import ARM_COLOR, ARMS, launch_resamples, new_fig, save_fig, write_table
from paired import (QUANTUM_S, REGIME_LABEL, REGIMES, arm_comparisons, hodges_lehmann, key_of, order_stat_ci,
                    runs_with_stalls, t_ci)

REGIME_COLOR = {"none": "#1f77b4", "minority": "#ff7f0e", "majority": "#d62728"}


def normality(runs):
    rows = []
    for (nodes, kind, matrix, arm), g in runs.groupby(["nodes", "kind", "matrix", "arm"]):
        q = g.q_ms.to_numpy()
        for scale, v in (("q", q), ("ln q", np.log(q))):
            rows.append({"unit": "cell", "nodes": nodes, "kind": kind, "matrix": matrix, "arm": arm, "scale": scale,
                         "n": len(v), "shapiro_p": stats.shapiro(v).pvalue, "skew": stats.skew(v),
                         "excess_kurtosis": stats.kurtosis(v)})
    for c in arm_comparisons(runs):
        rows.append({"unit": "comparison", "nodes": c["nodes"], "kind": c["kind"], "matrix": c["matrix"],
                     "arm": f'{c["A"]}/{c["B"]}', "scale": "d", "n": len(c["d"]),
                     "shapiro_p": stats.shapiro(c["d"]).pvalue, "skew": stats.skew(c["d"]),
                     "excess_kurtosis": stats.kurtosis(c["d"])})
    t = pd.DataFrame(rows)
    t["rejects_5pct"] = t.shapiro_p < 0.05
    write_table(t, "normality.csv")
    print(t.groupby(["unit", "scale"]).agg(n=("rejects_5pct", "size"), rejects=("rejects_5pct", "mean"),
                                           median_skew=("skew", "median"),
                                           median_kurt=("excess_kurtosis", "median")).round(3).to_string())


def side(lo, hi):
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def main():
    runs = runs_with_stalls()
    normality(runs)
    rows = []
    for c in arm_comparisons(runs):
        d, qa, qb = c["d"], c["qA"], c["qB"]
        lo, hi = (v[0] for v in order_stat_ci(d[None]))
        tlo, thi = (v[0] for v in t_ci(d[None]))
        idx = launch_resamples(len(d), seed_offset=11)
        boot = np.log(qa[idx].mean(axis=1) / qb[idx].mean(axis=1))
        rlo, rhi = np.quantile(boot, [0.025, 0.975])
        b = int((c["stA"] & ~c["stB"]).sum())
        cc = int((~c["stA"] & c["stB"]).sum())
        rows.append({**key_of(c), "launches": len(d),
                     "median_lr": np.median(d), "median_lo": lo, "median_hi": hi,
                     "mean_lr": d.mean(), "mean_lo": tlo, "mean_hi": thi,
                     "rom_lr": np.log(qa.mean() / qb.mean()), "rom_lo": rlo, "rom_hi": rhi,
                     "hl_lr": hodges_lehmann(d), "clean_median_lr": np.median(c["d_clean"]),
                     "stall_share_A": c["stA"].mean(), "stall_share_B": c["stB"].mean(),
                     "mcnemar_p": stats.binomtest(b, b + cc).pvalue if b + cc else 1.0,
                     "median_side": side(lo, hi), "mean_side": side(tlo, thi), "rom_side": side(rlo, rhi)})
    t = pd.DataFrame(rows)
    sgn = np.sign
    t["flip_median_rom"] = sgn(t.median_lr) != sgn(t.rom_lr)
    t["flip_median_mean"] = sgn(t.median_lr) != sgn(t.mean_lr)
    t["confident_flip_median_rom"] = t.median_side * t.rom_side == -1
    t["confident_flip_median_mean"] = t.median_side * t.mean_side == -1
    t["clean_changes_median_side"] = sgn(t.clean_median_lr) != sgn(t.median_lr)
    write_table(t, "estimands.csv")

    print(f"{len(t)} arm-pair comparisons")
    print(t.groupby("regime").agg(n=("median_lr", "size"), flip_med_rom=("flip_median_rom", "sum"),
                                  conf_flip_med_rom=("confident_flip_median_rom", "sum"),
                                  flip_med_mean=("flip_median_mean", "sum"),
                                  conf_flip_med_mean=("confident_flip_median_mean", "sum"),
                                  clean_changes=("clean_changes_median_side", "sum"),
                                  mcnemar_sig=("mcnemar_p", lambda p: int((p < 0.05).sum()))).to_string())
    print(t[t.confident_flip_median_rom | t.confident_flip_median_mean][
              ["nodes", "kind", "matrix", "A", "B", "regime", "median_lr", "mean_lr", "rom_lr",
               "stall_share_A", "stall_share_B"]].round(3).to_string(index=False))

    fig, ax = new_fig("The typical launch and the average cost can pick different arms",
                      "median over launches of ln(time A / time B)", "ln(mean time A / mean time B)",
                      width=6.5, height=6.0,
                      caption=f"One point per arm pair of every node count, kind, and matrix ({len(t)} comparisons, 40 "
                              f"launches each). The horizontal axis is the typical launch, the vertical axis the "
                              f"expected cost including stalls; below 0 means the first arm of the pair is faster. "
                              f"Points in the upper-left and lower-right quadrants are comparisons where the two "
                              f"questions pick different arms. Color is the stall regime (share of launches with a "
                              f"stalled run, the larger of the two arms).")
    for r in REGIMES:
        s = t[t.regime == r]
        ax.scatter(s.median_lr, s.rom_lr, s=12, color=REGIME_COLOR[r], label=f"{REGIME_LABEL[r]} ({len(s)})")
    lim = np.abs(t[["median_lr", "rom_lr"]].to_numpy()).max() * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.6, ls=":")
    ax.legend(fontsize=7.5, loc="upper left")
    save_fig(fig, "estimand_median_vs_rom.png")

    for kind in ("pm", "abcgspmv", "abcgcg"):
        d = runs[(runs.nodes == 4) & (runs.kind == kind) & (runs.matrix == "crankseg_2")]
        fig, ax = new_fig(f"Extra time per run in whole stalls: {kind}, crankseg_2, 4 nodes", "arm",
                          "timed total minus the fastest launch's (units of 0.160 s)", width=6.0, height=4.6,
                          caption=f"One dot per launch (40 per arm), slightly spread sideways. The vertical axis is how "
                                  f"much longer the run's timed loop took than the fastest launch of the same arm, in "
                                  f"units of 0.160 s. n = "
                                  f"{', '.join(f'{a} {int(d[d.arm == a].n.iloc[0])}' for a in ARMS)} timed iterations.")
        rng = np.random.default_rng(4)
        for i, arm in enumerate(ARMS):
            s = d[d.arm == arm]
            ex = (s.timedTotal_max - s.timedTotal_max.min()) / 1000 / QUANTUM_S
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(ex)), ex, s=12, color=ARM_COLOR[arm])
        ax.set_xticks(range(len(ARMS)), ARMS)
        top = max(1, int(np.ceil(((d.timedTotal_max - d.groupby("arm").timedTotal_max.transform("min")) / 1000
                                  / QUANTUM_S).max())))
        ax.set_yticks(np.arange(0, top + 1, max(1, top // 12)))
        save_fig(fig, f"stall_steps_{kind}_crankseg_2_4nodes.png")


if __name__ == "__main__":
    main()
