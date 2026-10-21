"""Analysis 12 (paper): do the tests behave as advertised on this data?

The referee is the data's own distribution, not a nonparametric statistic. For every arm
pair of every (nodes, kind, matrix) (504 comparisons), each test is run in a bootstrap
world where its own null is exactly true: the 40 paired values are shifted so that the
test's estimand is 0, and samples of n launches are drawn from them with replacement
(no overlap problem with a 40-launch target).
  t_ln      paired t on ln q (one-sample t on d), world: d - mean(d)
  t_q       paired t on raw q (one-sample t on q_A - q_B), world: x - mean(x)
  sign      exact sign test on d, world: d - median(d)
  wilcoxon  exact signed-rank on d, world: d - HL(d)
  t_ln_vs_median  paired t on ln q in the sign test's world (the typical launch is a tie):
                  not a size, an estimand mismatch
Size: rejection rate at 5% for n in {5, 10, 20, 40}, B = 2000 draws (Monte Carlo SE about
0.5 points at 5%). Power: the same worlds shifted by ln 1.05 and ln 1.10. Coverage: each
95% interval against its own estimand in the unshifted world (t: mean of d; order
statistic: median of d; bootstrap percentile and BCa: median of d, on 60 comparisons drawn
at random, 400 draws x 999 inner resamples). Sample size: per (nodes, kind, matrix, arm)
cell, the launches a normal-theory calculation asks for to pin the mean of ln q to
+/-10% at 95% (the method used to choose 40 launches), the coverage the t interval then
actually has, and the launches the distribution-free median interval needs for the same
width.

Writes tables/calibration_size.csv, calibration_power.csv, calibration_coverage.csv,
sample_size.csv, and figures calibration_size_<regime>.png, calibration_power_<pct>.png,
calibration_coverage.png, sample_size_normal_vs_median.png.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import SEED, new_fig, save_fig, write_table
from paired import (ALPHA, ARM_PAIRS, KINDS, REGIME_LABEL, REGIMES, arm_comparisons, hl_rows, hodges_lehmann, jitter,
                    key_of, order_stat_ci, order_stat_index, runs_with_stalls, sign_p, t_ci, t_p, wilcoxon_p)

NS = [5, 10, 20, 40]
B = 2000
SHIFTS = {"5pct": np.log(1.05), "10pct": np.log(1.10)}
TESTS = ["t_ln", "t_q", "sign", "wilcoxon"]
TEST_LABEL = {"t_ln": "paired t on ln q", "t_q": "paired t on q", "sign": "sign test", "wilcoxon": "Wilcoxon signed-rank",
              "t_ln_vs_median": "paired t on ln q, typical launch tied"}
TEST_COLOR = {"t_ln": "#1f77b4", "t_q": "#9467bd", "sign": "#2ca02c", "wilcoxon": "#ff7f0e",
              "t_ln_vs_median": "#7f7f7f"}
MARGIN = np.log(1.1)
# Starts at 6, the smallest n whose order-statistic interval of the median reaches 95%.
N_GRID = [6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400]


def mad(x):
    return float(np.median(np.abs(x - np.median(x)))) or float(np.std(x)) or 1.0


def bca_median_rows(X, rng, inner=999, level=0.95):
    """BCa interval of the median for each row of X (inner bootstrap per row)."""
    n = X.shape[1]
    lo, hi = np.empty(len(X)), np.empty(len(X))
    za = stats.norm.ppf([(1 - level) / 2, (1 + level) / 2])
    for r, x in enumerate(X):
        theta = np.median(x)
        boot = np.median(x[rng.integers(0, n, (inner, n))], axis=1)
        prop = np.clip((boot < theta).mean() + 0.5 * (boot == theta).mean(), 1 / inner, 1 - 1 / inner)
        z0 = stats.norm.ppf(prop)
        jack = np.array([np.median(np.delete(x, i)) for i in range(n)])
        u = jack.mean() - jack
        den = 6 * (u ** 2).sum() ** 1.5
        a = (u ** 3).sum() / den if den > 0 else 0.0
        adj = stats.norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
        lo[r], hi[r] = np.quantile(boot, adj)
    return lo, hi


def main():
    runs = runs_with_stalls()
    comps = arm_comparisons(runs)
    rng = np.random.default_rng(SEED + 12)
    size, power, cover = [], [], []
    for ci, c in enumerate(comps):
        d = c["d"]
        x = c["qA"] - c["qB"]
        s = mad(d)
        worlds = {"t_ln": d - d.mean(), "t_q": x - x.mean(), "sign": d - np.median(d),
                  "wilcoxon": d - hodges_lehmann(d), "t_ln_vs_median": d - np.median(d)}
        for n in NS:
            idx = rng.integers(0, len(d), (B, n))
            row = {**key_of(c), "n": n}
            P = {"t_ln": t_p(worlds["t_ln"][idx]), "t_q": t_p(worlds["t_q"][idx]),
                 "sign": sign_p(jitter(worlds["sign"][idx], rng, s)),
                 "wilcoxon": wilcoxon_p(jitter(worlds["wilcoxon"][idx], rng, s)),
                 "t_ln_vs_median": t_p(worlds["t_ln_vs_median"][idx])}
            size.append({**row, **{k: (p < ALPHA).mean() for k, p in P.items()}})
            for name, delta in SHIFTS.items():
                power.append({**row, "shift": name,
                              "t_ln": (t_p(worlds["t_ln"][idx] + delta) < ALPHA).mean(),
                              "sign": (sign_p(jitter(worlds["sign"][idx] + delta, rng, s)) < ALPHA).mean(),
                              "wilcoxon": (wilcoxon_p(jitter(worlds["wilcoxon"][idx] + delta, rng, s)) < ALPHA).mean()})
            X = d[idx]
            tlo, thi = t_ci(X)
            olo, ohi = order_stat_ci(X)
            med = np.median(d)
            cover.append({**row, "method": "t (mean)", "coverage": ((tlo <= d.mean()) & (thi >= d.mean())).mean(),
                          "median_halfwidth": np.median((thi - tlo) / 2)})
            cover.append({**row, "method": "order statistic (median)",
                          "coverage": ((olo <= med) & (ohi >= med)).mean(),
                          "median_halfwidth": np.median((ohi - olo) / 2)})
    size, power = pd.DataFrame(size), pd.DataFrame(power)

    # Bootstrap intervals of the median: nested, so on a random subset of comparisons.
    sub = np.random.default_rng(SEED + 120).choice(len(comps), 60, replace=False)
    for ci in sub:
        c = comps[ci]
        d = c["d"]
        med = np.median(d)
        for n in (10, 20, 40):
            X = d[rng.integers(0, len(d), (400, n))]
            boot = np.median(X[np.arange(400)[:, None, None], rng.integers(0, n, (400, 999, n))], axis=2)
            plo, phi = np.quantile(boot, [0.025, 0.975], axis=1)
            blo, bhi = bca_median_rows(X, rng)
            row = {**key_of(c), "n": n}
            cover.append({**row, "method": "bootstrap percentile (median)",
                          "coverage": ((plo <= med) & (phi >= med)).mean(), "median_halfwidth": np.median((phi - plo) / 2)})
            cover.append({**row, "method": "BCa (median)",
                          "coverage": ((blo <= med) & (bhi >= med)).mean(), "median_halfwidth": np.median((bhi - blo) / 2)})
    cover = pd.DataFrame(cover)
    write_table(size, "calibration_size.csv")
    write_table(power, "calibration_power.csv")
    write_table(cover, "calibration_coverage.csv")

    tests = TESTS + ["t_ln_vs_median"]
    print("size: median [p10, p90] rejection rate over comparisons; share of comparisons above 7.5%")
    for regime in REGIMES:
        s = size[size.regime == regime]
        print(f"-- {regime} ({s.groupby(['nodes', 'kind', 'matrix', 'A', 'B']).ngroups} comparisons)")
        for n in NS:
            g = s[s.n == n]
            print(f"  n={n:2d} " + "  ".join(f"{k} {g[k].median():.3f} [{g[k].quantile(.1):.3f},{g[k].quantile(.9):.3f}]"
                                              f" >7.5%:{(g[k] > .075).mean():.2f}" for k in tests))
    print("power: median over comparisons")
    print(power.groupby(["shift", "regime", "n"])[["t_ln", "sign", "wilcoxon"]].median().round(3).to_string())
    print("coverage: median [p10] over comparisons")
    print(cover.groupby(["method", "n"]).coverage.agg(["median", lambda v: v.quantile(.1), "size"]).round(3).to_string())

    sample = sample_size(runs, rng)
    write_table(sample, "sample_size.csv")
    print(sample.groupby("arm")[["n_normal", "t_coverage_at_n_normal", "n_median"]].describe().round(2).to_string())
    big = sample[sample.n_normal >= N_GRID[0]]
    print(f"cells with n_normal >= {N_GRID[0]}: {len(big)} of {len(sample)}; of those, the median needs more "
          f"launches in {int((big.n_median > big.n_normal).sum() + big.n_median_over_400.sum())} "
          f"(over 400 in {int(big.n_median_over_400.sum())}), fewer in {int((big.n_median < big.n_normal).sum())}")
    print(f"t coverage at n_normal below 0.93 in {int((sample.t_coverage_at_n_normal < 0.93).sum())} of {len(sample)} cells")

    figures(size, power, cover, sample)


def sample_size(runs, rng):
    rows = []
    z = 1.959963984540054
    for (nodes, kind, matrix, arm), g in runs.groupby(["nodes", "kind", "matrix", "arm"]):
        y = np.log(g.q_ms.to_numpy())
        sd = y.std(ddof=1)
        n_normal = next((n for n in range(2, 100000) if stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) <= MARGIN), None)
        nn = min(n_normal, 400)
        X = y[rng.integers(0, len(y), (B, nn))]
        lo, hi = t_ci(X)
        n_med = None
        for n in N_GRID:
            Xm = y[rng.integers(0, len(y), (500, n))]
            olo, ohi = order_stat_ci(Xm)
            if np.median((ohi - olo) / 2) <= MARGIN:
                n_med = n
                break
        rows.append({"nodes": nodes, "kind": kind, "matrix": matrix, "arm": arm, "sd_ln_q": sd,
                     "skew_ln_q": stats.skew(y), "n_normal": n_normal, "n_normal_simulated": nn,
                     "t_coverage_at_n_normal": ((lo <= y.mean()) & (hi >= y.mean())).mean(),
                     "n_median": n_med if n_med is not None else np.nan, "n_median_over_400": n_med is None})
    return pd.DataFrame(rows)


def figures(size, power, cover, sample):
    tests = TESTS + ["t_ln_vs_median"]
    for regime in REGIMES:
        s = size[size.regime == regime]
        k = s.groupby(["nodes", "kind", "matrix", "A", "B"]).ngroups
        fig, ax = new_fig(f"False alarms when the null is true: {REGIME_LABEL[regime]}", "launches per comparison (n)",
                          "rejection rate at the 5% level",
                          caption=f"{k} arm-pair comparisons. Each test runs on n launches drawn with replacement from "
                                  f"the comparison's own 40 paired values, shifted so that the test's own null is "
                                  f"exactly true ({B} draws per n). Points are the median over comparisons, bars the "
                                  f"10th-90th percentiles; the dashed line is the nominal 5%. The grey series is "
                                  f"paired t on ln q when the typical launch is a tie: it is not a false alarm under "
                                  f"t's own null, but it is how often t calls a winner the median would not.")
        for j, test in enumerate(tests):
            g = s.groupby("n")[test]
            x = np.array(NS) * (1 + 0.04 * (j - 2))
            med = g.median().reindex(NS)
            ax.errorbar(x, med, yerr=np.vstack([med - g.quantile(.1).reindex(NS), g.quantile(.9).reindex(NS) - med]),
                        fmt="o-", ms=4, capsize=2, color=TEST_COLOR[test], label=TEST_LABEL[test])
        ax.axhline(ALPHA, color="black", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.legend(fontsize=7.5)
        save_fig(fig, f"calibration_size_{regime}.png")

    for name in SHIFTS:
        p = power[power["shift"] == name]
        fig, ax = new_fig(f"Chance of detecting a true {name.replace('pct', '%')} difference", "launches per comparison (n)",
                          "rejection rate at the 5% level",
                          caption=f"The worlds of the size test, shifted by ln {1 + int(name[:-3]) / 100:.2f} so that one "
                                  f"arm is truly {name.replace('pct', '%')} slower by each test's own estimand. Lines "
                                  f"are the median over the {p.groupby(['nodes', 'kind', 'matrix', 'A', 'B']).ngroups} "
                                  f"comparisons, one line style per stall regime. At n = 5 neither the sign test nor "
                                  f"Wilcoxon can reach p < 0.05 (their smallest p-value is 0.0625).")
        for test in ("t_ln", "sign", "wilcoxon"):
            for regime, ls in zip(REGIMES, ["-", "--", ":"]):
                g = p[p.regime == regime].groupby("n")[test].median().reindex(NS)
                ax.plot(NS, g, ls=ls, marker="o", ms=3, color=TEST_COLOR[test],
                        label=f"{TEST_LABEL[test]}, {regime}")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6.5, ncol=1)
        save_fig(fig, f"calibration_power_{name}.png")

    fig, ax = new_fig("How often each 95% interval contains its own target", "launches per comparison (n)",
                      "coverage", caption="Each interval is computed on n launches drawn with replacement from a "
                                          "comparison's 40 paired values and scored against that comparison's own "
                                          "mean (t) or median (the others). Points are medians over comparisons and "
                                          "bars the 10th-90th percentiles: all 504 comparisons for t and the order "
                                          "statistic, 60 drawn at random for the two bootstrap intervals. The "
                                          "order-statistic interval cannot reach 95% at n = 5 (its widest interval, "
                                          "minimum to maximum, covers 93.75%).")
    colors = {"t (mean)": "#1f77b4", "order statistic (median)": "#2ca02c",
              "bootstrap percentile (median)": "#8c564b", "BCa (median)": "#e377c2"}
    for j, (m, col) in enumerate(colors.items()):
        g = cover[cover.method == m].groupby("n").coverage
        ns = sorted(cover[cover.method == m].n.unique())
        med = g.median().reindex(ns)
        ax.errorbar(np.array(ns) * (1 + 0.04 * (j - 1.5)), med,
                    yerr=np.vstack([med - g.quantile(.1).reindex(ns), g.quantile(.9).reindex(ns) - med]),
                    fmt="o-", ms=4, capsize=2, color=col, label=m)
    ax.axhline(0.95, color="black", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.legend(fontsize=8)
    save_fig(fig, "calibration_coverage.png")

    s = sample.copy()
    s["n_median_plot"] = s.n_median.fillna(600)
    fig, ax = new_fig("Launches needed for a +/-10% interval: normal theory vs. the median",
                      "launches a normal-theory calculation asks for (mean of ln q)",
                      "launches the distribution-free median interval needs", width=6.5, height=6.0,
                      caption=f"One point per node count, kind, matrix, and arm ({len(s)} cells). Horizontal: the n at "
                              f"which a t interval for the mean of ln q is +/-10% wide, from the 40 launches' standard "
                              f"deviation. Vertical: the smallest n (grid 6-400, drawn with replacement) at which the "
                              f"median order-statistic interval is typically that narrow; cells needing more than 400 "
                              f"sit at the top edge. The dotted line is equality.")
    for arm, col in (("MPI", "#1f77b4"), ("STANDARD", "#ff7f0e"), ("LOCALITY", "#2ca02c")):
        g = s[s.arm == arm]
        ax.scatter(g.n_normal, g.n_median_plot, s=10, color=col, label=arm, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = [2, max(s.n_normal.max(), 600) * 1.2]
    ax.plot(lim, lim, color="grey", ls=":", lw=0.8)
    ax.axhline(40, color="grey", lw=0.6)
    ax.axvline(40, color="grey", lw=0.6)
    ax.legend(fontsize=8)
    save_fig(fig, "sample_size_normal_vs_median.png")


if __name__ == "__main__":
    main()
