"""Normality check of the per-launch statistics behind the bounds in python.py.

The t bound and the normal tolerance bound assume the launch statistics are
Gaussian across launches. D(N) is driven by A (startup saved) at short hold
lengths and by d (per-iteration saving) at long ones, so both are checked, for
every node count, pattern, and comparison in RESULTS_DIR/launches.csv.

Usage: python qq.py [RESULTS_DIR]        (default: results/ next to this file)
Writes RESULTS_DIR/normality.csv and one QQ plot per case under RESULTS_DIR/qq/.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ALPHA = 0.05
QUANTITIES = [("d_ms", "d: per-iteration saving (ms)"), ("A_ms", "A: startup saved (ms)")]

# Same chart styling as python.py.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
POINT, LINE = "#2a78d6", "#52514e"

HERE = Path(__file__).resolve().parent


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def qq_chart(values, title, ylabel, path):
    (theoretical, ordered), (slope, intercept, r) = stats.probplot(values, dist="norm")
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=11)
    ax.plot(theoretical, slope * theoretical + intercept, color=LINE, linewidth=1.5,
            linestyle="--", label="normal fit")
    ax.plot(theoretical, ordered, linestyle="none", marker="o", markersize=7, color=POINT,
            markeredgecolor=SURFACE, markeredgewidth=0.8, label="launches")
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("normal quantile", color=INK_2, fontsize=12)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=12)
    ax.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    groups = defaultdict(lambda: defaultdict(list))
    with (results / "launches.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["nodes"]), row["pattern"], row["comparison"])
            for col, _ in QUANTITIES:
                groups[key][col].append(float(row[col]))

    rows = []
    for (nodes, pattern, comp), cols in sorted(groups.items()):
        for col, label in QUANTITIES:
            x = np.array(cols[col])
            if len(x) < 3:
                continue
            w, p = stats.shapiro(x)
            rows.append([nodes, pattern, comp, col, len(x), w, p, stats.skew(x),
                         stats.kurtosis(x), "no" if p < ALPHA else "yes"])
            qq_chart(x, f"{nodes} nodes, {pattern}, {comp}: {col[0]}\n"
                        f"Shapiro-Wilk p = {p:.3g}, L = {len(x)}",
                     label, results / "qq" / f"{nodes}nodes_{slug(pattern)}_{slug(comp)}_{col[0]}.png")

    with (results / "normality.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nodes", "pattern", "comparison", "quantity", "launches",
                         "shapiro_W", "shapiro_p", "skewness", "excess_kurtosis",
                         f"consistent_with_normal_at_{ALPHA}"])
        writer.writerows(rows)

    print(f"{len(rows)} cases from {results / 'launches.csv'}")
    print(f"rejected normality at p < {ALPHA}: {sum(r[6] < ALPHA for r in rows)} "
          f"(about {ALPHA * len(rows):.0f} expected by chance if all were normal)")
    print()
    print(f"{'quantity':<9}{'comparison':<23}{'rejected':>9}{'cases':>7}"
          f"{'median |skew|':>15}{'median kurt':>13}")
    by = defaultdict(list)
    for r in rows:
        by[(r[3], r[2])].append(r)
    for (col, comp), rs in sorted(by.items()):
        print(f"{col:<9}{comp:<23}{sum(r[6] < ALPHA for r in rs):>9}{len(rs):>7}"
              f"{np.median([abs(r[7]) for r in rs]):>15.2f}{np.median([r[8] for r in rs]):>13.2f}")


if __name__ == "__main__":
    main()
