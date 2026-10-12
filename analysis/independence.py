"""Launch-independence check for the per-launch statistics behind the bounds in python.py.

METHODOLOGY.md leaves launch-level independence as the one assumption to check, "by
examining the launch contrasts against wall-clock start and against node-set overlap".
For every node count, pattern, comparison, and quantity (A and d) in
RESULTS_DIR/launches.csv this computes:

  drift     Spearman rho of the contrast against launch start time (two-sided p).
  lag1      lag-1 autocorrelation of the contrast in start order; a permutation p for
            positive autocorrelation (neighbors in time look alike).
  mantel    correlation between node-set overlap (Jaccard) and |x_i - x_j| over launch
            pairs; a permutation p for negative correlation (shared nodes look alike).

Start time and node list come from each launch's provenance file in DATA_DIR. Cells at
one node count share their launches, so a single odd launch can reject many cells at
once; the rejection counts are a diagnostic, not a binomial test.

Usage: python independence.py [DATA_DIR [RESULTS_DIR]]   (defaults: data/, results/)
Writes RESULTS_DIR/independence.csv and RESULTS_DIR/independence/*.png.
"""
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ALPHA = 0.05
PERMUTATIONS = 9999
SEED = 20261006
QUANTITIES = [("d_ms", "d: per-iteration saving (ms)"), ("A_ms", "A: startup saved (ms)")]

# Same chart styling as python.py and qq.py.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
POINT = "#2a78d6"

HERE = Path(__file__).resolve().parent


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def read_provenance(data_dir):
    """launch id -> (start datetime, frozenset of node names)."""
    out = {}
    for path in data_dir.glob("*-gatherdata-*/provenance_*[0-9].txt"):
        start = nodes = None
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("job start:"):
                start = datetime.fromisoformat(line.split(":", 1)[1].strip())
            elif line.startswith("nodes:"):
                nodes = frozenset(line.split(":", 1)[1].split())
        launch = path.parent.name.split("-")[0]
        if start is None or not nodes:
            print(f"warning: {path} lacks job start or nodes; launch {launch} skipped")
            continue
        out[launch] = (start, nodes)
    return out


def jaccard_matrix(node_sets):
    L = len(node_sets)
    J = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            a, b = node_sets[i], node_sets[j]
            J[i, j] = J[j, i] = len(a & b) / len(a | b)
    return J


def lag1(x):
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def lag1_test(x, rng):
    """x in start order -> (r1, p for r1 this large or larger under random order)."""
    r = lag1(x)
    perm = np.array([lag1(rng.permutation(x)) for _ in range(PERMUTATIONS)])
    return r, (1 + np.sum(perm >= r)) / (PERMUTATIONS + 1)


def mantel_test(x, overlap_pairs, iu, ju, rng):
    """-> (r, p for r this negative or more) between overlap and |x_i - x_j| over pairs."""
    o = overlap_pairs - overlap_pairs.mean()
    if not np.any(o):
        return float("nan"), float("nan")  # no pair shares a node: nothing to test

    def corr(diffs):  # diffs: (P, pairs)
        d = diffs - diffs.mean(axis=1, keepdims=True)
        return (d @ o) / (np.linalg.norm(d, axis=1) * np.linalg.norm(o))

    r = corr(np.abs(x[iu] - x[ju])[None, :])[0]
    xp = np.array([rng.permutation(x) for _ in range(PERMUTATIONS)])
    perm = corr(np.abs(xp[:, iu] - xp[:, ju]))
    return float(r), (1 + np.sum(perm <= r)) / (PERMUTATIONS + 1)


def time_chart(nodes, comp, label, series, path):
    """series: [(pattern, hours since first start, values)], one panel per pattern."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE, sharex=True)
    for ax, (pattern, hours, values, rho, p) in zip(axes.flat, series):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=10)
        ax.plot(hours, values, linestyle="none", marker="o", markersize=7, color=POINT,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_title(f"{pattern}   rho = {rho:.2f}, p = {p:.2g}", color=INK, fontsize=11,
                     loc="left")
    for ax in axes.flat[len(series):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("hours since the first launch started", color=INK_2, fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(label, color=INK_2, fontsize=11)
    fig.suptitle(f"{nodes} nodes, {comp}: contrast against launch start", color=INK,
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
    results = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results"
    prov = read_provenance(data_dir)
    rng = np.random.default_rng(SEED)

    groups = defaultdict(dict)  # (nodes, pattern, comparison) -> launch -> row
    with (results / "launches.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            groups[(int(row["nodes"]), row["pattern"], row["comparison"])][row["launch"]] = row

    missing = sorted({launch for g in groups.values() for launch in g if launch not in prov})
    if missing:
        print(f"warning: no provenance for {len(missing)} launches, left out: {missing}")

    print("node-set overlap between launches, per node count:")
    for nodes in sorted({k[0] for k in groups}):
        launches = sorted({l for k, g in groups.items() if k[0] == nodes for l in g if l in prov})
        J = jaccard_matrix([prov[l][1] for l in launches])
        pairs = J[np.triu_indices(len(launches), 1)]
        print(f"  {nodes:>2} nodes: {len(launches)} launches, {len(pairs)} pairs, "
              f"{np.mean(pairs > 0):.0%} share a node, mean Jaccard {pairs.mean():.3f}, "
              f"max {pairs.max():.2f}")
    print()

    rows = []
    charts = defaultdict(list)  # (nodes, comparison, quantity) -> panels
    for (nodes, pattern, comp), by_launch in sorted(groups.items()):
        launches = sorted((l for l in by_launch if l in prov), key=lambda l: (prov[l][0], int(l)))
        if len(launches) < 4:
            continue
        starts = np.array([prov[l][0].timestamp() for l in launches])
        J = jaccard_matrix([prov[l][1] for l in launches])
        iu, ju = np.triu_indices(len(launches), 1)
        for col, label in QUANTITIES:
            x = np.array([float(by_launch[l][col]) for l in launches])
            rho, p_rho = stats.spearmanr(starts, x)
            r1, p_r1 = lag1_test(x, rng)
            rm, p_m = mantel_test(x, J[iu, ju], iu, ju, rng)
            rows.append([nodes, pattern, comp, col, len(x), rho, p_rho, r1, p_r1, rm, p_m])
            charts[(nodes, comp, col, label)].append(
                (pattern, (starts - starts.min()) / 3600, x, rho, p_rho))

    for (nodes, comp, col, label), series in charts.items():
        time_chart(nodes, comp, label, series,
                   results / "independence" / f"{nodes}nodes_{slug(comp)}_{col[0]}.png")

    with (results / "independence.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nodes", "pattern", "comparison", "quantity", "launches",
                         "drift_spearman_rho", "drift_p", "lag1_r", "lag1_p",
                         "mantel_r", "mantel_p"])
        writer.writerows(rows)

    tests = [("drift", 6), ("lag1", 8), ("mantel", 10)]
    print(f"{len(rows)} cases from {results / 'launches.csv'}; rejections at p < {ALPHA} "
          f"(chance column: rejections expected per test if launches were independent)")
    print()
    print(f"{'nodes':>5}  {'quantity':<9}{'cases':>6}{'chance':>8}" + "".join(f"{t:>9}" for t, _ in tests))
    by = defaultdict(list)
    for r in rows:
        by[(r[0], r[3])].append(r)
    for (nodes, col), rs in sorted(by.items()):
        counts = []
        for _, i in tests:
            tested = [r for r in rs if not np.isnan(r[i])]
            counts.append(f"{sum(r[i] < ALPHA for r in tested)}/{len(tested)}")
        print(f"{nodes:>5}  {col:<9}{len(rs):>6}{ALPHA * len(rs):>8.1f}" + "".join(f"{c:>9}" for c in counts))


if __name__ == "__main__":
    main()
