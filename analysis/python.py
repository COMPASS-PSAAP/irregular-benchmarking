"""Amortization bounds for the irregular-benchmarking launches.

Implements the decision method of the mpi-performance-modeling-statistics README
on the output of scripts/gatherdata.sbatch:

  data/<jobid>-gatherdata-<nodes>/run_*.txt   one benchmark run each (-r output)
  data/<jobid>-gatherdata-<nodes>/manifest.csv  exit status per run, when present

Each job folder is one launch. For every launch, pattern, and comparison of an
optimization against a baseline it forms

  A = W_base - W_opt        startup saved (ms)
  d = b_base - b_opt        per-iteration saving (ms)
  D(N) = A + (N - k) d - f (W_base + (N - k) b_base)

and across launches computes four one-sided lower bounds on D(N) (t bound on
the mean, sign-test bound on the median, normal tolerance bound, order-statistic
bound). N* is the smallest N >= k at which a bound is positive. Every node count
is analyzed on its own.

Usage: python python.py [DATA_DIR [RESULTS_DIR]]
       (defaults: data/ and results/ next to this file)
Writes results/launches.csv, results/nstar.csv, results/bounds.csv,
results/dropped.txt, and one PNG per chart under results/graphs/ (see plot_results()).
"""
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASELINE_COMPARISONS = [          # (optimization, baseline)
    ("STANDARD", "MPI"),
    ("LOCALITY", "MPI"),
    ("LOCALITY", "STANDARD"),
]
# True times startup from the setup call, as the README does: halo build + gather
# setup + the k-apply window. False (the default) uses the k-apply window alone:
# under -W nearest the halo build carries ~200 ms of connection setup that is the
# same for every arm but varies by ~250 ms between runs, which buries the real
# startup differences.
STARTUP_INCLUDES_SETUP = False
MARGINS = [0.0, 0.05, 0.10, 0.20]   # f, as a fraction of the baseline's cost
N_MAX = 1e7                         # largest hold length evaluated
N_POINTS = 400                      # log-spaced hold lengths from k to N_MAX
CONFIDENCE = 0.95
COVERAGE = 0.90                     # fraction of launches the tolerance bounds cover
# The bound whose N* the scaling graph shows: the README's decision functional is a
# quantile over launches, which the normal tolerance bound targets.
SUMMARY_PROCEDURE = "normal_tolerance"

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PROCEDURES = ["t_mean", "sign_median", "normal_tolerance", "order_statistic"]


def parse_run(path):
    """One run file -> dict with arm, pattern, k, n and max-over-ranks timings (ms)."""
    run = {"file": path.name}
    comm = alltoallv = None
    metrics = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("-MPI:"):
            comm = line.split(":", 1)[1].strip()
        elif line.startswith("-alltoallv:"):
            alltoallv = line.split(":", 1)[1].strip()
        elif line.startswith("-window (k):"):
            run["k"] = int(line.split(":", 1)[1])
        elif line.startswith("-timed iterations (n):"):
            run["n"] = int(line.split(":", 1)[1])
        else:
            # "(pattern_2)  haloTime   min   max   avg"
            m = re.match(r"^\((.+)\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
            if m:
                run["pattern"] = m.group(1)
                try:
                    metrics[m.group(2)] = float(m.group(4))
                except ValueError:
                    pass
    if comm == "MPI":
        run["arm"] = "MPI"
    elif comm == "MPIADVANCE" and alltoallv:
        run["arm"] = alltoallv.rsplit("_", 1)[-1]  # ..._STANDARD / ..._LOCALITY
    for name in ("haloTime", "gatherTime", "startupWindow", "timedTotal"):
        if name in metrics:
            run[name] = metrics[name]
    return run


def read_manifest(folder):
    """file name -> exit status, from manifest.csv if the job wrote one."""
    path = folder / "manifest.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["file"]: row["exit"] for row in csv.DictReader(f)}


def load_launches(data_dir):
    """-> {nodes: {launch: {pattern: {arm: (k, W, b)}}}}, plus discard notes."""
    launches = defaultdict(dict)
    notes = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        m = re.match(r"^(\d+)-.+-(\d+)$", folder.name)
        if not m:
            continue
        launch, nodes = m.group(1), int(m.group(2))
        run_files = sorted(folder.glob("run_*.txt"))
        if not run_files:
            continue  # e.g. a pilot folder: not a launch
        exits = read_manifest(folder)
        per_arm = defaultdict(list)
        for path in run_files:
            run = parse_run(path)
            needed = ("arm", "pattern", "k", "n", "startupWindow", "timedTotal")
            if STARTUP_INCLUDES_SETUP:
                needed += ("haloTime", "gatherTime")
            missing = [key for key in needed if key not in run]
            if exits.get(path.name, "0") != "0" or missing:
                notes.append(f"{folder.name}/{path.name}: dropped "
                             f"(exit {exits.get(path.name, '?')}, missing {missing})")
                continue
            startup = run["startupWindow"]
            if STARTUP_INCLUDES_SETUP:
                startup += run["haloTime"] + run["gatherTime"]
            per_arm[(run["pattern"], run["arm"])].append(
                (run["k"], startup, run["timedTotal"] / run["n"]))
        patterns = defaultdict(dict)
        for (pattern, arm), samples in per_arm.items():
            ks = {s[0] for s in samples}
            if len(ks) != 1:
                notes.append(f"{folder.name}: {pattern} {arm} has runs with different k {ks}; dropped")
                continue
            # Several runs of one pair in one launch are within-launch replicates: average them.
            patterns[pattern][arm] = (ks.pop(),
                                      float(np.mean([s[1] for s in samples])),
                                      float(np.mean([s[2] for s in samples])))
        launches[nodes][launch] = patterns
    return launches, notes


def bound_setup(L):
    """Constants that depend only on the launch count L (None where a bound does not exist)."""
    alpha = 1.0 - CONFIDENCE
    t_q = stats.t.ppf(CONFIDENCE, L - 1) if L >= 2 else None
    # Sign test: j-th smallest bounds the median when P(Binomial(L, 1/2) <= j - 1) <= alpha.
    js = [j for j in range(1, L + 1) if stats.binom.cdf(j - 1, L, 0.5) <= alpha]
    # One-sided normal tolerance factor for COVERAGE at CONFIDENCE.
    k_tol = (stats.nct.ppf(CONFIDENCE, L - 1, stats.norm.ppf(COVERAGE) * math.sqrt(L)) / math.sqrt(L)
             if L >= 2 else None)
    # Order statistic: r-th smallest bounds the (1 - COVERAGE) quantile when
    # P(Binomial(L, 1 - COVERAGE) <= r - 1) <= alpha.
    rs = [r for r in range(1, L + 1) if stats.binom.cdf(r - 1, L, 1.0 - COVERAGE) <= alpha]
    return t_q, (max(js) if js else None), k_tol, (max(rs) if rs else None)


def lower_bounds(D, setup):
    """D: (L, len(N)) launch statistics -> {procedure: bound per N}."""
    t_q, j, k_tol, r = setup
    L = D.shape[0]
    m = D.mean(axis=0)
    s = D.std(axis=0, ddof=1) if L >= 2 else np.full(D.shape[1], np.nan)
    ordered = np.sort(D, axis=0)
    nan = np.full(D.shape[1], np.nan)
    return {
        "t_mean": m - t_q * s / math.sqrt(L) if t_q is not None else nan,
        "sign_median": ordered[j - 1] if j is not None else nan,
        "normal_tolerance": m - k_tol * s if k_tol is not None else nan,
        "order_statistic": ordered[r - 1] if r is not None else nan,
    }


def main():
    global RESULTS_DIR, GRAPHS_DIR
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
    if len(sys.argv) > 2:
        RESULTS_DIR = Path(sys.argv[2])
        GRAPHS_DIR = RESULTS_DIR / "graphs"
    launches, notes = load_launches(data_dir)
    RESULTS_DIR.mkdir(exist_ok=True)

    launch_rows, nstar_rows, bound_rows = [], [], []
    for nodes in sorted(launches):
        by_launch = launches[nodes]
        patterns = sorted({p for pats in by_launch.values() for p in pats})
        for pattern in patterns:
            for opt, base in BASELINE_COMPARISONS:
                rows = []  # (launch, k, W_base, b_base, A, d)
                for launch, pats in sorted(by_launch.items()):
                    arms = pats.get(pattern, {})
                    if opt not in arms or base not in arms:
                        notes.append(f"{nodes} nodes, launch {launch}, {pattern}: "
                                     f"{opt} vs {base} dropped (arm missing)")
                        continue
                    (k_o, W_o, b_o), (k_b, W_b, b_b) = arms[opt], arms[base]
                    if k_o != k_b:
                        notes.append(f"{nodes} nodes, launch {launch}, {pattern}: k differs; dropped")
                        continue
                    rows.append((launch, k_b, W_b, b_b, W_b - W_o, b_b - b_o))
                    launch_rows.append([nodes, pattern, f"{opt} vs {base}", launch, k_b,
                                        W_b, b_b, W_b - W_o, b_b - b_o])
                if not rows:
                    continue
                k = max(r[1] for r in rows)
                N = np.unique(np.round(np.geomspace(k, N_MAX, N_POINTS)))
                L = len(rows)
                setup = bound_setup(L)
                W = np.array([r[2] for r in rows])[:, None]
                b = np.array([r[3] for r in rows])[:, None]
                A = np.array([r[4] for r in rows])[:, None]
                d = np.array([r[5] for r in rows])[:, None]
                baseline_cost = (W + (N - k) * b).mean(axis=0)
                for f in MARGINS:
                    D = A + (N - k) * d - f * (W + (N - k) * b)
                    bounds = lower_bounds(D, setup)
                    for i, n_val in enumerate(N):
                        bound_rows.append([nodes, pattern, f"{opt} vs {base}", f, L, int(n_val),
                                           baseline_cost[i]] + [bounds[p][i] for p in PROCEDURES])
                    for p in PROCEDURES:
                        positive = np.nonzero(bounds[p] > 0)[0]
                        if np.all(np.isnan(bounds[p])):
                            n_star = "undefined"   # too few launches for this bound
                        elif positive.size:
                            n_star = int(N[positive[0]])
                        else:
                            n_star = "never"
                        nstar_rows.append([nodes, pattern, f"{opt} vs {base}", p, f, L, n_star])

    write_csv(RESULTS_DIR / "launches.csv",
              ["nodes", "pattern", "comparison", "launch", "k", "W_base_ms", "b_base_ms", "A_ms", "d_ms"],
              launch_rows)
    write_csv(RESULTS_DIR / "nstar.csv",
              ["nodes", "pattern", "comparison", "procedure", "f", "launches", "N_star"], nstar_rows)
    write_csv(RESULTS_DIR / "bounds.csv",
              ["nodes", "pattern", "comparison", "f", "launches", "N", "mean_baseline_cost_ms"]
              + PROCEDURES, bound_rows)
    (RESULTS_DIR / "dropped.txt").write_text("\n".join(notes) + ("\n" if notes else ""))

    print(f"data: {data_dir}")
    print(f"startup includes setup: {STARTUP_INCLUDES_SETUP}")
    print(f"dropped: {len(notes)} (see {RESULTS_DIR / 'dropped.txt'})")
    print()
    print(f"N* at f = {MARGINS[0]:g} (smallest N >= k with a positive bound)")
    print(f"{'nodes':>5}  {'pattern':<12} {'comparison':<22} {'L':>3}  "
          + "  ".join(f"{p:>16}" for p in PROCEDURES))
    table = defaultdict(dict)
    for nodes, pattern, comp, p, f, L, n_star in nstar_rows:
        if f == MARGINS[0]:
            table[(nodes, pattern, comp, L)][p] = n_star
    for (nodes, pattern, comp, L), row in table.items():
        print(f"{nodes:>5}  {pattern:<12} {comp:<22} {L:>3}  "
              + "  ".join(f"{str(row[p]):>16}" for p in PROCEDURES))

    written = plot_results(launch_rows, nstar_rows, bound_rows)
    print()
    print(f"graphs: {len(written)} PNG files in {RESULTS_DIR / 'graphs'}")


# Chart styling: a validated categorical palette in fixed slot order, thin marks,
# recessive grid, text in ink colors (never the series color). Every series also
# has its own line style or marker, so color is never the only cue.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
PROC_STYLE = {p: (SLOTS[i], ls) for i, (p, ls) in enumerate(zip(PROCEDURES, ["-", "--", "-.", ":"]))}
PROC_LABEL = {"t_mean": "t bound (mean)", "sign_median": "sign test (median)",
              "normal_tolerance": "normal tolerance (90%)", "order_statistic": "order statistic (90%)"}
COMP_NAMES = [f"{opt} vs {base}" for opt, base in BASELINE_COMPARISONS]
COMP_STYLE = {c: (SLOTS[i], m) for i, (c, m) in enumerate(zip(COMP_NAMES, ["o", "s", "^"]))}
GRAPHS_DIR = RESULTS_DIR / "graphs"


def new_chart(title, xlabel, ylabel):
    """One chart per figure, sized to read on a normal screen."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=11)
    ax.set_title(title, color=INK, fontsize=14, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=12)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=12)
    return fig, ax


def plot_nstar(ax, xs, values, color, style, marker, label):
    """N* on a log axis; 'never' sits on its own row above N_MAX as a hollow marker."""
    pts = [(x, v) for x, v in zip(xs, values) if v != "undefined"]
    if not pts:
        return
    ys = [N_MAX * 2 if v == "never" else v for _, v in pts]
    ax.plot([x for x, _ in pts], ys, color=color, linestyle=style, linewidth=2,
            marker=marker, markersize=8, markeredgecolor=SURFACE, label=label)
    never = [(x, y) for (x, v), y in zip(pts, ys) if v == "never"]
    if never:
        ax.plot(*zip(*never), linestyle="none", marker=marker, markersize=11,
                markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=2)


def finish_nstar_axis(ax):
    """Same range on every N* chart, with 'never' as the top tick above a separator."""
    ax.set_yscale("log")
    decades = [10.0 ** e for e in range(1, int(math.log10(N_MAX)) + 1)]
    ax.set_yticks(decades + [N_MAX * 2],
                  [f"$10^{{{int(math.log10(v))}}}$" for v in decades] + ["never"])
    ax.minorticks_off()
    ax.set_ylim(5, N_MAX * 4)
    ax.axhline(N_MAX * 1.4, color=BASELINE, linewidth=1)


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def save_chart(fig, ax, subdir, name):
    ax.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="best")
    fig.tight_layout()
    path = GRAPHS_DIR / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_results(launch_rows, nstar_rows, bound_rows):
    """Write one chart per PNG under results/graphs/ and return their paths.

    nstar_vs_nodes/<pattern>.png                 N* (SUMMARY_PROCEDURE, f = 0) against
                                                 node count, one line per comparison
    bounds/<n>nodes_<pattern>_<comparison>.png   the four lower bounds on the saving,
                                                 as a fraction of the baseline's time,
                                                 against hold length N (f = 0)
    nstar_vs_f/<n>nodes_<pattern>_<comparison>.png  N* of each bound against margin f
    launches/<n>nodes_<pattern>_d.png and _A.png     one dot per launch, per comparison
    """
    written = []
    nodes_list = sorted({r[0] for r in nstar_rows})
    patterns = sorted({r[1] for r in nstar_rows})
    nstar = {(r[0], r[1], r[2], r[3], r[4]): r[6] for r in nstar_rows}

    for pattern in patterns:
        fig, ax = new_chart(f"{pattern}: N* against node count "
                            f"({PROC_LABEL[SUMMARY_PROCEDURE]}, f = {MARGINS[0]:g})",
                            "nodes", "N* (iterations)")
        for comp in COMP_NAMES:
            color, marker = COMP_STYLE[comp]
            plot_nstar(ax, nodes_list,
                       [nstar.get((n, pattern, comp, SUMMARY_PROCEDURE, MARGINS[0]), "undefined")
                        for n in nodes_list], color, "-", marker, comp)
        finish_nstar_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(nodes_list, [str(n) for n in nodes_list])
        written.append(save_chart(fig, ax, "nstar_vs_nodes", f"{slug(pattern)}.png"))

    for nodes in nodes_list:
        for pattern in patterns:
            for comp in COMP_NAMES:
                sel = [r for r in bound_rows
                       if r[0] == nodes and r[1] == pattern and r[2] == comp and r[3] == MARGINS[0]]
                if sel:
                    fig, ax = new_chart(f"{nodes} nodes, {pattern}, {comp}: lower bound on the saving",
                                        "hold length N (iterations)",
                                        "saving / baseline time (0 = break even)")
                    N = np.array([r[5] for r in sel])
                    cost = np.array([r[6] for r in sel])
                    for k_idx, p in enumerate(PROCEDURES):
                        y = np.array([r[7 + k_idx] for r in sel], dtype=float) / cost
                        if np.all(np.isnan(y)):
                            continue
                        color, style = PROC_STYLE[p]
                        ax.plot(N, y, color=color, linestyle=style, linewidth=2.2, label=PROC_LABEL[p])
                    ax.axhline(0, color=INK_2, linewidth=1)
                    ax.set_xscale("log")
                    written.append(save_chart(fig, ax, "bounds",
                                              f"{nodes}nodes_{slug(pattern)}_{slug(comp)}.png"))

                if any((nodes, pattern, comp, p, MARGINS[0]) in nstar for p in PROCEDURES):
                    fig, ax = new_chart(f"{nodes} nodes, {pattern}, {comp}: N* against margin f",
                                        "margin f (fraction of the baseline's time)", "N* (iterations)")
                    for p in PROCEDURES:
                        color, style = PROC_STYLE[p]
                        plot_nstar(ax, MARGINS, [nstar.get((nodes, pattern, comp, p, f), "undefined")
                                                 for f in MARGINS], color, style, "o", PROC_LABEL[p])
                    finish_nstar_axis(ax)
                    ax.set_xticks(MARGINS)
                    written.append(save_chart(fig, ax, "nstar_vs_f",
                                              f"{nodes}nodes_{slug(pattern)}_{slug(comp)}.png"))

            # Per-launch d and A: two measures on different scales, so two charts.
            for col, short, label in ((8, "d", "d: per-iteration saving (ms)"),
                                      (7, "A", "A: startup saved (ms)")):
                groups = [(comp, [r[col] for r in launch_rows
                                  if r[0] == nodes and r[1] == pattern and r[2] == comp])
                          for comp in COMP_NAMES]
                if not any(vals for _, vals in groups):
                    continue
                fig, ax = new_chart(f"{nodes} nodes, {pattern}: {label.split(':')[0]} per launch",
                                    "comparison", label)
                for c_idx, (comp, vals) in enumerate(groups):
                    color, marker = COMP_STYLE[comp]
                    x = c_idx + np.linspace(-0.12, 0.12, len(vals))
                    ax.plot(x, vals, linestyle="none", marker=marker, markersize=8, color=color,
                            markeredgecolor=SURFACE, markeredgewidth=0.8, label=comp)
                ax.axhline(0, color=INK_2, linewidth=1)
                ax.set_xticks(range(len(COMP_NAMES)), COMP_NAMES)
                ax.set_xlim(-0.5, len(COMP_NAMES) - 0.5)
                written.append(save_chart(fig, ax, "launches",
                                          f"{nodes}nodes_{slug(pattern)}_{short}.png"))
    return written


def write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
