"""Shared definitions for the five-way analyses: the run table, the slow-run flag,
launch-level resampling, Wilson and BCa intervals, and one-plot-per-image figures.

Definitions fixed before any result was computed (see results/SUMMARY.md):
  q          timedTotal max over ranks / n, ms per iteration (per exchange for the
             benchmark and abcgspmv*, per CG iteration for abcgcg*)
  slow run   q > SLOW_FACTOR x the median q of its (nodes, pattern, arm) cell
  tie band   two arms within TIE (10%) of each other are tied
  unit       the launch; every interval resamples launches
"""
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGS = RESULTS / "figures"
FIGS_NOCAP = RESULTS / "figures_nocap"
TABLES = RESULTS / "tables"

ARMS = ["MPI", "STANDARD", "LOCALITY"]
NODES = [2, 4, 8, 16]
MATRICES = ["crankseg_2", "nd12k", "x104", "crankseg_1", "m_t1", "nd24k"]   # capture order
STAND_INS = ["pm", "fine", "coarse"]
REFERENCES = ["abcgspmv", "abcgcg", "abcgspmvnp", "abcgcgnp"]
KIND_LABEL = {"pm": "<m> (.pm)", "fine": "fine_", "coarse": "coarse_", "abcgspmv": "abcgspmv",
              "abcgcg": "abcgcg", "abcgspmvnp": "abcgspmvnp", "abcgcgnp": "abcgcgnp"}
HUMAN = {"pm": "the matrix replay (<m>)", "fine": "the fine Vernier replay", "coarse": "the coarse Vernier replay",
         "abcgspmv": "ABCG's SpMV exchange", "abcgcg": "ABCG's CG iteration",
         "abcgspmvnp": "ABCG's SpMV exchange (non-persistent)", "abcgcgnp": "ABCG's CG iteration (non-persistent)"}
ARM_COLOR ={"MPI": "#1f77b4", "STANDARD": "#ff7f0e", "LOCALITY": "#2ca02c"}

SLOW_FACTOR = 5.0
TIE = 0.10
B = 10_000
SEED = 20261014


def rng():
    return np.random.default_rng(SEED)


def load_runs():
    runs = pd.read_csv(RESULTS / "runs.csv")
    cell_median = runs.groupby(["nodes", "pattern", "arm"])["q_ms"].transform("median")
    runs["slow"] = runs["q_ms"] > SLOW_FACTOR * cell_median
    return runs


def q_table(runs, nodes, matrix, kinds, drop_slow=False):
    """Launch x (kind, arm) array of q, keeping only launches where every cell exists
    (complete pairs). Slow runs become missing when drop_slow, which drops the launch
    from this comparison."""
    sub = runs[(runs.nodes == nodes) & (runs.matrix == matrix) & runs.kind.isin(kinds)]
    if drop_slow:
        sub = sub[~sub.slow]
    wide = sub.pivot_table(index="launch", columns=["kind", "arm"], values="q_ms")
    cols = pd.MultiIndex.from_product([kinds, ARMS], names=["kind", "arm"])
    return wide.reindex(columns=cols).dropna()


def launch_resamples(L, b=B, seed_offset=0):
    """b x L indices drawing launches with replacement."""
    return np.random.default_rng(SEED + seed_offset).integers(0, L, size=(b, L))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def bca(values, statistic=np.median, level=0.95):
    """BCa interval of statistic over launches (values: one per launch)."""
    values = np.asarray(values, dtype=float)
    if np.ptp(values) == 0:
        return (values[0], values[0])
    res = stats.bootstrap((values,), statistic, n_resamples=B, confidence_level=level,
                          method="BCa", random_state=np.random.default_rng(SEED))
    return (res.confidence_interval.low, res.confidence_interval.high)


def percentile_interval(samples, level=0.95):
    a = (1 - level) / 2
    return tuple(np.nanquantile(samples, [a, 1 - a]))


def new_fig(title, xlabel, ylabel, caption=None, width=7.5, height=4.8):
    """title: a short plain-language headline. caption: what the reader is looking at
    (encodings, sample sizes, intervals), printed under the plot by save_fig."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.3)
    fig.caption = caption
    return fig, ax


def save_fig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    FIGS_NOCAP.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # The LaTeX report captions figures itself, so it uses a copy without the caption text.
    fig.savefig(FIGS_NOCAP / name, dpi=200, bbox_inches="tight", pad_inches=0.05)
    if getattr(fig, "caption", None):
        # Put the caption below everything already drawn (axis labels, legends set
        # below the axes), then let bbox_inches="tight" grow the image to hold it.
        fig.canvas.draw()
        bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
        width_chars = int(fig.get_figwidth() * 15)
        fig.text(0.01, bottom - 0.02, "\n".join(textwrap.wrap(fig.caption, width_chars)),
                 ha="left", va="top", fontsize=8.5, color="#333333")
    fig.savefig(FIGS / name, dpi=130, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def write_table(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False)
