"""Shared code for the paper's paired-comparison analyses (a11-a14).

A comparison is two series measured in the same 40 launches: two arms of one kind
(arm pairs), or the arm log ratio of two kinds (tracking). For arm pair (A, B) the
per-launch paired value is d = ln(q_A / q_B); d < 0 means A was faster in that launch.

Estimands (fixed before computing):
  median    median of d over launches: the typical launch
  mean      mean of d: what a paired t-test on ln q targets
  ratio of means  ln(mean q_A / mean q_B): expected cost, stalls included
  HL        Hodges-Lehmann pseudo-median of d: what Wilcoxon signed-rank targets
Stalls: in cells whose launch-to-launch differences line up on 0.160 s (coherence at
0.160 s > 0.5 in tables/stall_quantum.csv), a run's stall count is its excess timed
total over the cell's fastest launch in whole 0.160 s units; q_clean removes them.
A run is "stalled" if it has a stall count >= 1 or is a slow run (q > 5x cell median).

Tests are vectorized over rows: X has shape (draws, n) and each row is one sample.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import ARMS, MATRICES, NODES, TABLES, load_runs

KINDS = ["coarse", "fine", "pm", "abcgspmv", "abcgcg", "abcgspmvnp", "abcgcgnp"]
ARM_PAIRS = [("MPI", "STANDARD"), ("MPI", "LOCALITY"), ("STANDARD", "LOCALITY")]
QUANTUM_S = 0.160
COHERENT = 0.5
ALPHA = 0.05
REGIMES = ["none", "minority", "majority"]
REGIME_LABEL = {"none": "no stalled runs", "minority": "stalls in under half the launches",
                "majority": "stalls in over half the launches (either arm)"}


def runs_with_stalls():
    runs = load_runs()
    sq = pd.read_csv(TABLES / "stall_quantum.csv").set_index(["nodes", "pattern", "arm"])["coherence_at_0.160"]
    key = pd.MultiIndex.from_frame(runs[["nodes", "pattern", "arm"]])
    runs["coherent"] = sq.reindex(key).fillna(0).to_numpy() > COHERENT
    tmin = runs.groupby(["nodes", "pattern", "arm"]).timedTotal_max.transform("min")
    k = np.rint((runs.timedTotal_max - tmin) / 1000 / QUANTUM_S)
    runs["stalls"] = np.where(runs.coherent, k, 0).astype(int)
    runs["q_clean"] = (runs.timedTotal_max - runs.stalls * QUANTUM_S * 1000) / runs.n
    runs["stalled"] = (runs.stalls > 0) | runs.slow
    return runs


def launch_table(runs):
    """One row per launch: start and end (hours since the first launch) and concurrency (other
    launches of any node count overlapping it)."""
    t = runs.assign(s=pd.to_datetime(runs.start, utc=True), e=pd.to_datetime(runs.end, utc=True))
    L = t.groupby("launch").agg(nodes=("nodes", "first"), s=("s", "min"), e=("e", "max")).reset_index()
    t0 = L.s.min()
    L["start_h"] = (L.s - t0).dt.total_seconds() / 3600
    L["end_h"] = (L.e - t0).dt.total_seconds() / 3600
    s, e = L.start_h.to_numpy(), L.end_h.to_numpy()
    L["concurrency"] = ((s[:, None] < e[None, :]) & (s[None, :] < e[:, None])).sum(axis=1) - 1
    return L.drop(columns=["s", "e"])


def wide(runs, nodes, kinds, matrix, column):
    sub = runs[(runs.nodes == nodes) & (runs.matrix == matrix) & runs.kind.isin(kinds)]
    return sub.pivot_table(index="launch", columns=["kind", "arm"], values=column).sort_index()


def arm_comparisons(runs):
    """Every arm pair of every (nodes, kind, matrix): dicts with launch ids, qA, qB, d, the
    clean d, stall indicators, mean position, and the stall regime."""
    out = []
    for nodes in NODES:
        for matrix in MATRICES:
            q = wide(runs, nodes, KINDS, matrix, "q_ms")
            qc = wide(runs, nodes, KINDS, matrix, "q_clean")
            st = wide(runs, nodes, KINDS, matrix, "stalled")
            pos = wide(runs, nodes, KINDS, matrix, "position")
            for kind in KINDS:
                for a, b in ARM_PAIRS:
                    qa, qb = q[(kind, a)].to_numpy(), q[(kind, b)].to_numpy()
                    sa, sb = st[(kind, a)].to_numpy() > 0, st[(kind, b)].to_numpy() > 0
                    share = max(sa.mean(), sb.mean())
                    out.append({"nodes": nodes, "kind": kind, "matrix": matrix, "A": a, "B": b,
                                "launch": q.index.to_numpy(), "qA": qa, "qB": qb, "d": np.log(qa / qb),
                                "d_clean": np.log(qc[(kind, a)].to_numpy() / qc[(kind, b)].to_numpy()),
                                "stA": sa, "stB": sb,
                                "pos": (pos[(kind, a)].to_numpy() + pos[(kind, b)].to_numpy()) / 2,
                                "regime": "none" if share == 0 else ("majority" if share > 0.5 else "minority")})
    return out


def key_of(c):
    return {k: c[k] for k in ("nodes", "kind", "matrix", "A", "B", "regime")}


# ---- estimators ---------------------------------------------------------------------------

def hodges_lehmann(x):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i] + x[j]) / 2))


def hl_rows(X):
    n = X.shape[1]
    i, j = np.triu_indices(n)
    return np.median((X[:, i] + X[:, j]) / 2, axis=1)


# ---- tests (two-sided, rows are samples) ---------------------------------------------------

def t_p(X):
    n = X.shape[1]
    m, s = X.mean(axis=1), X.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (s / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    return np.where(s > 0, p, np.where(m == 0, 1.0, 0.0))


def sign_p(X):
    k = (X > 0).sum(axis=1)
    m = (X != 0).sum(axis=1)
    p = 2 * stats.binom.cdf(np.minimum(k, m - k), m, 0.5)
    return np.where(m > 0, np.minimum(p, 1.0), 1.0)


_WILCOXON_CDF = {}


def _wilcoxon_cdf(n):
    """P(W+ <= w) for w = 0..n(n+1)/2 under the exact null (no ties)."""
    if n not in _WILCOXON_CDF:
        counts = np.zeros(n * (n + 1) // 2 + 1)
        counts[0] = 1
        for r in range(1, n + 1):
            counts[r:] = counts[r:] + counts[:-r].copy()
        _WILCOXON_CDF[n] = np.cumsum(counts) / counts.sum()
    return _WILCOXON_CDF[n]


def wilcoxon_p(X):
    """Exact signed-rank test. Ties and zeros must already be broken (see jitter)."""
    n = X.shape[1]
    ranks = np.argsort(np.argsort(np.abs(X), axis=1), axis=1) + 1
    w = np.where(X > 0, ranks, 0).sum(axis=1)
    cdf = _wilcoxon_cdf(n)
    lower = cdf[w]
    upper = 1 - np.where(w > 0, cdf[np.maximum(w - 1, 0)], 0.0)
    return np.minimum(1.0, 2 * np.minimum(lower, upper))


def jitter(X, rng, scale):
    """Break ties and exact zeros (bootstrap draws repeat values) with noise far below any
    effect of interest; scale is the comparison's MAD of d."""
    return X + rng.normal(0.0, 1e-6 * max(scale, 1e-12), X.shape)


# ---- intervals ----------------------------------------------------------------------------

def order_stat_index(n, level=0.95):
    """Distribution-free interval for the median: (x_(l), x_(n-l+1)), 1-based, with the
    largest l whose coverage is still >= level. Returns (l, coverage); l = 1 with coverage
    below level when n is too small to reach it (n = 5: min-max covers 93.75%)."""
    a = (1 - level) / 2
    l = 1
    for j in range(1, n // 2 + 1):
        if stats.binom.cdf(j - 1, n, 0.5) <= a:
            l = j
    cov = 1 - 2 * stats.binom.cdf(l - 1, n, 0.5)
    return l, cov


def order_stat_ci(X, level=0.95):
    n = X.shape[1]
    l, _ = order_stat_index(n, level)
    S = np.sort(X, axis=1)
    return S[:, l - 1], S[:, n - l]


def t_ci(X, level=0.95):
    n = X.shape[1]
    m, s = X.mean(axis=1), X.std(axis=1, ddof=1)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * s / np.sqrt(n)
    return m - h, m + h


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def verdict(p, est, alpha=ALPHA):
    """+1: B faster (d > 0), -1: A faster, 0: no decision."""
    return np.where(p < alpha, np.sign(est), 0).astype(int)
