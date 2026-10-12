#!/usr/bin/env python3
"""How closely does the replayed pattern rhyme with the real SpMV halo?

irregular-benchmarking reads input/t<nodes>.json and replays a communication
pattern from it. This measures how close that pattern is to the halo an SpMV
over the real matrix actually performs, without building the benchmark: the
true pattern is computed from the matrix, the config is read, and the
benchmark's sampling loop is reproduced in numpy.

Two levels are compared, because they fail for different reasons:

  config vs truth   are the histograms in the JSON a faithful summary of the
                    real pattern?
  replay vs truth   does drawing from those histograms reproduce it?

What the benchmark does, from irregular-benchmarking/src/benchmark.cpp:105-130.
Per rank: draw a neighbour count k from comm_partners; then k times draw a
buffer size from buffer_size and an offset from dist_to_neighbors[k],
independently; place each neighbour at (offset + rank + size) % size.

Three things follow that no distance between marginals can see:

  independence   the three quantities are drawn separately, so any correlation
                 in the real pattern between how many partners a rank has, how
                 much it sends each, and how far away they are, is gone.
  no merging     seen_neighbors is built and never read, so drawing the same
                 offset twice sends two messages where a real SpMV coalesces
                 them into one. The realised distinct-partner count is below k.
  asymmetry      each rank picks its own targets, so A can export to B while B
                 does not export to A. A symmetric matrix gives a symmetric
                 halo, so this is structural, not noise.

Also: export_ids is 0..n-1 for every neighbour, so a rank sends the same local
indices to all partners. That is the maximum possible index overlap, the best
case for deduplication, and not what a real halo looks like.

--distinct-neighbors switches the simulated sampling to draw offsets without
replacement, which is the proposed fix: it makes the realised partner count
equal the drawn k instead of falling short by the duplicates. Without it, the
simulation reproduces what the benchmark does today.

Metrics: cosine similarity on the histograms, Wasserstein-1 in the units of the
quantity, and total variation. Cosine is reported because it was asked for and
read with care, since it is dominated by the largest bins and can sit at 0.99
while the tail, where the max partner count lives, is wrong. The tail scalars
are compared directly rather than folded into a distance.

Outputs, written into the PythonProject3 folder:
  replay_similarity.csv     every metric per (matrix, nodes, quantity, level)
  replay_similarity.png     cosine and Wasserstein per quantity
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from halo_pattern import DOUBLE_BYTES, analyze, read_pattern  # noqa: E402

PROJECT = Path("/mnt/c/Users/Nicholas/PycharmProjects/PythonProject3")
INPUT_DIR = PROJECT / "input"
OUT_DIR = PROJECT / "data-9-17-2026"
PATTERN_TO_MATRIX = {
    0: "crankseg_2", 1: "nd12k", 2: "x104",
    3: "crankseg_1", 4: "m_t1", 5: "nd24k",
}
RANKS_PER_NODE = 112
QUANTITIES = ("partners", "buffer_size", "offsets")


def stoi(text):
    """int() the way C++ std::stoi does: stop at the first non-digit."""
    head = text.strip().split(".")[0]
    return int(head) if head not in ("", "-") else 0


def read_config(path):
    """Per pattern: the three histograms, keys truncated as the C++ sees them."""
    raw = json.loads(path.read_text())
    out = {}
    for name, body in raw.items():
        index = int(name.rsplit("_", 1)[1])
        partners, sizes = {}, {}
        for key, count in body["comm_partners"].items():
            partners[stoi(key)] = partners.get(stoi(key), 0) + int(count)
        for key, count in body["buffer_size"].items():
            sizes[stoi(key)] = sizes.get(stoi(key), 0) + int(count)
        offsets = {}
        for outer, inner in body["dist_to_neighbors"].items():
            per_k = {}
            for key, count in inner.items():
                per_k[stoi(key)] = per_k.get(stoi(key), 0) + int(count)
            offsets[stoi(outer)] = per_k
        out[PATTERN_TO_MATRIX[index]] = {
            "partners": partners, "buffer_size": sizes, "offsets": offsets,
        }
    return out


def histogram(values):
    keys, counts = np.unique(np.asarray(values), return_counts=True)
    return dict(zip(keys.tolist(), counts.tolist()))


def as_vectors(left, right):
    """Two histograms onto one sorted support, as (support, weights, weights)."""
    support = np.array(sorted(set(left) | set(right)), dtype=np.float64)
    a = np.array([left.get(int(v), 0) for v in support], dtype=np.float64)
    b = np.array([right.get(int(v), 0) for v in support], dtype=np.float64)
    return support, a, b


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def total_variation(a, b):
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def wasserstein1(support, a, b):
    """Earth-mover distance between two discrete distributions, in units of x."""
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    ca = np.cumsum(a / a.sum())
    cb = np.cumsum(b / b.sum())
    gaps = np.diff(support)
    return float(np.sum(np.abs(ca[:-1] - cb[:-1]) * gaps))


def rebin_to(hist, bins):
    """Pool hist into the given bin lower edges, the way the config was binned.

    The config stores buffer sizes in 17 bins spaced 58 bytes apart while the
    true sizes are multiples of 8, so the two histograms share almost no
    support and a cosine between them is near zero for that reason alone
    rather than because they disagree. Rebinning the truth onto the config's
    edges makes the comparison mean something.
    """
    edges = np.array(sorted(bins), dtype=np.float64)
    out = {int(e): 0 for e in edges}
    for value, count in hist.items():
        index = int(np.searchsorted(edges, value, side="right")) - 1
        out[int(edges[max(index, 0)])] += count
    return out


def compare(left, right):
    support, a, b = as_vectors(left, right)
    return {
        "cosine": cosine(a, b),
        "wasserstein": wasserstein1(support, a, b),
        "total_variation": total_variation(a, b),
    }


def weighted_percentile(hist, q):
    keys = np.array(sorted(hist), dtype=np.float64)
    weights = np.array([hist[int(k)] for k in keys], dtype=np.float64)
    cdf = np.cumsum(weights) / weights.sum()
    index = int(np.searchsorted(cdf, q / 100.0, side="left"))
    return float(keys[min(index, keys.size - 1)])


def tail_scalars(hist):
    keys = np.array(sorted(hist), dtype=np.float64)
    return {
        "max": float(keys.max()),
        "p99": weighted_percentile(hist, 99),
        "p90": weighted_percentile(hist, 90),
        "median": weighted_percentile(hist, 50),
    }


def draw_offsets(table, k, rows, p, rng, replace):
    """Draw k offsets per row from a weighted table, reduced modulo p.

    With replace=False this is Efraimidis-Spirakis weighted sampling without
    replacement: give each candidate the key u^(1/w) and take the top k, which
    needs no rejection and cannot fail to terminate. Offsets congruent modulo p
    are collapsed first, since they name the same target rank.
    """
    keys = np.array(sorted(table), dtype=np.int64)
    weights = np.array([table[int(o)] for o in keys], dtype=np.float64)
    if replace:
        probs = weights / weights.sum()
        return np.mod(rng.choice(keys, size=(rows, int(k)), p=probs), p), int(k)
    targets, inverse = np.unique(np.mod(keys, p), return_inverse=True)
    pooled = np.zeros(targets.size)
    np.add.at(pooled, inverse, weights)
    reach = int(targets.size)
    take = min(int(k), reach)
    # log u / w ranks the same way as u^(1/w) and avoids underflow
    score = np.log(rng.random((rows, reach))) / pooled
    chosen = np.argpartition(-score, take - 1, axis=1)[:, :take]
    return targets[chosen], reach


def simulate_replay(config, p, repeats, rng, replace=True):
    """Reproduce benchmark.cpp's per-rank sampling; return realised histograms.

    A rank's distinct-partner count depends only on how many of its k drawn
    offsets are distinct modulo p, since (o + r) % p collides for two offsets
    exactly when they are congruent mod p. So the rank id never enters.

    replace=True is what the benchmark does today. replace=False is the
    --distinct-neighbors behaviour: no duplicate partners, so the realised
    count equals the drawn k whenever the offset table can reach that many
    distinct targets.
    """
    partners = config["partners"]
    sizes = config["buffer_size"]
    offsets = config["offsets"]

    k_keys = np.array(sorted(partners), dtype=np.int64)
    k_probs = np.array([partners[int(k)] for k in k_keys], dtype=np.float64)
    k_probs /= k_probs.sum()

    size_keys = np.array(sorted(sizes), dtype=np.int64)
    size_probs = np.array([sizes[int(s)] for s in size_keys], dtype=np.float64)
    size_probs /= size_probs.sum()

    realised_partners = []
    realised_self = 0
    total_draws = 0
    short = 0
    for _ in range(repeats):
        drawn_k = rng.choice(k_keys, size=p, p=k_probs)
        for k in np.unique(drawn_k):
            rows = int(np.count_nonzero(drawn_k == k))
            table = offsets.get(int(k))
            if table is None:
                raise SystemExit(f"config has no dist_to_neighbors for k={k}")
            wrapped, reach = draw_offsets(table, k, rows, p, rng, replace)
            if reach < int(k):
                short += rows
            wrapped.sort(axis=1)
            distinct = 1 + np.count_nonzero(np.diff(wrapped, axis=1), axis=1)
            realised_partners.append(distinct)
            # the C++ does not exclude the rank itself; offset 0 mod p targets self
            realised_self += int(np.count_nonzero(wrapped == 0))
            total_draws += wrapped.size
    realised_partners = np.concatenate(realised_partners)
    if short:
        print(f"[warn] {short} ranks could not reach k distinct targets", flush=True)
    realised_sizes = rng.choice(size_keys, size=min(realised_partners.sum(), 2_000_000),
                                p=size_probs)
    return (histogram(realised_partners), histogram(realised_sizes),
            realised_self / max(total_draws, 1))


def true_pattern(name, rows, cols, diag, n, p):
    """Per-rank partner counts, per-edge message bytes, per-edge sender offsets."""
    _summary, detail = analyze(name, rows, cols, diag, n, p, 0.0)
    neighbours = detail["neighbors"]
    # msg_src receives from msg_dst, so from the sender's view the target is
    # msg_src and the offset is target - sender, matching the benchmark's
    # (offset + rank) placement.
    offsets = (detail["msg_src"] - detail["msg_dst"]).astype(np.int64)
    return (histogram(neighbours), histogram(detail["msg_bytes"]),
            histogram(offsets))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--repeats", type=int, default=20,
                    help="Monte Carlo repeats of the whole rank field (default 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--distinct-neighbors",
        action="store_true",
        help="sample offsets without replacement, so a rank never draws the same "
        "partner twice; this is the proposed --distinct-neighbors behaviour, "
        "against the benchmark's current draw-with-replacement default",
    )
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    records = []
    cache = {}
    for nodes in args.nodes:
        path = INPUT_DIR / f"t{nodes}.json"
        if not path.is_file():
            print(f"[skip] no {path.name}", flush=True)
            continue
        configs = read_config(path)
        p = nodes * RANKS_PER_NODE
        ranks_in_config = sum(next(iter(configs.values()))["partners"].values())
        stale = ranks_in_config != p
        if stale:
            print(f"[warn] {path.name} describes {ranks_in_config} ranks, not {p}",
                  flush=True)
        for matrix, config in sorted(configs.items()):
            if matrix not in cache:
                mtx = args.data_dir / f"{matrix}.mtx"
                cache[matrix] = read_pattern(mtx, args.data_dir)
            rows, cols, diag, n = cache[matrix]
            t_partners, t_sizes, t_offsets = true_pattern(matrix, rows, cols, diag, n, p)

            c_offsets = {}
            for inner in config["offsets"].values():
                for key, count in inner.items():
                    c_offsets[key] = c_offsets.get(key, 0) + count

            r_partners, r_sizes, self_frac = simulate_replay(
                config, p, args.repeats, rng, replace=not args.distinct_neighbors
            )

            bins = config["buffer_size"].keys()
            pairs = {
                ("config", "partners"): (config["partners"], t_partners),
                ("config", "buffer_size"): (config["buffer_size"], t_sizes),
                ("config", "buffer_size_binned"):
                    (config["buffer_size"], rebin_to(t_sizes, bins)),
                ("config", "offsets"): (c_offsets, t_offsets),
                ("replay", "partners"): (r_partners, t_partners),
                ("replay", "buffer_size"): (r_sizes, t_sizes),
                ("replay", "buffer_size_binned"):
                    (r_sizes, rebin_to(t_sizes, bins)),
            }
            summary_line = {}
            for (level, quantity), (left, right) in pairs.items():
                record = {
                    "matrix": matrix, "nodes": nodes, "ranks": p,
                    "config_ranks": ranks_in_config, "stale_config": stale,
                    "level": level, "quantity": quantity,
                    "sampling": "without_replacement" if args.distinct_neighbors
                    else "with_replacement",
                }
                record.update(compare(left, right))
                got, want = tail_scalars(left), tail_scalars(right)
                for key in ("max", "p99", "p90", "median"):
                    record[f"{key}_model"] = got[key]
                    record[f"{key}_true"] = want[key]
                record["self_target_fraction"] = self_frac if level == "replay" else float("nan")
                records.append(record)
                summary_line[(level, quantity)] = record
            print(
                f"[{matrix} {nodes}n] partners: cosine config="
                f"{summary_line[('config', 'partners')]['cosine']:.4f} "
                f"replay={summary_line[('replay', 'partners')]['cosine']:.4f}, "
                f"W1 replay={summary_line[('replay', 'partners')]['wasserstein']:.2f}, "
                f"max true={max(t_partners)} config={max(config['partners'])} "
                f"replay={max(r_partners)}",
                flush=True,
            )

    import pandas as pd

    frame = pd.DataFrame(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv = args.out_dir / "replay_similarity.csv"
    frame.to_csv(csv, index=False)
    draw(frame, args.out_dir / "replay_similarity.png")
    print(f"[done] {len(frame)} rows -> {csv}", flush=True)
    return frame


def draw(frame, path):
    import matplotlib.pyplot as plt

    colors = {
        "crankseg_1": "#1f77b4", "crankseg_2": "#ff7f0e", "m_t1": "#2ca02c",
        "nd12k": "#d62728", "nd24k": "#9467bd", "x104": "#8c564b",
    }
    shown = ("partners", "buffer_size_binned", "offsets")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for col, quantity in enumerate(shown):
        for row, (metric, label) in enumerate(
            (("cosine", "cosine similarity"), ("wasserstein", "Wasserstein-1"))
        ):
            ax = axes[row, col]
            for matrix, part in frame[frame.quantity == quantity].groupby("matrix"):
                for level, style in (("config", "-"), ("replay", "--")):
                    sub = part[(part.level == level) & (~part.stale_config)]
                    if sub.empty:
                        continue
                    sub = sub.sort_values("nodes")
                    ax.plot(sub["nodes"], sub[metric], style, marker="o", ms=4,
                            color=colors[matrix], lw=1.5,
                            label=matrix if level == "config" else None)
                # outside the level loop: `level` above is exhausted by now
                stale = part[(part.level == "config") & (part.stale_config)]
                if not stale.empty:
                    ax.plot(stale["nodes"], stale[metric], "x",
                            color=colors[matrix], ms=9, mew=2)
            ax.set_xscale("log", base=2)
            ax.set_xticks([2, 4, 8, 16, 32])
            ax.set_xticklabels(["2", "4", "8", "16", "32"])
            ax.grid(alpha=0.25)
            if row == 1:
                # symlog, not log: a perfect match gives W1 exactly 0
                ax.set_yscale("symlog", linthresh=0.1)
                ax.set_ylim(bottom=0)
                ax.set_xlabel("nodes (112 ranks each)")
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(quantity)
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle(
        "Does the replayed pattern rhyme with the real SpMV halo?\n"
        "solid: the config's histogram against the truth;  dashed: the pattern "
        "sampling actually produces;  x: the stale 32-node config"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
