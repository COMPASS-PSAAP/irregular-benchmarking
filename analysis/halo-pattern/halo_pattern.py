#!/usr/bin/env python3
"""Halo communication-graph analyzer for SuiteSparse matrices.

Applies the row distribution PETSc gives a matrix loaded with MatLoad (contiguous
row blocks, the remainder spread over the first ranks) and reports, per process
count, the halo exchange that an SpMV over that distribution would perform:

  neighbour count n_i   distinct remote ranks owning a column that rank i's rows
                        reference, i.e. the number of halo partners of rank i
  message size          distinct remote indices per (src, dst) pair, x8 bytes
  nnz_i                 nonzeros in rank i's rows, diagonal included, the
                        compute-load proxy

and the order statistics that decide whether the interval maximum max_i T_i is
paced by the top-degree rank or contested by many ranks.

Pinning index
-------------
The matching term of the per-rank cost model has mean proportional to k^2 and
spread proportional to k^1.5 with one shared constant, so the constant cancels in

    Pi = (n_(1)^2 - n_(2)^2) / n_(1)^1.5

Pi > 1 means the top-degree rank's deterministic lead over the runner-up exceeds
the fluctuation scale of the term that produces it: the maximum is pinned to that
rank, and a regime change moves max_i n_i directly. Pi < 1 means the maximum is
contested and extreme-value behaviour of the whole field matters instead. The
contender count says how many ranks sit within one fluctuation scale of the top.

Exogenous jitter is a machine constant and is not in the matrix; --jitter-sigma
adds it to the comparison, in the same units as the matching constant. Left at 0
the reported numbers depend on the matrix alone.

Matching-queue cost, and what aggregation does to it
----------------------------------------------------
A rank that posts n_i receives pays a posted-receive queue search per arriving
message, so its matching cost goes as n_i^2. MPI Advance's locality-aware path
splits that one long queue into four short ones (out/regimes.csv):

  local_L  on-region traffic, which bypasses aggregation entirely
  local_S  intra-region gather to the aggregator owning the destination region
  global   one inter-region message per (region, remote region) pair
  local_R  intra-region scatter from the aggregator to the consumers

Destination regions are dealt across every rank of the region in the snake order
of msg_node_to_local, so there is no single leader: at 112 ranks per node and at
most 64 nodes here, a rank owns 0 or 1 remote nodes.

Bytes always go up, because off-region data is copied twice on-region. The
message count can go either way and the tool reports which: aggregation adds
intra-region hops but collapses every (rank, rank) pair between two regions into
one inter-region message, so a region pair carrying many rank pairs comes out
ahead. What reliably falls is the longest queue any one rank searches, and
whether that pays is decided by max_i n_i against the region size.

q_la_split adds the four phases as squares, which assumes each is a separate set
of posted receives. q_la_joint squares their sum, the pessimistic case where all
four are outstanding at once. The truth is between them.
"""

import argparse
import json
import math
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {
    "m_t1": "DNVS",
    "x104": "DNVS",
    "crankseg_1": "GHS_psdef",
    "crankseg_2": "GHS_psdef",
    "nd12k": "ND",
    "nd24k": "ND",
}
DEFAULT_MATRICES = ("m_t1", "x104", "crankseg_1", "crankseg_2", "nd12k", "nd24k")
DEFAULT_NODES = (2, 4, 8, 16, 32, 64)
DANE_CORES_PER_NODE = 112
DOUBLE_BYTES = 8
# Dane: 112 ranks per node over 8 NUMA regions, so 14 is one NUMA domain and is
# the value ABCG hard-codes as ppn/8.
REGION_NAMES = {14: "numa", 28: "2numa", 56: "socket", 112: "node"}
DEFAULT_REGIONS = (14, 28, 56, 112)
URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"


def fetch(name, group, data_dir):
    mtx = data_dir / f"{name}.mtx"
    if mtx.exists():
        return mtx
    tgz = data_dir / f"{name}.tar.gz"
    url = URL.format(group=group, name=name)
    print(f"[fetch] {url}", flush=True)
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as tf:
        member = tf.getmember(f"{name}/{name}.mtx")
        member.name = mtx.name
        tf.extract(member, path=data_dir)
    tgz.unlink()
    return mtx


def read_pattern(mtx, cache_dir):
    """Return (rows, cols, diag, n).

    rows/cols are the full symmetric off-diagonal pattern; diag holds the row
    indices carrying a diagonal entry, which move no data but do carry work.
    """
    cache = cache_dir / f"{mtx.stem}.pattern.npz"
    if cache.exists():
        z = np.load(cache)
        return z["rows"], z["cols"], z["diag"], int(z["n"][0])

    with open(mtx) as fh:
        banner = fh.readline()
        fields = banner.lower().split()
        if "coordinate" not in fields:
            raise SystemExit(f"{mtx.name}: not coordinate format: {banner.strip()}")
        if "symmetric" not in fields:
            # Send and recv neighbour sets only coincide for a symmetric pattern;
            # everything downstream reports one set, so refuse rather than mislead.
            raise SystemExit(f"{mtx.name}: pattern is not symmetric: {banner.strip()}")
        skip = 1
        for line in fh:
            if not line.startswith("%"):
                nrow, ncol, _ = (int(v) for v in line.split())
                break
            skip += 1
    if nrow != ncol:
        raise SystemExit(f"{mtx.name}: not square: {nrow}x{ncol}")

    print(f"[parse] {mtx.name} n={nrow}", flush=True)
    df = pd.read_csv(
        mtx,
        sep=r"\s+",
        skiprows=skip + 1,
        header=None,
        usecols=[0, 1],
        names=["i", "j", "v"],
        dtype={"i": np.int32, "j": np.int32},
    )
    i = df["i"].to_numpy()
    j = df["j"].to_numpy()
    # to_numpy() is read-only under copy-on-write, so shift to 0-based after the
    # concatenation, which copies.
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    del df, i, j
    rows -= 1
    cols -= 1
    keep = rows != cols
    diag = rows[~keep][: (~keep).sum() // 2]
    rows = rows[keep]
    cols = cols[keep]
    del keep
    np.savez(cache, rows=rows, cols=cols, diag=diag, n=np.array([nrow], dtype=np.int64))
    return rows, cols, diag, nrow


def petsc_offsets(n, p):
    """PetscSplitOwnership: floor(n/p) rows each, the first n%p ranks take one more."""
    base, rem = divmod(n, p)
    sizes = np.full(p, base, dtype=np.int64)
    sizes[:rem] += 1
    offsets = np.zeros(p + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return offsets


def owners_of(idx, offsets, chunk=8_000_000):
    """Owning rank of each global index, chunked to keep peak memory down."""
    out = np.empty(idx.size, dtype=np.int32)
    for s in range(0, idx.size, chunk):
        e = min(s + chunk, idx.size)
        found = np.searchsorted(offsets, idx[s:e], side="right")
        np.subtract(found, 1, out=out[s:e], casting="unsafe")
    return out


def pearson(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())


def histogram(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(c) for k, c in zip(keys, counts)}


def analyze(name, rows, cols, diag, n, p, jitter_sigma):
    offsets = petsc_offsets(n, p)
    ro = owners_of(rows, offsets)
    co = owners_of(cols, offsets)
    nnz_per_rank = np.bincount(ro, minlength=p).astype(np.int64)
    nnz_per_rank += np.bincount(owners_of(diag, offsets), minlength=p)
    rows_per_rank = np.diff(offsets)

    remote = ro != co
    ro_o = ro[remote]
    co_o = co[remote]
    cols_o = cols[remote]
    del ro, co, remote

    edge_key = np.unique(ro_o.astype(np.int64) * p + co_o)
    esrc = (edge_key // p).astype(np.int32)
    neighbors = np.bincount(esrc, minlength=p).astype(np.int64)
    del esrc

    idx_key = np.unique(ro_o.astype(np.int64) * n + cols_o)
    del ro_o, co_o, cols_o
    isrc = (idx_key // n).astype(np.int32)
    icol = idx_key - isrc.astype(np.int64) * n
    del idx_key
    recv_indices = np.bincount(isrc, minlength=p).astype(np.int64)
    idst = owners_of(icol, offsets)
    del icol
    msg_key, msg_indices = np.unique(isrc.astype(np.int64) * p + idst, return_counts=True)
    del isrc, idst
    if not np.array_equal(msg_key, edge_key):
        raise SystemExit(f"{name} P={p}: message keys disagree with neighbour edges")

    msrc = (msg_key // p).astype(np.int64)
    mdst = msg_key - msrc * p
    msg_bytes = msg_indices * DOUBLE_BYTES

    order = np.sort(neighbors)[::-1]
    n1 = int(order[0])
    n2 = int(order[1]) if p > 1 else 0
    n3 = int(order[2]) if p > 2 else 0
    cost = neighbors.astype(np.float64) ** 2
    pin = pinning(cost, jitter_sigma)
    n_contenders = contenders(cost, jitter_sigma)

    p50 = float(np.percentile(neighbors, 50))
    p99 = float(np.percentile(neighbors, 99))
    summary = {
        "matrix": name,
        "n_rows": n,
        "nnz_offdiag_full": int(rows.size),
        "P": p,
        "rows_per_rank_min": int(rows_per_rank.min()),
        "rows_per_rank_max": int(rows_per_rank.max()),
        "nnz_per_rank_mean": float(nnz_per_rank.mean()),
        "nnz_per_rank_max": int(nnz_per_rank.max()),
        "nnz_imbalance_max_over_mean": float(nnz_per_rank.max() / nnz_per_rank.mean()),
        "neighbors_mean": float(neighbors.mean()),
        "neighbors_p50": p50,
        "neighbors_p90": float(np.percentile(neighbors, 90)),
        "neighbors_p99": p99,
        "neighbors_p99_over_p50": float(p99 / p50) if p50 > 0 else float("nan"),
        "neighbors_max": n1,
        "neighbors_2nd": n2,
        "neighbors_3rd": n3,
        "neighbors_min": int(neighbors.min()),
        "argmax_neighbors_rank": int(np.argmax(neighbors)),
        "pinning_index": pin,
        "pinned": bool(pin > 1.0) if not math.isnan(pin) else False,
        "contender_count": n_contenders,
        "two_sqrt_P": 2.0 * math.sqrt(p),
        "max_over_2sqrtP": n1 / (2.0 * math.sqrt(p)),
        "sqrt_P": math.sqrt(p),
        "messages": int(msg_key.size),
        "msg_bytes_p50": float(np.percentile(msg_bytes, 50)),
        "msg_bytes_p99": float(np.percentile(msg_bytes, 99)),
        "msg_bytes_max": int(msg_bytes.max()),
        "recv_bytes_max": int(recv_indices.max() * DOUBLE_BYTES),
        "m90": int(np.count_nonzero(neighbors**2 >= 0.90 * n1**2)),
        "m50": int(np.count_nonzero(neighbors**2 >= 0.50 * n1**2)),
        "share_top": float(n1**2 / (neighbors**2).sum() * p) if neighbors.max() else float("nan"),
        "corr_nnz_neighbors_pearson": pearson(nnz_per_rank, neighbors),
        "corr_nnz_neighbors_spearman": spearman(nnz_per_rank, neighbors),
        "argmax_nnz_is_argmax_neighbors": bool(
            int(np.argmax(nnz_per_rank)) == int(np.argmax(neighbors))
        ),
        "jitter_sigma": float(jitter_sigma),
    }

    detail = {
        "neighbors": neighbors,
        "nnz_per_rank": nnz_per_rank,
        "rows_per_rank": rows_per_rank,
        "recv_indices": recv_indices,
        "msg_bytes": msg_bytes,
        "msg_src": msrc,
        "msg_dst": mdst,
    }
    return summary, detail



def snake_assign(count, slots, incr):
    """Port of msg_node_to_local in MPI Advance's neighbor_locality.cpp.

    Deals `count` destination regions across `slots` local ranks, walking up then
    back down. The C++ loop leaves local_proc unchanged on the turn, so the first
    and last slots each take two consecutive regions.
    """
    out = np.empty(count, dtype=np.int32)
    local = 0 if incr else slots - 1
    inc = 1 if incr else -1
    for k in range(count):
        out[k] = local
        if local == slots - 1 and inc == 1:
            inc = -1
        elif local == 0 and inc == -1:
            inc = 1
        else:
            local += inc
    return out


def regimes(name, detail, p, r_size):
    """Matching-queue cost under point-to-point and under locality-aware aggregation."""
    msrc = detail["msg_src"].astype(np.int64)  # the receiving rank
    mdst = detail["msg_dst"].astype(np.int64)  # the sending rank
    counts = (detail["msg_bytes"] // DOUBLE_BYTES).astype(np.int64)
    neighbors = detail["neighbors"].astype(np.int64)

    ranks = np.arange(p, dtype=np.int64)
    node_of = ranks // r_size
    num_nodes = int(node_of[-1]) + 1
    node_size = np.bincount(node_of, minlength=num_nodes)

    a = node_of[msrc]
    b = node_of[mdst]
    on = a == b
    local_l = np.bincount(msrc[on], minlength=p)

    ar, br = a[~on], b[~on]
    rr, sr = msrc[~on], mdst[~on]
    cr = counts[~on]

    # pair_size[X, Y]: indices region X receives from region Y, summed over its
    # ranks because the current implementation does not deduplicate.
    pair_size = np.bincount(
        ar * num_nodes + br, weights=cr, minlength=num_nodes * num_nodes
    ).reshape(num_nodes, num_nodes)

    # A symmetric pattern gives symmetric partner sets even though the sizes
    # differ, and the global step indexes owner_recv by those pairs.
    if not np.array_equal(pair_size > 0, pair_size.T > 0):
        raise SystemExit(f"{name} P={p} R={r_size}: region partner sets are not symmetric")

    owner_recv = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
    owner_send = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
    max_partners = 0
    for A in range(num_nodes):
        partners = np.nonzero(pair_size[A] + pair_size[:, A])[0]
        if partners.size == 0:
            continue
        max_partners = max(max_partners, int(partners.size))
        slots = int(node_size[A])
        # std::sort puts the largest first; node id breaks ties so this port is
        # deterministic where the library's unstable sort is not.
        rec = partners[np.lexsort((partners, -pair_size[A, partners]))]
        snd = partners[np.lexsort((partners, -pair_size[partners, A]))]
        owner_recv[A, rec] = snake_assign(rec.size, slots, incr=False)
        owner_send[A, snd] = snake_assign(snd.size, slots, incr=True)

    pair_a, pair_b = np.nonzero(pair_size)
    glob_agg = pair_a * r_size + owner_recv[pair_a, pair_b]
    global_recv = np.bincount(glob_agg, minlength=p)

    agg_r = ar * r_size + owner_recv[ar, br]
    keep_r = agg_r != rr
    pairs_r = np.unique(rr[keep_r] * p + agg_r[keep_r])
    local_r = np.bincount(pairs_r // p, minlength=p)

    agg_s = br * r_size + owner_send[br, ar]
    keep_s = agg_s != sr
    pairs_s = np.unique(agg_s[keep_s] * p + sr[keep_s])
    local_s = np.bincount(pairs_s // p, minlength=p)

    steps = np.stack([local_l, local_s, global_recv, local_r]).astype(np.float64)
    q_p2p = neighbors.astype(np.float64) ** 2
    q_split = (steps**2).sum(axis=0)
    q_joint = steps.sum(axis=0) ** 2

    top = int(np.argmax(q_split))
    bytes_on = int(counts[on].sum()) * DOUBLE_BYTES
    bytes_off = int(cr.sum()) * DOUBLE_BYTES
    row = {
        "matrix": name,
        "P": p,
        "R": r_size,
        "region": REGION_NAMES.get(r_size, str(r_size)),
        "num_regions": num_nodes,
        "max_partner_regions": max_partners,
        "max_q_p2p": float(q_p2p.max()),
        "max_q_la_split": float(q_split.max()),
        "max_q_la_joint": float(q_joint.max()),
        "max_q_ratio_split": float(q_p2p.max() / q_split.max()) if q_split.max() else float("nan"),
        "max_q_ratio_joint": float(q_p2p.max() / q_joint.max()) if q_joint.max() else float("nan"),
        "sum_q_p2p": float(q_p2p.sum()),
        "sum_q_la_split": float(q_split.sum()),
        "sum_q_ratio_split": float(q_p2p.sum() / q_split.sum()) if q_split.sum() else float("nan"),
        "la_wins_on_max": bool(q_split.max() < q_p2p.max()),
        "argmax_la_rank": top,
        "at_argmax_local_L": int(local_l[top]),
        "at_argmax_local_S": int(local_s[top]),
        "at_argmax_global": int(global_recv[top]),
        "at_argmax_local_R": int(local_r[top]),
        "max_queue_p2p": int(neighbors.max()),
        "max_queue_la_step": int(steps.max()),
        "messages_p2p": int(msrc.size),
        "messages_la": int(on.sum()) + int(pairs_s.size) + int(pair_a.size) + int(pairs_r.size),
        "messages_local_L": int(on.sum()),
        "messages_local_S": int(pairs_s.size),
        "messages_global": int(pair_a.size),
        "messages_local_R": int(pairs_r.size),
        "max_messages_per_rank_p2p": int(neighbors.max()),
        "max_messages_per_rank_la": int(steps.sum(axis=0).max()),
        "bytes_p2p": bytes_on + bytes_off,
        "bytes_la": bytes_on + 3 * bytes_off,
        "bytes_inter_region": bytes_off,
        "pinning_index_p2p": pinning(q_p2p),
        "pinning_index_la": pinning(q_split),
        "contenders_la": contenders(q_split),
    }
    row["messages_ratio_la_over_p2p"] = row["messages_la"] / row["messages_p2p"]
    return row


def pinning(q, jitter_sigma=0.0):
    """(Q_(1) - Q_(2)) / max(Q_(1)^0.75, sigma).

    The matching term has mean proportional to k^2 and spread to k^1.5, so on a
    cost scale Q ~ k^2 the spread is Q^0.75 and the shared constant cancels.
    """
    order = np.sort(q)[::-1]
    q1 = float(order[0])
    q2 = float(order[1]) if order.size > 1 else 0.0
    scale = max(q1**0.75, float(jitter_sigma))
    if scale <= 0.0:
        return float("nan")
    return (q1 - q2) / scale


def contenders(q, jitter_sigma=0.0):
    q1 = float(np.max(q))
    scale = max(q1**0.75, float(jitter_sigma))
    if scale <= 0.0:
        return int(np.size(q))
    return int(np.count_nonzero(q1 - q <= scale))


def write_perrank(out_dir, name, p, detail):
    path = out_dir / "perrank" / f"{name}_P{p}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix": name,
        "P": p,
        "neighbors": detail["neighbors"].tolist(),
        "nnz_per_rank": detail["nnz_per_rank"].tolist(),
        "rows_per_rank": detail["rows_per_rank"].tolist(),
        "recv_bytes_per_rank": (detail["recv_indices"] * DOUBLE_BYTES).tolist(),
        "message_bytes_histogram": histogram(detail["msg_bytes"]),
    }
    path.write_text(json.dumps(payload))
    return path


def write_pattern(out_dir, name, p, detail):
    """Emit the pattern config format irregular-benchmarking's pattern.hpp reads."""
    neighbors = detail["neighbors"]
    msrc = detail["msg_src"]
    dist = {}
    k_of_edge = neighbors[msrc]
    delta = detail["msg_dst"] - msrc
    for k in np.unique(k_of_edge):
        dist[str(int(k))] = histogram(delta[k_of_edge == k])
    payload = {
        f"{name}_P{p}": {
            "comm_partners": histogram(neighbors),
            "buffer_size": histogram(detail["msg_bytes"]),
            "dist_to_neighbors": dist,
            "pattern_count": 1,
            "message_count": int(msrc.size),
        }
    }
    path = out_dir / "patterns" / f"{name}_P{p}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_plots(out_dir, name, per_p, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p, detail in sorted(per_p.items()):
        srt = np.sort(detail["neighbors"])[::-1]
        ax.plot(np.arange(1, srt.size + 1), srt, label=f"P={p}")
    ax.set_xscale("log")
    ax.set_xlabel("rank, ordered by neighbour count")
    ax.set_ylabel("neighbour count $n_i$")
    ax.set_title(f"{name}: halo degree, PETSc row blocks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    degree_path = plot_dir / f"{name}_degree.png"
    fig.savefig(degree_path, dpi=130)
    plt.close(fig)

    rows = sorted((s["P"], s) for s in summaries)
    ps = [p for p, _ in rows]
    pins = [s["pinning_index"] for _, s in rows]
    conts = [s["contender_count"] for _, s in rows]
    maxes = [s["neighbors_max"] for _, s in rows]
    thresh = [s["two_sqrt_P"] for _, s in rows]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    a1.plot(ps, pins, "o-", label=r"pinning index $\Pi$")
    a1.axhline(1.0, color="k", ls="--", lw=1, label=r"$\Pi=1$")
    # symlog, not log: a tie between the top two ranks gives Pi exactly 0
    a1.set_yscale("symlog", linthresh=0.1)
    a1.set_ylim(bottom=0)
    a1.set_ylabel(r"$\Pi$")
    a1.legend(fontsize=8)
    a1.set_title(f"{name}: is the interval maximum pinned?")
    a2.plot(ps, maxes, "s-", label=r"$\max_i n_i$")
    a2.plot(ps, thresh, "--", label=r"$2\sqrt{P}$ (H22)")
    a2.plot(ps, conts, "^-", label="contender count")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("processes $P$")
    a2.legend(fontsize=8)
    fig.tight_layout()
    pin_path = plot_dir / f"{name}_pinning.png"
    fig.savefig(pin_path, dpi=130)
    plt.close(fig)
    return degree_path, pin_path


def write_queue_plot(out_dir, name, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ps = sorted({r["P"] for r in rows})
    p2p = [next(r["max_q_p2p"] for r in rows if r["P"] == p) for p in ps]
    ax.plot(ps, p2p, "ko-", label="point-to-point")
    for r_size in sorted({r["R"] for r in rows}):
        sel = {r["P"]: r for r in rows if r["R"] == r_size}
        ax.plot(
            ps,
            [sel[p]["max_q_la_split"] for p in ps],
            "o--",
            label=f"locality-aware, R={r_size} ({REGION_NAMES.get(r_size, r_size)})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("processes $P$")
    ax.set_ylabel(r"$\max_i$ matching cost (queue$^2$)")
    ax.set_title(f"{name}: matching-queue cost, point-to-point vs aggregated")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = plot_dir / f"{name}_queue.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--matrices",
        nargs="+",
        default=list(DEFAULT_MATRICES),
        help="SuiteSparse matrix names (default: the six ABCG matrices)",
    )
    ap.add_argument(
        "--group",
        help="SuiteSparse group for a matrix not in the built-in table",
    )
    ap.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=list(DEFAULT_NODES),
        help="node counts; ranks = nodes x --cores-per-node (default: 2 4 8 16 32 64)",
    )
    ap.add_argument(
        "--cores-per-node",
        type=int,
        default=DANE_CORES_PER_NODE,
        help=f"cores per node (default: {DANE_CORES_PER_NODE}, Dane)",
    )
    ap.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        help="explicit process counts, overriding --nodes",
    )
    ap.add_argument(
        "--region-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_REGIONS),
        help="aggregation region sizes in ranks (default: 14 28 56 112, i.e. "
        "NUMA domain, two NUMA domains, socket, node on Dane)",
    )
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out")
    ap.add_argument(
        "--jitter-sigma",
        type=float,
        default=0.0,
        help="exogenous jitter scale, in units of the matching constant (default 0)",
    )
    ap.add_argument("--no-perrank", action="store_true", help="skip per-rank JSON")
    ap.add_argument("--no-patterns", action="store_true", help="skip benchmark pattern JSON")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    args = ap.parse_args(argv)

    ranks = args.ranks or [n * args.cores_per_node for n in args.nodes]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    regime_rows = []
    for name in args.matrices:
        group = args.group or GROUPS.get(name)
        if group is None:
            raise SystemExit(f"{name}: unknown group, pass --group")
        mtx = fetch(name, group, args.data_dir)
        rows, cols, diag, n = read_pattern(mtx, args.data_dir)
        per_p = {}
        for p in ranks:
            if p > n:
                print(f"[skip] {name} P={p} exceeds {n} rows", flush=True)
                continue
            summary, detail = analyze(name, rows, cols, diag, n, p, args.jitter_sigma)
            summaries.append(summary)
            per_p[p] = detail
            if not args.no_perrank:
                write_perrank(args.out_dir, name, p, detail)
            if not args.no_patterns:
                write_pattern(args.out_dir, name, p, detail)
            for r_size in args.region_sizes:
                regime_rows.append(regimes(name, detail, p, r_size))
            print(
                f"[{name} P={p}] max n_i={summary['neighbors_max']} "
                f"(2nd {summary['neighbors_2nd']}) p50={summary['neighbors_p50']:.1f} "
                f"Pi={summary['pinning_index']:.2f} contenders={summary['contender_count']} "
                f"max/2sqrtP={summary['max_over_2sqrtP']:.2f}",
                flush=True,
            )
        if per_p and not args.no_plots:
            write_plots(args.out_dir, name, per_p, [s for s in summaries if s["matrix"] == name])
            write_queue_plot(
                args.out_dir, name, [r for r in regime_rows if r["matrix"] == name]
            )
        del rows, cols, diag, per_p

    if not summaries:
        raise SystemExit("no results")
    frame = pd.DataFrame(summaries)
    csv = args.out_dir / "summary.csv"
    frame.to_csv(csv, index=False)
    regime_csv = args.out_dir / "regimes.csv"
    pd.DataFrame(regime_rows).to_csv(regime_csv, index=False)
    print(f"[done] {len(summaries)} rows -> {csv}", flush=True)
    print(f"[done] {len(regime_rows)} rows -> {regime_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
