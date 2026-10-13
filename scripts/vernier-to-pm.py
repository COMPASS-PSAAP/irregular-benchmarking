#!/usr/bin/env python3
"""Write a PETSc binary matrix (.pm) whose SpMV halo is a Vernier pattern.

ABCG and irregular-benchmarking split a .pm's rows and columns into one contiguous
block per rank, and in an SpMV rank p sends rank r one message holding the x entries
of p's block that r's rows reference. So "p sends k doubles to r" is written as k
distinct columns of p's block referenced by r's rows. The matrix has to run on
exactly as many ranks as the pattern has.

  fine    CAPTURE_DIR OUT.pm      each rank's recorded sends, from its
                                  vernier-<rank>-0.hist-hierarchical.json (a
                                  capture of one exchange gives exact sizes)
  coarse  COARSE.json PATTERN RANKS OUT.pm --seed S
                                  one sample of an aggregate pattern (coarse.json,
                                  or any pattern config), drawn as the benchmark's
                                  sampledLists draws it; --distinct-neighbors as -U

Repeated draws to the same partner merge into one message, as Cabana merges them,
and a send to itself is dropped. Each rank owns n_local rows and columns, n_local
being the largest message in doubles; the local part is the diagonal only.

By default the matrix holds only the entries the sends need (value -1), with a
diagonal of 1 + off-diagonals in its row + off-diagonals in its column, so A and its
transpose are strictly diagonally dominant and p^T A p > 0 for every p: CG iterates
without an indefinite-matrix abort, though the matrix is not symmetric. With
--symmetric every p->r message must have an r->p one; each pair becomes a symmetric
block, the diagonal is 1 + off-diagonals in its row, and the matrix is symmetric
positive definite.

Writes OUT.pm and OUT.pm.json (source, settings, and every rank's (dest, bytes) as
the matrix's halo sends them: 8 bytes per double).
"""
import argparse
import array
import hashlib
import json
import math
import os
import random
import sys

PETSC_MAT_CODE = 1211216
DOUBLE = 8
# Vernier's send call codes (source/core/codex.hpp, 09d650b), as src/fine.cpp.
SEND_CALLS = {18, 19, 21, 30, 31, 32}


def draw_within(rng, edges, counts):
    """A size drawn as sample_interpolated in src/sampling.cpp: a bin by its count,
    then uniform inside it; the last bin as wide as the first; one bin is exact."""
    if len(edges) == 1:
        return int(edges[0])
    b = rng.choices(range(len(edges)), weights=counts)[0]
    upper = edges[b + 1] if b + 1 < len(edges) else edges[b] + (edges[1] - edges[0])
    return int(rng.uniform(edges[b], upper))


def fill_forward(hist):
    """src/pattern.cpp: a zero count takes the last non-zero count before it."""
    out, last = {}, 0
    for k in sorted(hist):
        v = hist[k]
        if v != 0:
            last = v
        elif last != 0:
            v = last
        out[k] = v
    return out


def load_vernier(path):
    with open(path) as f:
        header, comms, histograms = json.load(f)
    names = dict(pair.split(":") for pair in header["pattern_map"].split(","))
    return names, int(comms["comm_map"].split("|")[0]), histograms


def fine_sends(capture_dir, rng):
    names, ranks, _ = load_vernier(os.path.join(capture_dir, "vernier-0-0.hist-hierarchical.json"))
    if len(names) != 1:
        sys.exit(f"{capture_dir} holds patterns {list(names)}, expected one")
    (name, pid), = names.items()
    sends = []
    for r in range(ranks):
        _, size, histograms = load_vernier(os.path.join(capture_dir, f"vernier-{r}-0.hist-hierarchical.json"))
        if size != ranks:
            sys.exit(f"rank {r}'s file was captured on {size} ranks, rank 0's on {ranks}")
        mine = []
        for call, partners in histograms["pattern_" + pid].items():
            if int(call) not in SEND_CALLS:
                continue
            for partner, bins in partners.items():
                ordered = sorted((float(e), c) for e, c in bins.items())
                edges, counts = [e for e, _ in ordered], [c for _, c in ordered]
                for _ in range(sum(counts)):
                    mine.append((int(partner), draw_within(rng, edges, counts)))
        sends.append(mine)
    return name, ranks, sends


def coarse_sends(config, pattern, ranks, distinct, rng):
    with open(config) as f:
        p = json.load(f)[pattern]
    partners = fill_forward({int(k): v for k, v in p["comm_partners"].items()})
    sizes = fill_forward({int(float(k)): v for k, v in p["buffer_size"].items()})
    spread = p.get("buffer_size_bins") == "lower_edge"
    if "buffer_size_bins" in p and not spread:
        sys.exit(f"buffer_size_bins must be \"lower_edge\", not \"{p['buffer_size_bins']}\"")
    edges = sorted(sizes)
    counts = [sizes[e] for e in edges]
    offsets = {int(k): {int(d): v for d, v in inner.items()} for k, inner in p["dist_to_neighbors"].items()}
    sends = []
    for r in range(ranks):
        n = rng.choices(list(partners), weights=list(partners.values()))[0]
        dist = offsets[n]
        reachable = len({(d + r) % ranks for d in dist})
        seen, mine = set(), []
        for _ in range(n):
            size = draw_within(rng, edges, counts) if spread else rng.choices(edges, weights=counts)[0]
            while True:
                dest = (rng.choices(list(dist), weights=list(dist.values()))[0] + r) % ranks
                if not (distinct and len(seen) < reachable and dest in seen):
                    break
            seen.add(dest)
            mine.append((dest, size))
        sends.append(mine)
    return pattern, sends


def merge(sends):
    """Per rank, {dest: doubles}: repeated partners merged, self-sends dropped. Each
    draw is its own run of ceil(bytes / 8) elements, as bytes_to_elems."""
    merged = []
    for r, mine in enumerate(sends):
        out = {}
        for dest, size in mine:
            if dest != r:
                out[dest] = out.get(dest, 0) + math.ceil(size / DOUBLE)
        merged.append(out)
    return merged


def build(merged, symmetric):
    ranks = len(merged)
    n = max([k for out in merged for k in out.values()] + [1])
    rows = {}  # global row -> {global col: value}

    def add(i, j):
        rows.setdefault(i, {})[j] = -1.0

    if symmetric:
        for p, out in enumerate(merged):
            for r, k1 in out.items():
                if p not in merged[r]:
                    sys.exit(f"--symmetric: rank {p} sends to rank {r} but rank {r} sends nothing to rank {p}")
                if p > r:
                    continue
                k2 = merged[r][p]
                for t in range(max(k1, k2)):
                    ri, pj = r * n + t % k2, p * n + t % k1
                    add(ri, pj)
                    add(pj, ri)
    else:
        for p, out in enumerate(merged):
            for r, k in out.items():
                for t in range(k):
                    add(r * n + t, p * n + t)

    size = ranks * n
    col_off = [0] * size
    for i, entries in rows.items():
        for j in entries:
            col_off[j] += 1
    lengths, cols, vals = array.array("i"), array.array("i"), array.array("d")
    for i in range(size):
        entries = dict(rows.get(i, {}))
        entries[i] = 1.0 + len(entries) + (0 if symmetric else col_off[i])
        for j in sorted(entries):
            cols.append(j)
            vals.append(entries[j])
        lengths.append(len(entries))
    return n, size, lengths, cols, vals


def write_pm(path, size, lengths, cols, vals):
    assert lengths.itemsize == 4 and vals.itemsize == 8
    header = array.array("i", [PETSC_MAT_CODE, size, size, len(cols)])
    with open(path, "wb") as f:
        for a in (header, lengths, cols, vals):
            if sys.byteorder == "little":  # PETSc binary is big-endian
                a = array.array(a.typecode, a)
                a.byteswap()
            a.tofile(f)


def sha256(path):
    h = hashlib.sha256()
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.endswith(".hist-hierarchical.json"))
        for name in names:
            with open(os.path.join(path, name), "rb") as f:
                h.update(f.read())
    else:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    fine = sub.add_parser("fine")
    fine.add_argument("capture_dir")
    coarse = sub.add_parser("coarse")
    coarse.add_argument("config")
    coarse.add_argument("pattern")
    coarse.add_argument("ranks", type=int)
    coarse.add_argument("--distinct-neighbors", action="store_true")
    for p in (fine, coarse):
        p.add_argument("out")
        p.add_argument("--symmetric", action="store_true")
        p.add_argument("--seed", type=int, default=0, help="for draws (coarse; fine with several bins)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.mode == "fine":
        name, ranks, sends = fine_sends(args.capture_dir, rng)
        source = args.capture_dir
    else:
        name, sends = coarse_sends(args.config, args.pattern, args.ranks, args.distinct_neighbors, rng)
        ranks, source = args.ranks, args.config
    merged = merge(sends)
    n, size, lengths, cols, vals = build(merged, args.symmetric)
    write_pm(args.out, size, lengths, cols, vals)
    with open(args.out + ".json", "w") as f:
        json.dump({
            "mode": args.mode, "source": os.path.abspath(source), "source_sha256": sha256(source),
            "pattern": name, "seed": args.seed, "symmetric": args.symmetric,
            "distinct_neighbors": getattr(args, "distinct_neighbors", None),
            "ranks": ranks, "n_local": n, "rows": size, "nnz": len(cols),
            # What the matrix's halo sends: per rank, (dest, bytes = 8 x doubles).
            "sends": [[[dest, DOUBLE * k] for dest, k in sorted(out.items())] for out in merged],
        }, f)
    msgs = sum(len(out) for out in merged)
    print(f"{args.out}: {size} rows ({ranks} ranks x {n}), {len(cols)} nonzeros, {msgs} messages, "
          f"{'symmetric' if args.symmetric else 'nonsymmetric'}; run on exactly {ranks} ranks")


if __name__ == "__main__":
    main()
