"""Compare the old pattern profiles (t<n>.json) with the new Vernier captures.

For each matrix in PILOT_DIR/capture/order.csv, capture NN is paired with the old
profile's pattern_(NN-1) (the order the old profiles list the matrices in) and with
coarse_<matrix> from PILOT_DIR/coarse.json, and both are compared with the matrix's
true halo, read from the capture's per-rank Vernier traces (vernier-<rank>-0.txt, one
record per message: IS:comm:dest:bytes:req).

messages_recorded counts sends and receives for a profile (Vernier records both)
and sends only for the truth.

Per profile it reports the raw histograms and what irregular-benchmarking's sampler
draws from them (src/pattern.cpp: zero bins take the last non-zero count, and a
buffer_size bin is sent as its key, which Vernier sets to the bin's lower edge).
Usage: python compare_profiles.py OLD_PROFILE.json PILOT_DIR RANKS_PER_NODE [OUT.csv]
"""
import csv
import json
import os
import sys


def fill_forward(hist):
    out, last = {}, 0
    for k in sorted(hist):
        v = hist[k]
        if v != 0:
            last = v
        elif last != 0:
            v = last
        out[k] = v
    return out


def mean(hist):
    total = sum(hist.values())
    return sum(k * v for k, v in hist.items()) / total if total else float("nan")


def node_change_fraction(offsets, ranks, per_node):
    """Expected share of messages to another node when every rank sends at these offsets."""
    total = sum(offsets.values())
    off = 0.0
    for d, w in offsets.items():
        crossings = sum(1 for r in range(ranks) if (r + d) % ranks // per_node != r // per_node)
        off += w * crossings / ranks
    return off / total if total else float("nan")


def profile_stats(p, per_node):
    partners = {int(k): v for k, v in p["comm_partners"].items()}
    sizes = {int(float(k)): v for k, v in p["buffer_size"].items()}
    ranks = sum(partners.values())
    edges = sorted(sizes)
    width = edges[1] - edges[0] if len(edges) > 1 else 0
    ff_partners, ff_sizes = fill_forward(partners), fill_forward(sizes)
    # Offsets as the sampler draws them: a partner count by its weight, then offsets for it.
    ff_total = sum(ff_partners.values())
    offsets = {}
    for c, w in ff_partners.items():
        inner = {int(d): v for d, v in p["dist_to_neighbors"].get(str(c), {}).items()}
        inner_total = sum(inner.values())
        for d, v in inner.items():
            offsets[d] = offsets.get(d, 0.0) + (w / ff_total) * (v / inner_total)
    return {
        "ranks_recorded": ranks,
        "messages_recorded": sum(sizes.values()),
        "partners_avg": mean(partners),
        "partners_max": max(partners),
        "size_bins": len(edges),
        "size_min_edge": edges[0],
        "size_max_edge": edges[-1],
        "size_bin_width": width,
        "size_avg_midpoint": mean({k + width / 2: v for k, v in sizes.items()}),
        "sampler_partners_avg": mean(ff_partners),
        "sampler_size_avg": mean(ff_sizes),
        "sampler_bytes_per_rank": mean(ff_partners) * mean(ff_sizes),
        "sampler_off_node_frac": node_change_fraction(offsets, ranks, per_node),
    }


def truth_stats(capture_dir, ranks, per_node):
    msgs, byte_totals, off = [], [], 0
    for r in range(ranks):
        with open(os.path.join(capture_dir, f"vernier-{r}-0.txt")) as f:
            lines = f.read().split("\n")[2:]
        sends = []
        for line in lines:
            if "|" not in line:
                continue
            for rec in line.split("|", 1)[1].split(","):
                fields = rec.split(":")
                if fields[0] == "IS":
                    sends.append((int(fields[2]), int(fields[3])))
        msgs.append(len(sends))
        byte_totals.append(sum(b for _, b in sends))
        off += sum(1 for dest, _ in sends if dest // per_node != r // per_node)
    n = sum(msgs)
    return {
        "ranks_recorded": ranks,
        "messages_recorded": n,
        "partners_avg": n / ranks,
        "partners_max": max(msgs),
        "size_avg_midpoint": sum(byte_totals) / n,
        "sampler_partners_avg": n / ranks,
        "sampler_size_avg": sum(byte_totals) / n,
        "sampler_bytes_per_rank": sum(byte_totals) / ranks,
        "sampler_off_node_frac": off / n,
    }


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    old_path, pilot_dir, per_node = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_path = sys.argv[4] if len(sys.argv) == 5 else None
    with open(old_path) as f:
        old = json.load(f)
    with open(os.path.join(pilot_dir, "coarse.json")) as f:
        coarse = json.load(f)
    with open(os.path.join(pilot_dir, "capture", "order.csv")) as f:
        order = sorted(csv.DictReader(f), key=lambda row: int(row["order"]))

    rows = []
    for row in order:
        matrix = row["matrix"]
        capture_dir = os.path.join(pilot_dir, "capture", os.path.basename(row["capture_dir"]))
        new = profile_stats(coarse["coarse_" + matrix], per_node)
        ranks = new["ranks_recorded"]
        old_name = f"pattern_{int(row['order']) - 1}"
        for source, name, stats in [
            ("old", old_name, profile_stats(old[old_name], per_node)),
            ("new", "coarse_" + matrix, new),
            ("truth", matrix, truth_stats(capture_dir, ranks, per_node)),
        ]:
            rows.append({"matrix": matrix, "source": source, "pattern": name, **stats})

    cols = ["matrix", "source", "pattern", "ranks_recorded", "messages_recorded", "partners_avg",
            "partners_max", "size_bins", "size_min_edge", "size_max_edge", "size_bin_width",
            "size_avg_midpoint", "sampler_partners_avg", "sampler_size_avg", "sampler_bytes_per_rank",
            "sampler_off_node_frac"]
    if out_path:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
    print("\t".join(cols))
    for r in rows:
        print("\t".join(fmt(r.get(c, "-")) for c in cols))


if __name__ == "__main__":
    main()
