#!/usr/bin/env python3
"""Barrier exit skew from a run's per-iteration files (--iter-raw).

usage: barrier-skew.py RAW OFFSETS [BINDING]

RAW is PREFIX_<pattern>_s<sample>.raw.npy[.gz]; OFFSETS the matching .offsets.npy[.gz];
BINDING, optionally, provenance.sh's .binding.txt (rank host ...), to name nodes.

Each rank records start_s, its wall-clock time at the start of every iteration, which
in the replay under -b and in ABCG's benchmark_spmv is right after it leaves the
previous barrier (the first column, the barrier before the loop). The offsets file
holds each rank's clock offset from rank 0, measured before and after the loop; the
offset is interpolated linearly in time between the two and subtracted, and the skew
of each barrier is then the latest minus the earliest corrected exit. The correction is
good to +/- rtt/2 per rank, so skews below about the largest rtt are not resolved.
ABCG's benchmark_cg_pilot has no barrier: its numbers are the spread of iteration
starts after CG's allreduces, labelled as such. Not valid for replay runs with -N/-T:
the noisy-neighbor sleep falls between the barrier and start_s. start_s is float64
epoch seconds, which quantizes at about 0.24 us.
"""
import collections
import gzip
import sys

import numpy as np


def load(path):
    return np.load(gzip.open(path) if path.endswith(".gz") else path)


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    raw, off = load(argv[1]), load(argv[2])
    ranks, iters = raw.shape
    if off.shape[0] != ranks:
        sys.exit(f"{argv[2]} has {off.shape[0]} ranks, {argv[1]} has {ranks}")
    host = None
    if len(argv) == 4:
        host = {}
        for line in open(argv[3]):
            fields = line.split()
            if len(fields) >= 2 and fields[0].isdigit():
                host[int(fields[0])] = fields[1]

    start = raw["start_s"]
    # Offset of each rank at each of its iterations, linear between the two measurements
    # (rank 0's times; the rank's own start_s minus its offset is close enough to place it).
    t0, t1 = off["t_before_s"][:, None], off["t_after_s"][:, None]
    o0, o1 = off["offset_before_s"][:, None], off["offset_after_s"][:, None]
    span = np.where(t1 > t0, t1 - t0, 1.0)
    frac = np.clip((start - o0 - t0) / span, 0.0, 1.0)
    corrected = start - (o0 + frac * (o1 - o0))

    skew_us = (corrected.max(axis=0) - corrected.min(axis=0)) * 1e6
    rtt = np.concatenate([off["rtt_before_s"][1:], off["rtt_after_s"][1:]])
    resolution_us = (rtt.max() / 2 * 1e6) if rtt.size else 0.0
    drift_us = np.abs(off["offset_after_s"] - off["offset_before_s"]).max() * 1e6

    has_barrier = "apply_ms" in raw.dtype.names
    what = "barrier exit skew" if has_barrier else \
        "spread of iteration starts (no barrier: CG's allreduces end each iteration)"
    print(f"{argv[1]}: {ranks} ranks, {iters} iterations")
    print(f"{what} (us): median {np.median(skew_us):.2f}  p90 {np.percentile(skew_us, 90):.2f}  "
          f"p99 {np.percentile(skew_us, 99):.2f}  max {skew_us.max():.2f} (iteration {int(skew_us.argmax())})")
    print(f"offset correction: worst rtt/2 {resolution_us:.2f} us, median rtt/2 "
          f"{(np.median(rtt) / 2 * 1e6 if rtt.size else 0.0):.2f} us; largest offset drift over the run "
          f"{drift_us:.2f} us; largest |offset| {np.abs(off['offset_before_s']).max() * 1e6:.1f} us")
    if np.median(skew_us) < 2 * resolution_us:
        print(f"NOTE: the median skew is below twice the worst rtt/2 ({2 * resolution_us:.2f} us), "
              "so it is not resolved by this correction; treat it as an upper bound, not a value")

    last = collections.Counter(np.argmax(corrected, axis=0).tolist())
    print("most often last out (rank: share of iterations):",
          ", ".join(f"{r}{' (' + host[r] + ')' if host and r in host else ''}: {n / iters:.1%}"
                    for r, n in last.most_common(5)))
    if host:
        by_node = collections.Counter(host.get(r, "?") for r in np.argmax(corrected, axis=0).tolist())
        print("last out by node:", ", ".join(f"{h}: {n / iters:.1%}" for h, n in by_node.most_common()))


if __name__ == "__main__":
    main(sys.argv)
