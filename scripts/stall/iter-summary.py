#!/usr/bin/env python3
"""Summarize the reduced per-iteration files (*.iter.npy[.gz]) of one or more launches,
small enough to copy off dane through the tmux pane. Reads only; writes three CSVs.

usage: iter-summary.py OUT_PREFIX LAUNCH_DIR [LAUNCH_DIR ...]

The first K rows of each file (the startup window, K=25, as in the campaign) are skipped.
  OUT_PREFIX.runs.csv    one row per file: pattern, arm, timed rows, sum of iter_max_ms,
                         quantiles of iter_max_ms (and apply_max_ms where the file has it),
                         iterations over 100 ms, and iterations within 10% of a whole
                         multiple of 160 ms
  OUT_PREFIX.slow.csv    every timed iteration over 10x its file's median iter_max_ms and
                         over 20 ms, with the slowest and least-waiting ranks
  OUT_PREFIX.hist.csv    counts of iter_max_ms per file in 80 log-spaced bins, 1 us to 100 s
"""
import csv, glob, gzip, os, re, sys
import numpy as np

K = 25
EDGES = np.logspace(-3, 5, 81)
NAME = re.compile(r'run_(\d+)_(.+)_(MPI|STANDARD|LOCALITY)_\d{8}-\d{6}')


def load(path):
    if path.endswith('.gz'):
        with gzip.open(path) as f:
            return np.load(f)
    return np.load(path)


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    q = [50, 90, 99, 99.9]
    with open(out + '.runs.csv', 'w', newline='') as fr, open(out + '.slow.csv', 'w', newline='') as fs, \
            open(out + '.hist.csv', 'w', newline='') as fh:
        wr, ws, wh = csv.writer(fr), csv.writer(fs), csv.writer(fh)
        wr.writerow(['launch', 'file', 'run', 'pattern', 'arm', 'timed_rows', 'sum_iter_max_ms']
                    + [f'iter_p{p}' for p in q] + ['iter_max'] + [f'apply_p{p}' for p in q] + ['apply_max']
                    + ['over_100ms', 'near_160k', 'max_over_160'])
        ws.writerow(['launch', 'file', 'iteration', 'iter_max_ms', 'apply_max_ms', 'slowest_rank', 'iter_min_rank',
                     'over_160'])
        wh.writerow(['launch', 'file', 'pattern', 'arm'] + [f'{e:.6g}' for e in EDGES[:-1]])
        for d in dirs:
            launch = os.path.basename(os.path.normpath(d))
            for path in sorted(glob.glob(os.path.join(d, '*.iter.npy')) + glob.glob(os.path.join(d, '*.iter.npy.gz'))):
                name = os.path.basename(path)
                m = NAME.match(name)
                run, pattern, arm = m.groups() if m else ('', '', '')
                a = load(path)[K:]
                it = a['iter_max_ms'].astype(float)
                has_apply = 'apply_max_ms' in a.dtype.names
                ap = a['apply_max_ms'].astype(float) if has_apply else None
                x = it / 160.0
                near = int(((np.abs(x - np.rint(x)) < 0.1) & (np.rint(x) >= 1)).sum())
                wr.writerow([launch, name, run, pattern, arm, len(it), f'{it.sum():.3f}']
                            + [f'{v:.5g}' for v in np.percentile(it, q)] + [f'{it.max():.5g}']
                            + ([f'{v:.5g}' for v in np.percentile(ap, q)] + [f'{ap.max():.5g}'] if has_apply
                               else [''] * 5)
                            + [int((it > 100).sum()), near, f'{it.max() / 160:.3f}'])
                med = np.median(it)
                for i in np.where((it > 10 * med) & (it > 20))[0]:
                    ws.writerow([launch, name, i + K, f'{it[i]:.3f}', f'{ap[i]:.3f}' if has_apply else '',
                                 int(a['slowest_rank'][i]), int(a['iter_min_rank'][i]), f'{it[i] / 160:.3f}'])
                wh.writerow([launch, name, pattern, arm] + list(np.histogram(it, EDGES)[0]))


main()
