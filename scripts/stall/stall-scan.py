#!/usr/bin/env python3
"""Flag long iterations in a launch's per-iteration raw files and say where each went.

usage: stall-scan.py LAUNCH_DIR [MIN_MS]

For every run_*.raw.npy[.gz] in LAUNCH_DIR, an iteration is flagged when its slowest
rank (iter_max_ms) took more than 10x the run's median iter_max_ms and more than MIN_MS
(default 20). For each flagged iteration it prints the time in units of 160 ms and two
rank sets, by node (binding from provenance_*.binding.txt):
  apply: ranks whose own apply took more than half the iteration (the exchange waited;
         replay and ABCG SpMV only, CG has no apply column),
  late:  ranks that started the next iteration more than half the excess after the
         median rank (they left the barrier, or CG's previous iteration, late).
Late ranks are where the delay began; apply ranks are usually their partners waiting.
"""
import glob, gzip, os, sys
import numpy as np


def load(path):
    if path.endswith('.gz'):
        with gzip.open(path) as f:
            return np.load(f)
    return np.load(path, mmap_mode='r')


def nodes_of(ranks, binding):
    count = {}
    for r in ranks:
        count[binding[r]] = count.get(binding[r], 0) + 1
    return ' '.join(f'{n}:{c}' for n, c in sorted(count.items())) or '-'


def main():
    d = sys.argv[1]
    min_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    binding = {}
    for line in open(glob.glob(os.path.join(d, 'provenance_*.binding.txt'))[0]):
        f = line.split()
        binding[int(f[0])] = f[1]
    print('run,iteration,iter_max_ms,x160ms,median_ms,apply_ranks,apply_nodes,late_ranks,late_nodes')
    for raw_path in sorted(glob.glob(os.path.join(d, 'run_*.raw.npy*'))):
        raw = load(raw_path)
        it = raw['iter_ms'].max(axis=0)
        med = float(np.median(it))
        run = os.path.basename(raw_path).split('_')[1]
        for i in np.where((it > 10 * med) & (it > min_ms))[0]:
            col = raw[:, i]
            t = float(it[i])
            apply = (np.where(np.asarray(col['apply_ms']) > t / 2)[0]
                     if 'apply_ms' in raw.dtype.names else np.array([], int))
            late = np.array([], int)
            if i + 1 < raw.shape[1]:
                s = np.asarray(raw[:, i + 1]['start_s'])
                late = np.where(s - np.median(s) > (t - med) / 2000.0)[0]
            print(f'{run},{i},{t:.3f},{t / 160:.2f},{med:.4f},{len(apply)},{nodes_of(apply, binding)},'
                  f'{len(late)},{nodes_of(late, binding)}')


main()
