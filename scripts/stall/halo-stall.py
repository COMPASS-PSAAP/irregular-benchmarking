import csv, sys, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('spy', '/mnt/c/Users/Nicholas/PycharmProjects/PythonProject3/data-10-15-2026-vernier-to-pm/spy.py')
# spy.py runs main() only under __main__, check before relying on import
src = open(spec.origin).read(); assert "if __name__" in src, 'spy.py has no main guard'
spy = importlib.util.module_from_spec(spec); spec.loader.exec_module(spy)
R, PER = 448, 112
n, rows, cols = spy.load_real('/home/bacon/Claude/halo-pattern-analysis/data/crankseg_2.pattern.npz')
h = spy.halo(n, rows, cols, R)           # h[r, p]: doubles r receives from p
node = np.arange(R) // PER
stalls = list(csv.DictReader(open(sys.argv[1])))
w = np.zeros(R); single = {}
inter = {}
for s in stalls:
    W = [int(x) for x in s['waiting'].split()]
    for r in W: w[r] += 1
    if len(W) == 1: single[W[0]] = single.get(W[0], 0) + 1
    common = np.all(h[W] > 0, axis=0) if W else np.zeros(R, bool)
    for p in np.where(common)[0]: inter[p] = inter.get(p, 0) + 1
print('stalls', len(stalls), ' ranks that ever waited', int((w > 0).sum()))
nsend = (h > 0).sum(1); rbytes = h.sum(1) * 8; rmax = h.max(1) * 8
off = np.array([(h[r] > 0)[node != node[r]].sum() for r in range(R)])
offb = np.array([h[r][node != node[r]].sum() * 8 for r in range(R)])
nout = (h > 0).sum(0); sbytes = h.sum(0) * 8
def rank_corr(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]
for name, f in [('senders', nsend), ('off-node senders', off), ('recv bytes', rbytes), ('off-node recv bytes', offb), ('largest recv msg', rmax), ('receivers (out)', nout), ('send bytes', sbytes)]:
    print(f'spearman(wait count, {name:20s}) = {rank_corr(w, f):+.2f}')
top = np.argsort(-w)[:12]
print('\nrank node waits senders off-node recvB  maxmsgB  offB')
for r in top: print(f'{r:4d} {node[r]:4d} {int(w[r]):5d} {nsend[r]:7d} {off[r]:8d} {rbytes[r]:6d} {rmax[r]:7d} {offb[r]:6d}')
print('\nall ranks median: senders %d off-node %d recvB %d maxmsgB %d' % (np.median(nsend), np.median(off), np.median(rbytes), np.median(rmax)))
print('\nsingle-waiter stalls:', sum(single.values()), sorted(single.items(), key=lambda x: -x[1])[:10])
print('\nsender common to every waiting rank (count of stalls), top:')
for p, c in sorted(inter.items(), key=lambda x: -x[1])[:12]:
    print(f'  sender {p:4d} node {node[p]} in {c} of {len(stalls)} stalls; sends to {nout[p]} ranks, {sbytes[p]} B')
none = sum(1 for s in stalls if not np.any(np.all(h[[int(x) for x in s["waiting"].split()]] > 0, axis=0)))
print('stalls with no sender common to all waiters:', none)
