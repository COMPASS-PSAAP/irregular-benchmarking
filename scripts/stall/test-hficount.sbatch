#!/bin/bash
#SBATCH -J hficount
#SBATCH -p pdebug
#SBATCH -t 1:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=112
#SBATCH --exclude=dane261,dane764
#SBATCH -o hficount-%j.out
# Are the PSM2 error-check stalls caused by dropped packets, and where are they dropped?
# (10-20-2026.txt sec. 1.) 8065849 showed the stall length is PSM2_ERRCHK_TIMEOUT's
# minimum (160, 20, 5 ms), and on dane[35-38] nearly every iteration waited once. The PSM2
# stats file (8065876) holds only MQ counters, so this reads the hfi1 driver counters
# instead: every file under /sys/class/infiniband/hfi1_0/{hw_counters,ports/1/hw_counters,
# ports/1/counters} on every node, before and after each run. Host-side receive drops show
# as RcvOverflow, RcvHdrOvr<ctxt>, DRIVER_EgrBufFull, DRIVER_EgrHdrFull, RxTIDFullEr; port and
# fabric loss as PktDrop, port_rcv_errors, port_xmit_discards, DcDroppedPkt. If loss is the
# cause, some counter grows by about one per stalled iteration (n 2000 about 10x n 200) and
# not in the idle intervals; if none does, the timer fires without a drop (for example, an
# ACK the sender waits on is not sent until the error check).
#   idle            no program, 60 s (background growth of the counters)
#   replay-5-2000   replay, crankseg_2.pm, -c MPI, Open MPI default, PSM2_ERRCHK_TIMEOUT
#                   5:640:2, n 2000 (the 5 ms minimum keeps runs short)
#   replay-5-200    same, n 200
#   replay-def-200  same, PSM2_ERRCHK_TIMEOUT unset, n 200 (counts should match replay-5-200)
# k 25, per-iteration raw files; forward then reverse order.
# Read: summary.txt (nonzero counter deltas per run and node, stalls per run).
DIR=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-hficount
M=/usr/WS2/unm_cs/NicholasBaconInputsData
BIN=$HOME/cand/steps/old_2b1acb4/bench-build/src/irregular-benchmarking
SCRIPTS=$HOME/benchmark26/irregular-benchmarking-new/scripts
SCAN=/usr/workspace/unm_cs/NicksBenchmarkDumps/data-10-19-2026-stall4/stall-scan.py
GEOM=(--nodes=4 --ntasks-per-node=112)
mkdir -p "$DIR"

PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5 "$SCRIPTS/provenance.sh" "$DIR" "$BIN" \
    "$M/crankseg_2.pm" -- "${GEOM[@]}"

# Writes every hfi1 counter on every node, as "path:value" lines, to $1/<host>.
snapshot() {
    mkdir -p "$1"
    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c 'H=/sys/class/infiniband/hfi1_0
        grep -r . $H/hw_counters $H/ports/1/hw_counters $H/ports/1/counters > "$0/$(hostname)"' "$1"
}

echo "run,config,start,end,exit,file" > "$DIR/manifest.csv"
run=0
for cfg in idle replay-5-2000 replay-5-200 replay-def-200 \
           replay-def-200 replay-5-200 replay-5-2000 idle; do
    run=$((run + 1))
    f=$DIR/run_$(printf %04d "$run")_crankseg_2_MPI_$cfg.txt
    case $cfg in
        replay-5-*) env=(PSM2_VERBOSE_ENV=1 PSM2_ERRCHK_TIMEOUT=5:640:2) ;;
        *)          env=(PSM2_VERBOSE_ENV=1) ;;
    esac
    snapshot "${f%.txt}.before"
    start=$(date -Iseconds)
    if [ "$cfg" = idle ]; then
        sleep 60; status=0; : > "$f"
    else
        env "${env[@]}" srun "${GEOM[@]}" "$BIN" -r -c MPI -x EXPORT -q -b -W nearest -I 1 \
            -P crankseg_2 -k 25 -n "${cfg##*-}" -s NUMA -U --iter-prefix "${f%.txt}" --iter-raw \
            -f "$M/crankseg_2.pm" > "$f" 2>&1 < /dev/null
        status=$?
    fi
    echo "$run,$cfg,$start,$(date -Iseconds),$status,$(basename "$f")" >> "$DIR/manifest.csv"
    snapshot "${f%.txt}.after"
done

{
    echo "== PSM2_ERRCHK_TIMEOUT as libpsm2 reports it (distinct lines per run)"
    for f in "$DIR"/run_*.txt; do
        echo "$(basename "$f"):"
        grep -h "ERRCHK" "$f" | sed 's/^[^ ]*: *//' | sort | uniq -c | head -2
    done
    echo "== stalls per run (iterations over 2 ms, from the reduced files)"
    python3 "$SCAN" "$DIR" 2 > "$DIR/stall-scan.csv"
    python3 - "$DIR" <<'PY'
import glob, os, sys
import numpy as np
for p in sorted(glob.glob(os.path.join(sys.argv[1], 'run_*.iter.npy'))):
    t = np.load(p)['iter_max_ms']
    slow = t[t > 2.0]
    print(f'{os.path.basename(p)[:40]:40s} iterations {len(t):5d}  over 2 ms {len(slow):5d}  '
          f'median {np.median(t):.3f} ms')
PY
    echo "== counter deltas per run and node (after minus before, nonzero only; all of them"
    echo "   are in deltas.csv, here throughput and interrupt counters are left out and the"
    echo "   per-context RcvHdrOvr counters are summed into one line)"
    python3 - "$DIR" <<'PY'
import glob, os, re, sys
def load(path):
    d = {}
    for line in open(path):
        k, _, v = line.rstrip('\n').rpartition(':')
        d[k.replace('/sys/class/infiniband/hfi1_0/', '')] = int(v)
    return d
# Only names that are certainly throughput, interrupt, or gauge counts; anything unknown is shown.
traffic = re.compile(r'(TxPkt|TxPktVL\d+|RxPkt|TxWords|RxWords|TxFlitVL\d+|TxWaitVL\d+|Intr'
                     r'|Cce\w*Int|SDMA(Id|Pr)?Int(Cn)?\d+|SDEDscFdCn\d+|SendSched|lifespan'
                     r'|Dc(PrfRxFlit|PrfTxFlit|PrfClk|RcvFlits|XmitFlits|RcvPkts|XmitPkts|GoodLTP'
                     r'|AccLTP|RxPktVl\d+|RxFlitVl\d+)|DRIVER_(KernIntr|CtxtsOpen)|RcAcks|RcQacks'
                     r'|port_(xmit|rcv)_(data|packets)|unicast_(xmit|rcv)_packets)(,32)?')
out = open(os.path.join(sys.argv[1], 'deltas.csv'), 'w')
out.write('run,host,counter,delta\n')
for before in sorted(glob.glob(os.path.join(sys.argv[1], 'run_*.before'))):
    run = os.path.basename(before)[:-len('.before')]
    print(run)
    for b in sorted(glob.glob(os.path.join(before, '*'))):
        host = os.path.basename(b)
        a = os.path.join(before[:-len('.before')] + '.after', host)
        if not os.path.exists(a):
            print(f'  {host}: no after snapshot'); continue
        x, y = load(b), load(a)
        delta = {k: y[k] - x[k] for k in x if k in y and y[k] != x[k]}
        for k, v in sorted(delta.items()):
            out.write(f'{run},{host},"{k}",{v}\n')
        ovr = {k: v for k, v in delta.items() if '/RcvHdrOvr' in k}
        rest = {k: v for k, v in delta.items()
                if k not in ovr and not traffic.fullmatch(k.rpartition('/')[2])}
        tx = delta.get('ports/1/counters/port_xmit_packets', 0)
        rx = delta.get('ports/1/counters/port_rcv_packets', 0)
        line = f'  {host}: xmit_pkts {tx} rcv_pkts {rx}'
        if ovr:
            line += f'  RcvHdrOvr sum {sum(ovr.values())} over {len(ovr)} contexts'
        print(line)
        for k, v in sorted(rest.items()):
            print(f'      {k} {v:+d}')
PY
} > "$DIR/summary.txt" 2>&1
