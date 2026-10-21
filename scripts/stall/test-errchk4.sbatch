#!/bin/bash
#SBATCH -J errchk4
#SBATCH -p pdebug
#SBATCH -t 1:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=112
#SBATCH --exclude=dane261,dane764
#SBATCH -o errchk4-%j.out
# Is the 0.160 s stall PSM2's error-check (retransmit) timeout? (10-19-2026.txt sec. 11.)
# libpsm2 on dane has PSM2_ERRCHK_TIMEOUT "<min:max:factor>" in ms, default 160:640:2, and
# every transport that ran in 8065747 (Open MPI cm-psm2, cm-ofi via psmx2, MVAPICH2
# ch3:psm) stalled for exactly 160-164 ms. Here the same runs are repeated on pdebug (the
# earlier tests ran on pbatch) with that minimum lowered. If the timeout is the cause, the
# stalls become about 20 ms and 5 ms; if they stay at 160 ms, it is not.
#   replay-default   replay, crankseg_2.pm, -c MPI, Open MPI default (cm + psm2), unset
#   replay-20        same, PSM2_ERRCHK_TIMEOUT=20:640:2
#   replay-5         same, PSM2_ERRCHK_TIMEOUT=5:640:2
#   mv2-default      ABCG SpMV (MPI only, --persistent off), MVAPICH2 2.3.7, unset
#   mv2-20           same, PSM2_ERRCHK_TIMEOUT=20:640:2
# k 25, n 2000, per-iteration raw files; forward then reverse order. PSM2_VERBOSE_ENV=1 makes
# libpsm2 print the settings it read, so each log shows whether the value took effect.
# Read: summary.txt (stall durations per run), stall-scan.csv.
DIR=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-errchk4
M=/usr/WS2/unm_cs/NicholasBaconInputsData
BIN=$HOME/cand/steps/old_2b1acb4/bench-build/src/irregular-benchmarking
A=$HOME/benchmark26/ABCG-7fde517
MV=/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-13.3.1
SCRIPTS=$HOME/benchmark26/irregular-benchmarking-new/scripts
SCAN=/usr/workspace/unm_cs/NicksBenchmarkDumps/data-10-19-2026-stall4/stall-scan.py
GEOM=(--nodes=4 --ntasks-per-node=112)
mkdir -p "$DIR"

PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5 "$SCRIPTS/provenance.sh" "$DIR" "$BIN" \
    "$M/crankseg_2.pm" -- "${GEOM[@]}"

# Sets env and cmd for config $1, writing to $2 (the run file without .txt).
config_run() {
    replay=("$BIN" -r -c MPI -x EXPORT -q -b -W nearest -I 1 -P crankseg_2 -k 25 -n 2000
            -s NUMA -U --iter-prefix "$2" --iter-raw -f "$M/crankseg_2.pm")
    abcg=("$A/build-mvapich2/benchmarks/benchmark_spmv" "$M/crankseg_2.pm" -c MPI
          -P abcgspmvnp_crankseg_2 -W nearest -k 25 -n 2000 --persistent off
          --iter-prefix "$2" --iter-raw)
    # As in 8065747: MV2_ENABLE_AFFINITY=0 keeps Slurm's one-CPU binding.
    mv2=(LD_LIBRARY_PATH=$MV/lib:$LD_LIBRARY_PATH MV2_ENABLE_AFFINITY=0)
    case $1 in
        replay-default) env=(PSM2_VERBOSE_ENV=1); cmd=("${replay[@]}") ;;
        replay-20)      env=(PSM2_VERBOSE_ENV=1 PSM2_ERRCHK_TIMEOUT=20:640:2); cmd=("${replay[@]}") ;;
        replay-5)       env=(PSM2_VERBOSE_ENV=1 PSM2_ERRCHK_TIMEOUT=5:640:2); cmd=("${replay[@]}") ;;
        mv2-default)    env=("${mv2[@]}" PSM2_VERBOSE_ENV=1); cmd=("${abcg[@]}") ;;
        mv2-20)         env=("${mv2[@]}" PSM2_VERBOSE_ENV=1 PSM2_ERRCHK_TIMEOUT=20:640:2); cmd=("${abcg[@]}") ;;
    esac
}

echo "run,config,start,end,exit,file" > "$DIR/manifest.csv"
run=0
for cfg in replay-default replay-20 replay-5 mv2-default mv2-20 \
           mv2-20 mv2-default replay-5 replay-20 replay-default; do
    run=$((run + 1))
    f=$DIR/run_$(printf %04d "$run")_crankseg_2_MPI_$cfg.txt
    config_run "$cfg" "${f%.txt}"
    start=$(date -Iseconds)
    env "${env[@]}" srun "${GEOM[@]}" "${cmd[@]}" > "$f" 2>&1 < /dev/null
    status=$?
    echo "$run,$cfg,$start,$(date -Iseconds),$status,$(basename "$f")" >> "$DIR/manifest.csv"
done

{
    echo "== PSM2_ERRCHK_TIMEOUT as libpsm2 reports it (distinct lines per run)"
    for f in "$DIR"/run_*.txt; do
        echo "$(basename "$f"):"
        grep -h "ERRCHK" "$f" | sed 's/^[^ ]*: *//' | sort | uniq -c | head -3
    done
    echo "== timedTotal per run (ms; min max avg over ranks)"
    grep -H "timedTotal" "$DIR"/run_*.txt | sed "s#$DIR/##"
    echo "== stalls per run: iterations over 10x median and 20 ms (stall-scan), and the"
    echo "   iteration-time distribution over 2 ms from the reduced files"
    python3 "$SCAN" "$DIR" 2 > "$DIR/stall-scan.csv"
    python3 - "$DIR" <<'PY'
import glob, os, sys
import numpy as np
for p in sorted(glob.glob(os.path.join(sys.argv[1], 'run_*.iter.npy'))):
    t = np.load(p)['iter_max_ms']
    med = np.median(t); slow = t[t > 2.0]
    bins = [(2, 10), (10, 30), (30, 100), (100, 170), (170, 1e9)]
    counts = ' '.join(f'{lo}-{hi if hi < 1e9 else "inf"}:{int(((slow >= lo) & (slow < hi)).sum())}' for lo, hi in bins)
    print(f'{os.path.basename(p)[:40]:40s} median {med:.3f} ms  slow {len(slow):4d}  {counts}  '
          f'slow median {np.median(slow) if len(slow) else 0:.1f} ms')
PY
} > "$DIR/summary.txt" 2>&1
