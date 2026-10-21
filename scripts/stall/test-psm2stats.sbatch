#!/bin/bash
#SBATCH -J psm2stats
#SBATCH -p pdebug
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=112
#SBATCH --exclude=dane261,dane764
#SBATCH -o psm2stats-%j.out
# Are the 0.160 s stalls packet loss, and where? (10-19-2026.txt sec. 11-12.)
# libpsm2 keeps protocol counters (naks sent/recv, err_chk, "rcveager overflows") and
# PSM2_MQ_PRINT_STATS=-1 writes per-process stats files at finalize, named
# psm2-perf-stat-ep-<epid>-<pid> in the working directory. Each run gets its own working
# directory, and every rank records "<rank> <host> <pid>" in pidmap/<rank> before exec'ing
# the program, so each stats file can be tied to its rank and to that rank's per-iteration
# stalls (raw file). Which counters the stats files contain is read from the output, not
# assumed.
# usage: sbatch -N NODES test-psm2stats.sbatch NODES N RUNS
#   e.g. sbatch -N 2 ... 2 200 1 (format check), sbatch -N 4 ... 4 2000 2 (the measurement)
# Runs: the replay on crankseg_2.pm, -c MPI, Open MPI default (cm + psm2), k 25, n N,
# --iter-raw, RUNS times.
NODES=$1; N=$2; RUNS=$3
DIR=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-psm2stats-$NODES
M=/usr/WS2/unm_cs/NicholasBaconInputsData
BIN=$HOME/cand/steps/old_2b1acb4/bench-build/src/irregular-benchmarking
SCRIPTS=$HOME/benchmark26/irregular-benchmarking-new/scripts
SCAN=/usr/workspace/unm_cs/NicksBenchmarkDumps/data-10-19-2026-stall4/stall-scan.py
GEOM=(--nodes="$NODES" --ntasks-per-node=112)
mkdir -p "$DIR"

PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5 "$SCRIPTS/provenance.sh" "$DIR" "$BIN" \
    "$M/crankseg_2.pm" -- "${GEOM[@]}"

echo "run,start,end,exit,file" > "$DIR/manifest.csv"
for ((run = 1; run <= RUNS; run++)); do
    f=$DIR/run_$(printf %04d "$run")_crankseg_2_MPI_stats.txt
    wd=${f%.txt}.wd
    mkdir -p "$wd/pidmap"
    start=$(date -Iseconds)
    # exec keeps the pid, so the pid in pidmap is the one in the stats file name.
    PSM2_MQ_PRINT_STATS=-1 PSM2_VERBOSE_ENV=1 srun "${GEOM[@]}" --chdir="$wd" bash -c \
        'echo "$SLURM_PROCID $(hostname) $$" > pidmap/$SLURM_PROCID; exec "$@"' bash \
        "$BIN" -r -c MPI -x EXPORT -q -b -W nearest -I 1 -P crankseg_2 -k 25 -n "$N" -s NUMA -U \
        --iter-prefix "${f%.txt}" --iter-raw -f "$M/crankseg_2.pm" > "$f" 2>&1 < /dev/null
    status=$?
    echo "$run,$start,$(date -Iseconds),$status,$(basename "$f")" >> "$DIR/manifest.csv"
done

{
    echo "== stats files per run"
    for wd in "$DIR"/run_*.wd; do
        echo "$(basename "$wd"): $(ls "$wd" | grep -c '^psm2-perf-stat-ep-') stats files, $(ls "$wd/pidmap" | wc -l) pidmap entries"
    done
    echo "== one stats file, whole (rank 0 of run 1)"
    wd=$(ls -d "$DIR"/run_*.wd | head -1)
    read -r _ _ pid0 < "$wd/pidmap/0"
    cat "$(ls "$wd"/psm2-perf-stat-ep-*-"$pid0" | head -1)"
    echo "== stall-scan.py"
    python3 "$SCAN" "$DIR" > "$DIR/stall-scan.csv"
    awk -F, 'NR > 1 {n[$1]++} END {for (r in n) print "run " r ": " n[r] " flagged"}' "$DIR/stall-scan.csv" | sort
} > "$DIR/summary.txt" 2>&1
