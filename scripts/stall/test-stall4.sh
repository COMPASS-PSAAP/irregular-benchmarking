#!/bin/bash
# 0.160 s stall test at 4 nodes (todo item 1), and the smoke run of the rebuild from
# the per-iteration/provenance commits. 4 launches of the campaign's own gatherdata.sbatch
# (pbatch, 4 nodes, 2 h limit each), each running the 7 crankseg_2 patterns of the 10-13
# campaign (crankseg_2, coarse_, fine_, abcgspmv_, abcgcg_, abcgspmvnp_, abcgcgnp_) on
# MPI, STANDARD and LOCALITY in a random order: 21 runs per launch, with k and n from that
# campaign's 4-node pilot (8013780) and its Vernier capture, so the runs repeat the
# campaign's 4-node crankseg_2 cells exactly, now with ITER=1 ITER_RAW=1 (every rank's
# per-iteration times plus clock offsets; at most about 0.25 GB per run here).
# Question: in the MPI/STANDARD runs, are single iterations about 160 ms (or multiples),
# on which ranks/nodes, inside apply or inside the barrier?
# Pass (smoke): every manifest 21 rows exit 0; verify PASS on every benchmark
# STANDARD/LOCALITY run; no ABCG "check FAILED"; each run has .iter.npy, .raw.npy and
# .offsets.npy; each output's PROVENANCE/source commit names the new commit.
# usage: test-stall4.sh BENCH_LABEL ABCG_DIR   (the rebuild's ~/cand/steps/<label> and
#        ~/benchmark26/ABCG-<short>)
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 BENCH_LABEL ABCG_DIR" >&2; exit 1; }
DIR=/usr/workspace/unm_cs/NicksBenchmarkDumps/data-10-19-2026-stall4
SCRIPTS=/g/g20/bacon4/benchmark26/irregular-benchmarking-new/scripts
M=/usr/WS2/unm_cs/NicholasBaconInputsData
PILOT=/usr/workspace/unm_cs/NicksBenchmarkDumps/data-10-13-2026-five-way-40/campaign/8013780-pilot-4
A=$2
export BINARY=$HOME/cand/steps/$1/bench-build/src/irregular-benchmarking
export EXTRA_ARGS="-s NUMA -V -U"
export ABCG_SPMV=$A/build-old/benchmarks/benchmark_spmv
export ABCG_CG=$A/build-old/benchmarks/benchmark_cg_pilot
export ABCG_SPMV_MPI=$A/build-mpi/benchmarks/benchmark_spmv
export ABCG_CG_MPI=$A/build-mpi/benchmarks/benchmark_cg_pilot
export ABCG_EXTRA_ARGS="-s NUMA"
export ITER=1 ITER_RAW=1
for bin in "$BINARY" "$ABCG_SPMV" "$ABCG_CG" "$ABCG_SPMV_MPI" "$ABCG_CG_MPI"; do
    [ -x "$bin" ] || { echo "missing binary: $bin" >&2; exit 1; }
done
mkdir -p "$DIR/targets-4"
cd "$DIR"
# gatherdata adds the generated_inputs found next to the targets file (the pilot's
# coarse.json and capture directories), so it is copied beside the filtered targets.
grep -E '^(coarse_|fine_|abcgspmv_|abcgcg_|abcgspmvnp_|abcgcgnp_)?crankseg_2 ' "$PILOT/targets" \
    > targets-4/targets
cp "$PILOT/generated_inputs" targets-4/generated_inputs
[ "$(wc -l < targets-4/targets)" -eq 7 ] || { echo "expected 7 targets" >&2; exit 1; }
for i in 1 2 3 4; do
    sbatch -p pbatch -t 2:00:00 -N 4 --exclude=dane261,dane764 \
        "$SCRIPTS/gatherdata.sbatch" 4 "$M/crankseg_2.pm" "$DIR/targets-4/targets" nearest
done 2>&1 | tee -a submit_log.txt
