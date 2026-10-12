#!/bin/bash
# One pdebug job: 3 back-to-back runs of benchmark_cg_mpil on Dubcova2.pm, 2 nodes.
# Output names end with the time each run started. ~/cand/libcountsr.so is
# preloaded so each output ends with a COUNTS line (which MPI paths ran).
# usage: submit_abcg.sh <label> [matrix.pm]   (default: ABCG-new test_data/Dubcova2.pm)
LABEL=$1
BIN=$HOME/cand/abcg/$LABEL/build/benchmarks/benchmark_cg_mpil
OUT=/g/g20/bacon4/benchmark26/ABCG-new/results
M=${2:-/g/g20/bacon4/benchmark26/ABCG-new/test_data/Dubcova2.pm}
# Tag outputs with the matrix name unless it is the default Dubcova2.
TAG=$LABEL; [ -n "$2" ] && TAG=${LABEL}_$(basename $M .pm)
mkdir -p $OUT
cd $OUT && sbatch -p pdebug --nodes=2 --ntasks=224 --time=1:00:00 -J abcg_$TAG -o $HOME/cand/abcg/slurm_${TAG}_%j.out \
  --wrap="for r in 1 2 3; do srun --nodes=2 --ntasks-per-node=112 --export=ALL,LD_PRELOAD=$HOME/cand/libcountsr.so $BIN $M > out_abcg_${TAG}_\$(date +%Y%m%d-%H%M%S).txt; done"
