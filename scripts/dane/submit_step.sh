#!/bin/bash
# One pdebug job per step: STANDARD then LOCALITY, 5x1000 each, 2 nodes.
# Output names end with the time each run started.
# usage: submit_step.sh <label> [extra benchmark flags, e.g. -s NUMA]
LABEL=$1; shift
TAG=$LABEL; [ $# -gt 0 ] && TAG=${LABEL}_$(echo "$*" | tr -d ' -')
BIN=$HOME/cand/steps/$LABEL/bench-build/src/irregular-benchmarking
OUT=/g/g20/bacon4/benchmark26/irregular-benchmarking-new/build1/src
E=/g/g20/bacon4/benchmark26/baseline/build1/examples
RUN="srun --nodes=2 --ntasks-per-node=112 $BIN -r -c MPIA -I 5 -i 1000 -x EXPORT -q -S 42 -V $* -f $E/t2.json"
cd $OUT && sbatch -p pdebug --nodes=2 --ntasks=224 --time=1:00:00 -J step_$TAG -o $HOME/cand/steps/slurm_${TAG}_%j.out \
  --wrap="$RUN -a STANDARD > out_1000_STANDARD_t2_${TAG}_\$(date +%Y%m%d-%H%M%S).txt; $RUN -a LOCALITY > out_1000_LOCALITY_t2_${TAG}_\$(date +%Y%m%d-%H%M%S).txt"
