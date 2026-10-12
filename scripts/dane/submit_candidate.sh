#!/bin/bash
# Submit one 25x1000 run of a candidate build; output goes next to the other
# results with the job's start time in the name.
# usage: submit_candidate.sh <label> <nodes> <LOCALITY|STANDARD> [extra benchmark flags]
LABEL=$1; N=$2; MODE=$3; shift 3
BIN=$HOME/cand/$LABEL/bench-build/src/irregular-benchmarking
OUT=/g/g20/bacon4/benchmark26/irregular-benchmarking-new/build1/src
E=/g/g20/bacon4/benchmark26/baseline/build1/examples
cd $OUT && sbatch -p pdebug --nodes=$N --ntasks=$((N*112)) --time=1:00:00 \
  --wrap="srun --nodes=$N --ntasks-per-node=112 $BIN -r -c MPIA -I 25 -i 1000 -a $MODE -x EXPORT -q -S 42 -V $* -f $E/t$N.json > out_1000_${MODE}_t${N}_${LABEL}_\$(date +%Y%m%d-%H%M%S).txt"
