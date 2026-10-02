#!/bin/bash
#SBATCH -J gatherdata
#SBATCH -p pbatch
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -o gatherdata-%j.out

# sbatch runs a spooled copy of this file, so the checkout's path is spelled out.
REPO=/g/g20/bacon4/benchmark26/irregular-benchmarking-new
BINARY=$REPO/build1/src/irregular-benchmarking
export PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5   # read by provenance.sh
SAMPLES=1      # times each pattern is run on each arm, in shuffled order
ARMS="MPI STANDARD LOCALITY"   # MPI: plain MPI (-c MPI); the others: MPI Advance -a
SHUFFLE_SEED=   # empty: draw a new one; set it to repeat an earlier launch's order

# Usage: sbatch -N NODES scripts/gatherdata.sbatch NODES INPUT_FILE TARGETS_FILE [WARMUP]
# TARGETS_FILE is the targets file pilot.sbatch wrote for this node count, so every
# launch at a node count uses the same k and n. WARMUP is the benchmark's -W:
# none, nearest (rank +/- 1, the default) or all.
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: sbatch -N NODES $0 NODES INPUT_FILE TARGETS_FILE [WARMUP]" >&2
    exit 1
fi
nodes=$1
input_file=$(readlink -f "$2")
targets_file=$(readlink -f "$3")
warmup=${4:-nearest}
case $warmup in
    none|nearest|all) ;;
    *) echo "WARMUP must be none, nearest or all, not $warmup" >&2; exit 1 ;;
esac

# sbatch fixes the allocation before this script starts, so NODES can only be
# checked against it, not change it.
if [ "$nodes" != "$SLURM_JOB_NUM_NODES" ]; then
    echo "NODES is $nodes but the job has $SLURM_JOB_NUM_NODES nodes; submit with sbatch -N $nodes" >&2
    exit 1
fi
if [ ! -f "$input_file" ]; then
    echo "input file not found: $2" >&2
    exit 1
fi
if [ ! -s "$targets_file" ]; then
    echo "targets file missing or empty: $3" >&2
    exit 1
fi

out_dir=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-$SLURM_JOB_NAME-$nodes
mkdir -p "$out_dir"

# Node list, binding, build, MPI, Spack, git state, and environment, recorded
# with the same srun geometry the benchmark runs use.
"$REPO/scripts/provenance.sh" "$out_dir" "$BINARY" "$input_file" -- \
    --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE"

# Keep the launch's own copy, with where it came from.
cp "$targets_file" "$out_dir/targets"
echo "$targets_file" > "$out_dir/targets_source"

# Each pattern SAMPLES times on each arm, all shuffled together; the order
# actually run is kept in schedule. Every arm uses the n the STANDARD pilot found.
# The shuffle is seeded, and the seed is saved, so the same targets and seed
# give the same order again.
shuffle_seed=${SHUFFLE_SEED:-$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}
echo "$shuffle_seed" > "$out_dir/shuffle_seed"
while read -r pattern k n; do
    for arm in $ARMS; do
        for ((i = 0; i < SAMPLES; i++)); do
            echo "$pattern $k $n $arm"
        done
    done
done < "$out_dir/targets" | python3 -c '
import random, sys
lines = sys.stdin.readlines()
random.Random(int(sys.argv[1])).shuffle(lines)
sys.stdout.writelines(lines)' "$shuffle_seed" > "$out_dir/schedule"

# One srun per sample, each with its pattern's k and n from targets. The
# manifest gets one row per run, so a failed run is visible without opening files.
echo "run,pattern,arm,seed,k,n,start,end,exit,file" > "$out_dir/manifest.csv"
run=0
while read -r pattern k n arm; do
    run=$((run + 1))
    run_file=$out_dir/run_$(printf %04d "$run")_${pattern}_${arm}_$(date +%Y%m%d-%H%M%S).txt
    if [ "$arm" = MPI ]; then
        backend=(-c MPI)
    else
        backend=(-c MPIA -a "$arm")
    fi
    start=$(date -Iseconds)
    # srun would otherwise read the rest of the schedule from stdin.
    srun --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" "$BINARY" \
        -r "${backend[@]}" -x EXPORT -q -b -W "$warmup" -I 1 \
        -P "$pattern" -k "$k" -n "$n" -f "$input_file" > "$run_file" 2>&1 < /dev/null
    status=$?
    seed=$(sed -n 's/^-seed: \([0-9]*\).*/\1/p' "$run_file")
    echo "$run,$pattern,$arm,$seed,$k,$n,$start,$(date -Iseconds),$status,$(basename "$run_file")" \
        >> "$out_dir/manifest.csv"
done < "$out_dir/schedule"
