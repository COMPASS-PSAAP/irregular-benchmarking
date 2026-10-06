#!/bin/bash
#SBATCH -J gatherdata
#SBATCH -p pbatch
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -o gatherdata-%j.out

# sbatch runs a spooled copy of this file, so the checkout's path is spelled out.
REPO=/g/g20/bacon4/benchmark26/irregular-benchmarking-new
# BINARY and EXTRA_ARGS can be set in the environment at submission (sbatch passes
# it through), e.g. BINARY=<build against another locality_aware> EXTRA_ARGS="-s NUMA".
BINARY=${BINARY:-$REPO/build1/src/irregular-benchmarking}
EXTRA_ARGS=${EXTRA_ARGS:-}      # extra benchmark flags for every run, word-split
# For a BINARY built from a git archive (no checkout for provenance.sh to read),
# set BENCH_COMMIT to the benchmark commit it was built from.
BENCH_COMMIT=${BENCH_COMMIT:-}
export PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5   # read by provenance.sh
SAMPLES=1      # times each pattern is run on each arm, in shuffled order
ARMS="MPI STANDARD LOCALITY"   # MPI: plain MPI (-c MPI); the others: MPI Advance -a
SHUFFLE_SEED=   # empty: draw a new one; set it to repeat an earlier launch's order

# Usage: sbatch -N NODES scripts/gatherdata.sbatch NODES INPUTS TARGETS_FILE [WARMUP]
# INPUTS is one input file or several, comma-separated (JSON pattern configs and/or
# .pm matrices), as given to pilot.sbatch. TARGETS_FILE is the targets file the pilot
# wrote for this node count, so every launch at a node count uses the same k and n.
# WARMUP is the benchmark's -W: none, nearest (rank +/- 1, the default) or all.
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: sbatch -N NODES $0 NODES INPUTS TARGETS_FILE [WARMUP]" >&2
    exit 1
fi
nodes=$1
IFS=, read -ra input_files <<< "$2"
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
for i in "${!input_files[@]}"; do
    if [ ! -f "${input_files[$i]}" ]; then
        echo "input file not found: ${input_files[$i]}" >&2
        exit 1
    fi
    input_files[$i]=$(readlink -f "${input_files[$i]}")
done
inputs=$(IFS=,; echo "${input_files[*]}")
if [ ! -s "$targets_file" ]; then
    echo "targets file missing or empty: $3" >&2
    exit 1
fi

out_dir=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-$SLURM_JOB_NAME-$nodes
mkdir -p "$out_dir"

# Node list, binding, build, MPI, Spack, git state, and environment, recorded
# with the same srun geometry the benchmark runs use.
"$REPO/scripts/provenance.sh" "$out_dir" "$BINARY" "$inputs" -- \
    --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE"

# Keep the launch's own copy, with where it came from.
cp "$targets_file" "$out_dir/targets"
echo "$targets_file" > "$out_dir/targets_source"

# Which input file each pattern comes from, for each run's -f.
for input_file in "${input_files[@]}"; do
    python3 "$REPO/scripts/pattern-names.py" "$input_file" | sed "s|\$| $input_file|"
done > "$out_dir/pattern_files"

# Each pattern SAMPLES times on each arm, all shuffled together; the order
# actually run is kept in schedule. Every arm uses the n the STANDARD pilot found.
# The shuffle is seeded, and the seed is saved, so the same targets and seed
# give the same order again.
echo "binary: $BINARY" > "$out_dir/launch_settings"
echo "extra args: $EXTRA_ARGS" >> "$out_dir/launch_settings"
echo "benchmark commit: ${BENCH_COMMIT:-see provenance}" >> "$out_dir/launch_settings"
echo "inputs: $inputs" >> "$out_dir/launch_settings"
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
    input_file=$(awk -v p="$pattern" '$1 == p {print $2; exit}' "$out_dir/pattern_files")
    start=$(date -Iseconds)
    # srun would otherwise read the rest of the schedule from stdin.
    srun --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" "$BINARY" \
        -r "${backend[@]}" -x EXPORT -q -b -W "$warmup" -I 1 \
        -P "$pattern" -k "$k" -n "$n" $EXTRA_ARGS -f "$input_file" > "$run_file" 2>&1 < /dev/null
    status=$?
    seed=$(sed -n 's/^-seed: \([0-9]*\).*/\1/p' "$run_file")
    echo "$run,$pattern,$arm,$seed,$k,$n,$start,$(date -Iseconds),$status,$(basename "$run_file")" \
        >> "$out_dir/manifest.csv"
done < "$out_dir/schedule"
