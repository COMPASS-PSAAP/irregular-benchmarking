#!/bin/bash
#SBATCH -J pilot
#SBATCH -p pbatch
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -o pilot-%j.out

# sbatch runs a spooled copy of this file, so the checkout's path is spelled out.
REPO=/g/g20/bacon4/benchmark26/irregular-benchmarking-new
BINARY=$REPO/build1/src/irregular-benchmarking
export PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5   # read by provenance.sh
PILOT_K=25
PILOT_N=100

# Usage: sbatch -N NODES scripts/pilot.sbatch NODES INPUT_FILE TARGET_SECONDS [WARMUP]
# Writes <jobid>-pilot-<nodes>/targets ("pattern k n" per line), the k and n every
# gatherdata launch at this node count then uses. WARMUP is the benchmark's -W:
# none, nearest (rank +/- 1, the default) or all.
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: sbatch -N NODES $0 NODES INPUT_FILE TARGET_SECONDS [WARMUP]" >&2
    exit 1
fi
nodes=$1
input_file=$(readlink -f "$2")
target_seconds=$3
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

out_dir=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-$SLURM_JOB_NAME-$nodes
mkdir -p "$out_dir"

"$REPO/scripts/provenance.sh" "$out_dir" "$BINARY" "$input_file" -- \
    --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE"

# One sample of every pattern on STANDARD. --calc solves each pattern's window and
# timed totals for q and startup and prints the n that makes one sample take
# TARGET_SECONDS.
pilot_file=$out_dir/pilot_$(date +%Y%m%d-%H%M%S).txt
srun --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" "$BINARY" \
    -r -c MPIA -a STANDARD -x EXPORT -q -b -W "$warmup" -I 1 \
    -k "$PILOT_K" -n "$PILOT_N" --calc "$target_seconds" -f "$input_file" > "$pilot_file" 2>&1

# "(pattern_0)  calc: q Q ms  startup S ms  n N  (budget T s, k K)" -> "pattern_0 K N"
grep 'calc:' "$pilot_file" | awk '{gsub(/[()]/, "", $1); gsub(/\)/, "", $15); print $1, $15, $10}' \
    > "$out_dir/targets"

# Fail the job unless every pattern got a positive n, so launches that depend on
# it (afterok) never start with a partial or unusable targets file.
expected=$(python3 "$REPO/scripts/pattern-names.py" "$input_file" | wc -l)
good=$(awk '$3 >= 1' "$out_dir/targets" | wc -l)
if [ "$good" -ne "$expected" ]; then
    echo "pilot produced a usable n for $good of $expected patterns; see $pilot_file" >&2
    exit 1
fi
