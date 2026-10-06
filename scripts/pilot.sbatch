#!/bin/bash
#SBATCH -J pilot
#SBATCH -p pbatch
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -o pilot-%j.out

# sbatch runs a spooled copy of this file, so the checkout's path is spelled out.
REPO=/g/g20/bacon4/benchmark26/irregular-benchmarking-new
# BINARY, EXTRA_ARGS and BENCH_COMMIT can be set in the environment at submission, as
# for gatherdata.sbatch, so the pilot runs the same build and flags as the launches.
BINARY=${BINARY:-$REPO/build1/src/irregular-benchmarking}
EXTRA_ARGS=${EXTRA_ARGS:-}
BENCH_COMMIT=${BENCH_COMMIT:-}
export PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5   # read by provenance.sh
PILOT_K=25
PILOT_N=100

# Usage: sbatch -N NODES scripts/pilot.sbatch NODES INPUTS TARGET_SECONDS [WARMUP]
# INPUTS is one input file or several, comma-separated: JSON pattern configs and/or
# .pm matrices. Writes <jobid>-pilot-<nodes>/targets ("pattern k n" per line), the
# k and n every gatherdata launch at this node count then uses. WARMUP is the
# benchmark's -W: none, nearest (rank +/- 1, the default) or all.
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: sbatch -N NODES $0 NODES INPUTS TARGET_SECONDS [WARMUP]" >&2
    exit 1
fi
nodes=$1
IFS=, read -ra input_files <<< "$2"
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
for i in "${!input_files[@]}"; do
    if [ ! -f "${input_files[$i]}" ]; then
        echo "input file not found: ${input_files[$i]}" >&2
        exit 1
    fi
    input_files[$i]=$(readlink -f "${input_files[$i]}")
done
inputs=$(IFS=,; echo "${input_files[*]}")

out_dir=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-$SLURM_JOB_NAME-$nodes
mkdir -p "$out_dir"

"$REPO/scripts/provenance.sh" "$out_dir" "$BINARY" "$inputs" -- \
    --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE"
{
    echo "binary: $BINARY"
    echo "extra args: $EXTRA_ARGS"
    echo "benchmark commit: ${BENCH_COMMIT:-see provenance}"
    echo "inputs: $inputs"
} > "$out_dir/launch_settings"

# One sample of every pattern on STANDARD, one srun per input file. --calc solves
# each pattern's window and timed totals for q and startup and prints the n that
# makes one sample take TARGET_SECONDS.
: > "$out_dir/targets"
expected=0
for input_file in "${input_files[@]}"; do
    pilot_file=$out_dir/pilot_$(basename "$input_file")_$(date +%Y%m%d-%H%M%S).txt
    srun --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" "$BINARY" \
        -r -c MPIA -a STANDARD -x EXPORT -q -b -W "$warmup" -I 1 \
        -k "$PILOT_K" -n "$PILOT_N" --calc "$target_seconds" $EXTRA_ARGS -f "$input_file" \
        > "$pilot_file" 2>&1 < /dev/null
    # "(pattern_0)  calc: q Q ms  startup S ms  n N  (budget T s, k K)" -> "pattern_0 K N"
    grep 'calc:' "$pilot_file" | awk '{gsub(/[()]/, "", $1); gsub(/\)/, "", $15); print $1, $15, $10}' \
        >> "$out_dir/targets"
    expected=$((expected + $(python3 "$REPO/scripts/pattern-names.py" "$input_file" | wc -l)))
done

# Fail the job unless every pattern got a positive n and no name came from two
# files, so launches that depend on it (afterok) never start with a partial or
# ambiguous targets file.
good=$(awk '$3 >= 1' "$out_dir/targets" | wc -l)
if [ "$good" -ne "$expected" ]; then
    echo "pilot produced a usable n for $good of $expected patterns; see $out_dir/pilot_*.txt" >&2
    exit 1
fi
if [ "$(cut -d' ' -f1 "$out_dir/targets" | sort | uniq -d | wc -l)" -ne 0 ]; then
    echo "the same pattern name comes from more than one input file" >&2
    exit 1
fi
