#!/bin/bash
# Queue one node count: a pilot, then LAUNCHES gatherdata jobs that start only
# after the pilot succeeds and all read its targets file, so every launch at this
# node count uses the same k and n. Run it from the directory results should go to.
#
#   scripts/submit.sh NODES INPUTS TARGET_SECONDS LAUNCHES [WARMUP]
#
# INPUTS is one input file or several, comma-separated (JSON pattern configs and/or
# .pm matrices). BINARY, EXTRA_ARGS and BENCH_COMMIT in the environment reach both
# the pilot and the launches. MAX_RUNNING (env), if set, makes launch i also wait
# for launch i - MAX_RUNNING to end, so at most that many of this call's launches
# run at once and the rest are spread over time instead of landing together.
# EXCLUDE_NODES (env), if set, goes to every sbatch as --exclude (a Slurm node list).
# ABCG_SPMV and ABCG_CG reach both scripts the same way (see pilot.sbatch).
set -eu

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "usage: $0 NODES INPUTS TARGET_SECONDS LAUNCHES [WARMUP]" >&2
    exit 1
fi
nodes=$1
IFS=, read -ra input_files <<< "$2"
for i in "${!input_files[@]}"; do
    input_files[$i]=$(readlink -f "${input_files[$i]}")
done
inputs=$(IFS=,; echo "${input_files[*]}")
target_seconds=$3
launches=$4
warmup=${5:-nearest}
max_running=${MAX_RUNNING:-0}
exclude=()
if [ -n "${EXCLUDE_NODES:-}" ]; then
    exclude=(--exclude="$EXCLUDE_NODES")
fi
scripts=$(dirname "$(readlink -f "$0")")

# --parsable prints "jobid" or "jobid;cluster".
pilot=$(sbatch --parsable -N "$nodes" "${exclude[@]}" "$scripts/pilot.sbatch" \
    "$nodes" "$inputs" "$target_seconds" "$warmup" | cut -d';' -f1)
# pilot.sbatch writes to <jobid>-<job name>-<nodes>/ in the submit directory.
targets=$PWD/$pilot-pilot-$nodes/targets
echo "pilot: $pilot (targets: $targets)"

# If the pilot fails, --kill-on-invalid-dep cancels these instead of leaving them queued.
echo -n "gatherdata:"
jobs=()
for ((i = 0; i < launches; i++)); do
    dependency=afterok:$pilot
    # afterany, not afterok: a failed launch must not cancel the rest of the chain.
    if [ "$max_running" -gt 0 ] && [ "$i" -ge "$max_running" ]; then
        dependency=$dependency,afterany:${jobs[i - max_running]}
    fi
    job=$(sbatch --parsable -N "$nodes" "${exclude[@]}" --dependency="$dependency" --kill-on-invalid-dep=yes \
        "$scripts/gatherdata.sbatch" "$nodes" "$inputs" "$targets" "$warmup" | cut -d';' -f1)
    jobs+=("$job")
    echo -n " $job"
done
echo
