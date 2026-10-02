#!/bin/bash
# Queue one node count: a pilot, then LAUNCHES gatherdata jobs that start only
# after the pilot succeeds and all read its targets file, so every launch at this
# node count uses the same k and n. Run it from the directory results should go to.
#
#   scripts/submit.sh NODES INPUT_FILE TARGET_SECONDS LAUNCHES [WARMUP]
set -eu

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "usage: $0 NODES INPUT_FILE TARGET_SECONDS LAUNCHES [WARMUP]" >&2
    exit 1
fi
nodes=$1
input_file=$(readlink -f "$2")
target_seconds=$3
launches=$4
warmup=${5:-nearest}
scripts=$(dirname "$(readlink -f "$0")")

# --parsable prints "jobid" or "jobid;cluster".
pilot=$(sbatch --parsable -N "$nodes" "$scripts/pilot.sbatch" \
    "$nodes" "$input_file" "$target_seconds" "$warmup" | cut -d';' -f1)
# pilot.sbatch writes to <jobid>-<job name>-<nodes>/ in the submit directory.
targets=$PWD/$pilot-pilot-$nodes/targets
echo "pilot: $pilot (targets: $targets)"

# If the pilot fails, --kill-on-invalid-dep cancels these instead of leaving them queued.
echo -n "gatherdata:"
for ((i = 0; i < launches; i++)); do
    job=$(sbatch --parsable -N "$nodes" --dependency=afterok:"$pilot" --kill-on-invalid-dep=yes \
        "$scripts/gatherdata.sbatch" "$nodes" "$input_file" "$targets" "$warmup" | cut -d';' -f1)
    echo -n " $job"
done
echo
