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
# ABCG_SPMV and ABCG_CG (env), if set, are ABCG's benchmark_spmv (halo exchange only)
# and benchmark_cg_pilot (CG iterations). Each .pm input then also gets the patterns
# abcgspmv_<matrix> and abcgcg_<matrix>, sized here by that program's own STANDARD pilot.
# ABCG_PERSISTENT lists the MPI Advance modes to run, "on" (default), "off" or "on off":
# off adds abcgspmvnp_<matrix> and abcgcgnp_<matrix>, run with --persistent off (one MPI
# Advance call per exchange instead of the persistent init/start/wait).
ABCG_SPMV=${ABCG_SPMV:-}
ABCG_CG=${ABCG_CG:-}
ABCG_PERSISTENT=${ABCG_PERSISTENT:-on}
ABCG_EXTRA_ARGS=${ABCG_EXTRA_ARGS:-}   # extra flags for every ABCG run
# VERNIER_SPMV (env), if set, is ABCG's benchmark_spmv built with ABCG_VERNIER (MPI arm
# only, no MPI Advance). Before the pilots, it captures one SpMV exchange of each .pm
# input, one srun each, in INPUTS order, into capture/NN_<matrix>/; capture/order.csv
# records that order, since Vernier names its files only by rank. scripts/vernier-config.py
# then turns the captures into coarse.json (coarse_<matrix>, the same histograms for
# every rank) and the capture directories themselves (fine_<matrix>, each rank its own
# histograms), which are piloted like any other input and listed in generated_inputs
# for gatherdata.sbatch.
VERNIER_SPMV=${VERNIER_SPMV:-}
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
# "prefix binary persistent" per ABCG pattern family.
abcg_programs=()
for mode in $ABCG_PERSISTENT; do
    case $mode in
        on) suffix= ;;
        off) suffix=np ;;
        *) echo "ABCG_PERSISTENT takes on and/or off, not $mode" >&2; exit 1 ;;
    esac
    [ -n "$ABCG_SPMV" ] && abcg_programs+=("abcgspmv$suffix $ABCG_SPMV $mode")
    [ -n "$ABCG_CG" ] && abcg_programs+=("abcgcg$suffix $ABCG_CG $mode")
done
for program in "${abcg_programs[@]}"; do
    read -r prefix bin mode <<< "$program"
    [ -x "$bin" ] || { echo "$prefix program not executable: $bin" >&2; exit 1; }
done
if [ -n "$VERNIER_SPMV" ]; then
    [ -x "$VERNIER_SPMV" ] || { echo "VERNIER_SPMV not executable: $VERNIER_SPMV" >&2; exit 1; }
    case ",$inputs," in
        *.pm,*) ;;
        *) echo "VERNIER_SPMV is set but INPUTS has no .pm matrix to capture" >&2; exit 1 ;;
    esac
fi

out_dir=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-$SLURM_JOB_NAME-$nodes
mkdir -p "$out_dir"

"$REPO/scripts/provenance.sh" "$out_dir" "$BINARY" "$inputs" -- \
    --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE"
{
    echo "binary: $BINARY"
    echo "extra args: $EXTRA_ARGS"
    echo "benchmark commit: ${BENCH_COMMIT:-see provenance}"
    echo "inputs: $inputs"
    for program in "${abcg_programs[@]}"; do
        read -r prefix bin mode <<< "$program"
        echo "$prefix binary: $bin (sha256 $(sha256sum "$bin" | cut -d' ' -f1)), persistent $mode"
    done
    [ ${#abcg_programs[@]} -gt 0 ] && echo "abcg extra args: $ABCG_EXTRA_ARGS"
    [ -n "$VERNIER_SPMV" ] && echo "vernier capture: $out_dir/capture (see vernier_settings and order.csv there)"
} > "$out_dir/launch_settings"

if [ -n "$VERNIER_SPMV" ]; then
    capture_dir=$out_dir/capture
    mkdir -p "$capture_dir"
    vernier_lib=$(ldd "$VERNIER_SPMV" | awk '/libvernier/ {print $3}')
    # VERNIER_NO_REQUESTS: with request tracking on, Vernier 09d650b aborts every rank
    # but 0 in MPI_Finalize (its aggregate printer waits with MPI_Wait, not PMPI_Wait,
    # on a request it never recorded). Off, the trace shows "-" for request ids; the
    # histograms do not use them.
    capture_env="VERNIER_NO_REQUESTS=1 VERNIER_NO_HIERARCHICAL_HISTOGRAM_MANAGER=1"
    {
        echo "capture program: $VERNIER_SPMV (sha256 $(sha256sum "$VERNIER_SPMV" | cut -d' ' -f1))"
        echo "libvernier: $vernier_lib (sha256 $(sha256sum "$vernier_lib" | cut -d' ' -f1))"
        echo "vernier commit: $(cat "$(dirname "$vernier_lib")/../vernier-commit" 2>/dev/null || echo unknown)"
        echo "capture command: srun --nodes=$nodes --ntasks-per-node=$SLURM_NTASKS_PER_NODE $VERNIER_SPMV MATRIX.pm -c MPI -k 0 -n 1 -W none"
        echo "capture environment: $capture_env"
    } > "$capture_dir/vernier_settings"
    echo "order,matrix,pm_file,pm_sha256,capture_dir,exit" > "$capture_dir/order.csv"
    order=0
    for input_file in "${input_files[@]}"; do
        [ "${input_file%.pm}" != "$input_file" ] || continue
        order=$((order + 1))
        matrix=$(basename "$input_file" .pm)
        run_dir=$capture_dir/$(printf '%02d' "$order")_$matrix
        mkdir -p "$run_dir"
        # Vernier writes into the working directory, and srun's tasks start in this one.
        (cd "$run_dir" && env $capture_env srun --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
            "$VERNIER_SPMV" "$input_file" -c MPI -k 0 -n 1 -W none > run.txt 2>&1 < /dev/null)
        status=$?
        echo "$order,$matrix,$input_file,$(sha256sum "$input_file" | cut -d' ' -f1),$run_dir,$status" \
            >> "$capture_dir/order.csv"
    done
    python3 "$REPO/scripts/vernier-config.py" "$capture_dir" "$out_dir" || exit 1
    mapfile -t generated_inputs < "$out_dir/generated_inputs"
    input_files+=("${generated_inputs[@]}")
fi

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

# The ABCG patterns: each program's own pilot on each .pm input, on STANDARD, with the
# same k, n, warmup and budget. Their calc lines have the same format.
for input_file in "${input_files[@]}"; do
    [ "${input_file%.pm}" != "$input_file" ] || continue
    for program in "${abcg_programs[@]}"; do
        read -r prefix bin mode <<< "$program"
        pattern=${prefix}_$(basename "$input_file" .pm)
        pilot_file=$out_dir/pilot_${pattern}_$(date +%Y%m%d-%H%M%S).txt
        srun --nodes="$nodes" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" "$bin" "$input_file" \
            -c STANDARD -P "$pattern" -W "$warmup" -k "$PILOT_K" -n "$PILOT_N" \
            --calc "$target_seconds" --persistent "$mode" $ABCG_EXTRA_ARGS \
            > "$pilot_file" 2>&1 < /dev/null
        grep 'calc:' "$pilot_file" | awk '{gsub(/[()]/, "", $1); gsub(/\)/, "", $15); print $1, $15, $10}' \
            >> "$out_dir/targets"
        expected=$((expected + 1))
    done
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
