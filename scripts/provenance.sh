#!/bin/bash
# Record a launch's provenance. Run it inside the sbatch job, before the benchmark:
#
#   scripts/provenance.sh OUTDIR BINARY CONFIG [-- SRUN_ARGS...]
#
# SRUN_ARGS must be the geometry the benchmark's srun uses (e.g. --nodes=2
# --ntasks-per-node=112), because binding is read from a separate srun step and
# only matches the benchmark's if both steps are launched the same way.
# Set PROV_SPACK_ENV to the Spack environment directory to record the installed
# localityaware/cabana/kokkos and the state of their dev checkouts.
#
# Writes OUTDIR/provenance_<jobid>_<timestamp>.txt, plus .binding.txt,
# .fabric.txt and .benchmark.diff next to it. The seed, MPI library, compiler,
# and per-sample on/off-node and NUMA neighbor split are printed by the
# benchmark itself under -r.
set -u

if [ $# -lt 3 ]; then
    echo "usage: $0 OUTDIR BINARY CONFIG [-- SRUN_ARGS...]" >&2
    exit 1
fi
outdir=$1
bin=$(readlink -f "$2")
config=$(readlink -f "$3")
shift 3
if [ "${1:-}" = "--" ]; then
    shift
fi
srun_args=("$@")

script_dir=$(dirname "$(readlink -f "$0")")
base=$outdir/provenance_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d-%H%M%S)
out=$base.txt
mkdir -p "$outdir"

section() { printf '\n== %s\n' "$1"; }

{
    section "job"
    echo "job id: ${SLURM_JOB_ID:-unset}"
    echo "job name: ${SLURM_JOB_NAME:-unset}"
    echo "partition: ${SLURM_JOB_PARTITION:-unset}"
    echo "recorded at: $(date -Iseconds)"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "job start: $(scontrol show job "$SLURM_JOB_ID" | grep -o 'StartTime=[^ ]*' | cut -d= -f2)"
    fi
    echo "node count: ${SLURM_JOB_NUM_NODES:-unset}"
    echo "node list: ${SLURM_JOB_NODELIST:-unset}"
    if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
    fi
    echo "tasks: ${SLURM_NTASKS:-unset}"
    echo "tasks per node: ${SLURM_TASKS_PER_NODE:-unset}"
    echo "cpus per task: ${SLURM_CPUS_PER_TASK:-unset}"
    echo "srun args: ${srun_args[*]:-(none)}"

    section "binding"
    # One line per rank: rank, host, local id, allowed CPUs, allowed memory nodes.
    srun "${srun_args[@]}" bash -c 'echo "$SLURM_PROCID $(hostname -s) $SLURM_LOCALID $(grep Cpus_allowed_list /proc/self/status | cut -f2) $(grep Mems_allowed_list /proc/self/status | cut -f2)"' \
        2>&1 | sort -n > "$base.binding.txt"
    echo "per-rank binding (rank host localid cpus mems): $base.binding.txt ($(wc -l < "$base.binding.txt") lines)"
    echo "distinct CPU sets: $(awk '{print $4}' "$base.binding.txt" | sort -u | wc -l)"
    echo "SLURM_CPU_BIND: ${SLURM_CPU_BIND:-unset}"

    section "switch / group"
    # Slurm on dane uses topology/flat, so the fabric is the only source; record what it says.
    echo "slurm topology plugin: $(scontrol show config | awk -F'= ' '/^TopologyPlugin/{print $2}')"
    {
        for tool in opainfo iblinkinfo; do
            echo "-- $tool"
            srun --ntasks-per-node=1 --nodes="${SLURM_JOB_NUM_NODES:-1}" bash -c \
                "echo \"host \$(hostname -s)\"; $tool 2>&1; echo \"exit \$?\""
        done
    } > "$base.fabric.txt" 2>&1
    echo "raw fabric queries: $base.fabric.txt"

    section "benchmark build"
    echo "binary: $bin"
    echo "binary sha256: $(sha256sum "$bin" | cut -d' ' -f1)"
    echo "binary mtime: $(date -Iseconds -r "$bin")"
    cache=""
    dir=$(dirname "$bin")
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/CMakeCache.txt" ]; then
            cache=$dir/CMakeCache.txt
            break
        fi
        dir=$(dirname "$dir")
    done
    if [ -n "$cache" ]; then
        cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$cache")
        echo "cmake cache: $cache"
        echo "compiler: $cxx"
        echo "compiler version: $("$cxx" --version | head -1)"
        echo "build type: $(sed -n 's/^CMAKE_BUILD_TYPE:[A-Z]*=//p' "$cache")"
        echo "cxx flags: $(sed -n 's/^CMAKE_CXX_FLAGS:[A-Z]*=//p' "$cache")"
        src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:[A-Z]*=//p' "$cache")
    else
        echo "cmake cache: not found above $bin"
        src=""
    fi

    section "benchmark source"
    if [ -n "$src" ] && git -C "$src" rev-parse --git-dir > /dev/null 2>&1; then
        echo "source: $src"
        echo "commit: $(git -C "$src" rev-parse HEAD)"
        echo "branch: $(git -C "$src" rev-parse --abbrev-ref HEAD)"
        git -C "$src" diff HEAD > "$base.benchmark.diff"
        echo "uncommitted changes (tracked files): $(git -C "$src" diff HEAD --stat | tail -1)"
        echo "uncommitted diff: $base.benchmark.diff"
        echo "note: the binary may predate these changes; compare its mtime above"
    else
        echo "source: not a git checkout (${src:-unknown})"
    fi

    section "MPI"
    libmpi=$(ldd "$bin" | awk '/libmpi\.so/{print $3; exit}')
    echo "libmpi: ${libmpi:-not linked}"
    if [ -n "$libmpi" ]; then
        ompi_info=$(dirname "$(dirname "$libmpi")")/bin/ompi_info
        if [ -x "$ompi_info" ]; then
            echo "ompi_info: $("$ompi_info" --version | head -1)"
        fi
    fi

    section "spack"
    if [ -n "${PROV_SPACK_ENV:-}" ]; then
        spack_bin=$(command -v spack || echo "$HOME/spack/bin/spack")
        echo "environment: $PROV_SPACK_ENV"
        "$spack_bin" -e "$PROV_SPACK_ENV" find -lp --no-groups localityaware cabana kokkos 2>&1
        # The installed build is what the binary contains; a dev checkout edited after
        # install (see its dirty state) does not describe it.
        for pkg in localityaware cabana; do
            dev=$PROV_SPACK_ENV/$pkg
            if git -C "$dev" rev-parse --git-dir > /dev/null 2>&1; then
                echo "$pkg dev checkout: $dev $(git -C "$dev" rev-parse HEAD) ($(git -C "$dev" rev-parse --abbrev-ref HEAD), $(git -C "$dev" status --porcelain | wc -l) changed files)"
            fi
        done
    else
        echo "PROV_SPACK_ENV not set; skipped"
    fi

    section "pattern config"
    echo "config: $config"
    echo "config sha256: $(sha256sum "$config" | cut -d' ' -f1)"
    echo "config mtime: $(date -Iseconds -r "$config")"
    echo "patterns: $(python3 "$script_dir/pattern-names.py" "$config" | tr '\n' ' ')"
    echo "vernier commit: not recorded in the config"

    section "environment"
    env | grep -E '^(SLURM_|OMPI_|PMIX_|PMI_|PSM2_|HFI_|FI_|UCX_|OMP_|KOKKOS_|MPIL_)|^(LD_LIBRARY_PATH|LD_PRELOAD|PATH)=' | sort
} > "$out" 2>&1

echo "provenance: $out"
