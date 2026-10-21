#!/bin/bash
# Record a launch's provenance. Run it inside the sbatch job, before the benchmark:
#
#   scripts/provenance.sh OUTDIR BINARY INPUTS [-- SRUN_ARGS...]
#
# INPUTS is one input (JSON pattern config, .pm matrix, or Vernier capture
# directory) or several, comma-separated.
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
    echo "usage: $0 OUTDIR BINARY INPUTS [-- SRUN_ARGS...]" >&2
    exit 1
fi
outdir=$1
bin=$(readlink -f "$2")
inputs=$3  # one input file, or several comma-separated
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

# True only when $1 is the top of its own git checkout. A plain directory inside
# some other repository (e.g. a git-archive export under a home directory that is
# itself a repo) must not be reported as that repository.
is_checkout() {
    [ -d "$1" ] && [ "$(git -C "$1" rev-parse --show-toplevel 2>/dev/null)" = "$(readlink -f "$1")" ]
}

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

    section "nodes"
    # Read on every node inside the job, since compute nodes can differ from the login
    # node and from each other: clock source, CPU frequency setting, network devices.
    # One "host<TAB>field<TAB>value" line each; the script stays next to its output.
    cat > "$base.nodes.sh" <<'NODE_EOF'
h=$(hostname -s)
p() { printf '%s\t%s\t%s\n' "$h" "$1" "$2"; }
cs=/sys/devices/system/clocksource/clocksource0
p clocksource "$(cat $cs/current_clocksource 2>/dev/null || echo missing)"
p clocksources_available "$(cat $cs/available_clocksource 2>/dev/null || echo missing)"
cf=/sys/devices/system/cpu/cpu0/cpufreq
p cpufreq_driver "$(cat $cf/scaling_driver 2>/dev/null || echo missing)"
p cpufreq_governor "$(cat $cf/scaling_governor 2>/dev/null || echo missing)"
p cpufreq_khz "min $(cat $cf/scaling_min_freq 2>/dev/null || echo ?) max $(cat $cf/scaling_max_freq 2>/dev/null || echo ?) cpuinfo_max $(cat $cf/cpuinfo_max_freq 2>/dev/null || echo ?)"
p no_turbo "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo missing)"
p cpu_model "$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p')"
# A snapshot only: the current frequency of every CPU while nothing else of ours runs.
p cur_freq_khz_min_median_max "$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | sort -n | awk '{a[NR]=$1} END {if (NR) print a[1], a[int((NR+1)/2)], a[NR]; else print "missing"}')"
devs=
for d in /sys/class/infiniband/*; do
    [ -e "$d" ] || continue
    ports=
    for pt in "$d"/ports/*; do
        ports="$ports port $(basename "$pt"): $(cat "$pt/state") $(cat "$pt/phys_state") $(cat "$pt/rate") $(cat "$pt/link_layer");"
    done
    p ib_device "$(basename "$d") fw $(cat "$d/fw_ver" 2>/dev/null) ($(cat "$d/node_desc" 2>/dev/null)):$ports"
    devs="$devs $(basename "$d")"
done
p ib_devices "${devs:- none}"
p ucx_transports "$(ucx_info -d 2>/dev/null | awk '/Transport:/ {t = $3} /Device:/ {print t "/" $3}' | sort -u | tr '\n' ' ' || true)"
p fi_providers "$(fi_info -l 2>/dev/null | grep ':$' | tr -d ':' | tr '\n' ' ' || true)"
lspci 2>/dev/null | grep -Ei 'infiniband|omni-path|fabric|network controller|ethernet' | while read -r line; do p pci "$line"; done
NODE_EOF
    srun --ntasks-per-node=1 --nodes="${SLURM_JOB_NUM_NODES:-1}" bash "$base.nodes.sh" 2>&1 | sort > "$base.nodes.txt"
    echo "per-node facts (host field value): $base.nodes.txt, from $base.nodes.sh"
    # Each field's distinct values with how many nodes had them, so an odd node stands out.
    for field in clocksource cpufreq_driver cpufreq_governor cpufreq_khz no_turbo cpu_model ib_devices ib_device \
        ucx_transports fi_providers pci; do
        echo "$field:"
        awk -F'\t' -v f="$field" '$2 == f && !seen[$1 FS $3]++ {n[$3]++} END {for (v in n) printf "  %s  [%d nodes]\n", v, n[v]}' \
            "$base.nodes.txt" | sort
    done

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

    section "binary provenance"
    # The commits CMake compiled into the binary (src/CMakeLists.txt); binaries built
    # before 2026-10-15 have none.
    echo "$(strings "$bin" | grep -m1 '^PROVENANCE: ' || echo 'PROVENANCE: none in this binary')"
    echo "linked libraries (ldd):"
    ldd "$bin" 2>&1 | sed 's/^[[:space:]]*/  /'

    section "benchmark source"
    if is_checkout "$src"; then
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
            # What this Open MPI could use; which pml/mtl/btl a run picks is not recorded here.
            for kind in pml mtl btl; do
                echo "available $kind components: $("$ompi_info" | awk -v k="MCA $kind:" 'index($0, k) {print $3}' | tr '\n' ' ')"
            done
        fi
    fi

    section "spack"
    if [ -n "${PROV_SPACK_ENV:-}" ]; then
        spack_bin=$(command -v spack || echo "$HOME/spack/bin/spack")
        echo "environment: $PROV_SPACK_ENV"
        "$spack_bin" -e "$PROV_SPACK_ENV" find -lp --no-groups localityaware cabana kokkos 2>&1
        # The environment's installs describe the binary only if it links them; the dane
        # build scripts link their own locality_aware builds instead.
        la_lib=$(ldd "$bin" | awk '/locality_aware|mpi_advance/{print $3; exit}')
        case $la_lib in
            "") echo "binary's locality_aware: not a shared library of the binary; see PROVENANCE above" ;;
            */spack/opt/spack/*) echo "binary's locality_aware: $la_lib (a Spack install)" ;;
            *) echo "binary's locality_aware: $la_lib (NOT a Spack install: the localityaware lines above do not describe this binary; see PROVENANCE above)" ;;
        esac
        # The installed build is what the binary contains; a dev checkout edited after
        # install (see its dirty state) does not describe it.
        for pkg in localityaware cabana; do
            dev=$PROV_SPACK_ENV/$pkg
            if is_checkout "$dev"; then
                echo "$pkg dev checkout: $dev $(git -C "$dev" rev-parse HEAD) ($(git -C "$dev" rev-parse --abbrev-ref HEAD), $(git -C "$dev" status --porcelain | wc -l) changed files)"
            fi
        done
    else
        echo "PROV_SPACK_ENV not set; skipped"
    fi

    section "inputs"
    IFS=, read -ra input_files <<< "$inputs"
    for f in "${input_files[@]}"; do
        f=$(readlink -f "$f")
        echo "input: $f"
        if [ -d "$f" ]; then
            # A capture directory: its per-rank files in name order, hashed together.
            echo "  sha256 (hierarchical files): $(cat "$f"/vernier-*-0.hist-hierarchical.json | sha256sum | cut -d' ' -f1)"
            echo "  capture settings:"
            sed 's/^/    /' "$f/../vernier_settings"
        else
            echo "  sha256: $(sha256sum "$f" | cut -d' ' -f1)"
        fi
        echo "  mtime: $(date -Iseconds -r "$f")"
        echo "  patterns: $(python3 "$script_dir/pattern-names.py" "$f" | tr '\n' ' ')"
    done

    section "environment"
    env | grep -E '^(SLURM_|OMPI_|PMIX_|PMI_|PSM2_|HFI_|FI_|UCX_|OMP_|KOKKOS_|MPIL_)|^(LD_LIBRARY_PATH|LD_PRELOAD|PATH)=' | sort
} > "$out" 2>&1

echo "provenance: $out"
