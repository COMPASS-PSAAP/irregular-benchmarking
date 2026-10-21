#!/bin/bash
#SBATCH -J transport4
#SBATCH -p pbatch
#SBATCH -t 1:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=112
#SBATCH --exclude=dane261,dane764
#SBATCH -o transport4-%j.out
# Does the 0.160 s stall depend on Open MPI's transport? (todo item 1; 10-19-2026.txt sec. 8.)
# The 10-19 stall test (8064004-07) put the stall in ~12% of crankseg_2 MPI exchange
# iterations at 4 nodes. Here the same run (replay, crankseg_2.pm, -c MPI, k 25, n 2000,
# per-iteration raw files) is repeated under each transport this Open MPI 4.1.2 offers,
# selected with OMPI_MCA_* in the environment. pml/mtl selection is logged at verbose 10.
#   default    nothing forced (what every campaign used; the log says which pml)
#   cm-psm2    pml cm, mtl psm2 (Omni-Path PSM2)
#   cm-ofi     pml cm, mtl ofi (libfabric; provider logged by FI_LOG_LEVEL=info)
#   ob1-openib pml ob1, btl self,vader,openib (verbs over hfi1; vader on-node)
#   ucx        pml ucx (may refuse to start without a supported device: a nonzero exit is
#              a result, not a failure of the test)
# Then ABCG's MPI-only SpMV exchange (Isend/Irecv, --persistent off, the kind that stalled
# most, 11%), same matrix, k and n, built from ABCG 7fde517 against two MPIs:
#   abcg-ompi     build-mpi, Open MPI 4.1.2, default transport
#   abcg-mvapich2 build-mvapich2, MVAPICH2 2.3.7 (ch3:psm, i.e. PSM over Omni-Path)
# A stall under MVAPICH2 too puts the cause below MPI (PSM2, hfi1 driver, fabric).
# Order: the seven, then the seven reversed, so time-of-run is not confounded with config.
# Read: stall-scan.csv (stalls per run, in apply vs late), selected pml per run in
# summary.txt; a config with no ~160 ms iterations is the lead.
DIR=$SLURM_SUBMIT_DIR/$SLURM_JOB_ID-transport4
M=/usr/WS2/unm_cs/NicholasBaconInputsData
BIN=$HOME/cand/steps/old_2b1acb4/bench-build/src/irregular-benchmarking
A=$HOME/benchmark26/ABCG-7fde517
MV=/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-13.3.1
SCRIPTS=$HOME/benchmark26/irregular-benchmarking-new/scripts
SCAN=/usr/workspace/unm_cs/NicksBenchmarkDumps/data-10-19-2026-stall4/stall-scan.py
GEOM=(--nodes=4 --ntasks-per-node=112)
mkdir -p "$DIR"

PROV_SPACK_ENV=/g/g20/bacon4/spackenvs/spack-k5 "$SCRIPTS/provenance.sh" "$DIR" "$BIN" \
    "$M/crankseg_2.pm" -- "${GEOM[@]}"

for b in "$A/build-mpi/benchmarks/benchmark_spmv" "$A/build-mvapich2/benchmarks/benchmark_spmv"; do
    { echo "== $b"; strings "$b" | grep '^PROVENANCE'; LD_LIBRARY_PATH=$MV/lib:$LD_LIBRARY_PATH ldd "$b"; } \
        >> "$DIR/abcg-binaries.txt"
done

# Sets env (variables for the run) and cmd (the program and its flags) for config $1;
# output and per-iteration files go to $2 (the run file without .txt).
config_run() {
    verbose=(OMPI_MCA_pml_base_verbose=10 OMPI_MCA_mtl_base_verbose=10)
    replay=("$BIN" -r -c MPI -x EXPORT -q -b -W nearest -I 1 -P crankseg_2 -k 25 -n 2000
            -s NUMA -U --iter-prefix "$2" --iter-raw -f "$M/crankseg_2.pm")
    abcg=("$M/crankseg_2.pm" -c MPI -P abcgspmvnp_crankseg_2 -W nearest -k 25 -n 2000
          --persistent off --iter-prefix "$2" --iter-raw)
    case $1 in
        default)    env=("${verbose[@]}"); cmd=("${replay[@]}") ;;
        cm-psm2)    env=("${verbose[@]}" OMPI_MCA_pml=cm OMPI_MCA_mtl=psm2); cmd=("${replay[@]}") ;;
        cm-ofi)     env=("${verbose[@]}" OMPI_MCA_pml=cm OMPI_MCA_mtl=ofi FI_LOG_LEVEL=info)
                    cmd=("${replay[@]}") ;;
        ob1-openib) env=("${verbose[@]}" OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,openib)
                    cmd=("${replay[@]}") ;;
        ucx)        env=("${verbose[@]}" OMPI_MCA_pml=ucx); cmd=("${replay[@]}") ;;
        abcg-ompi)  env=("${verbose[@]}"); cmd=("$A/build-mpi/benchmarks/benchmark_spmv" "${abcg[@]}") ;;
        # Slurm binds every rank to one CPU for the Open MPI runs; MV2_ENABLE_AFFINITY=0 keeps
        # MVAPICH2 from replacing that binding with its own. MV2_SHOW_ENV_INFO=2 logs its settings.
        abcg-mvapich2)
                    env=(LD_LIBRARY_PATH=$MV/lib:$LD_LIBRARY_PATH MV2_ENABLE_AFFINITY=0 MV2_SHOW_ENV_INFO=2)
                    cmd=("$A/build-mvapich2/benchmarks/benchmark_spmv" "${abcg[@]}") ;;
    esac
}

echo "run,config,start,end,exit,file" > "$DIR/manifest.csv"
run=0
for cfg in default cm-psm2 cm-ofi ob1-openib ucx abcg-ompi abcg-mvapich2 \
           abcg-mvapich2 abcg-ompi ucx ob1-openib cm-ofi cm-psm2 default; do
    run=$((run + 1))
    f=$DIR/run_$(printf %04d "$run")_crankseg_2_MPI_$cfg.txt
    config_run "$cfg" "${f%.txt}"
    start=$(date -Iseconds)
    env "${env[@]}" srun "${GEOM[@]}" "${cmd[@]}" > "$f" 2>&1 < /dev/null
    status=$?
    echo "$run,$cfg,$start,$(date -Iseconds),$status,$(basename "$f")" >> "$DIR/manifest.csv"
done

{
    # The exact wording of the verbose selection lines was not checked in advance, so this
    # keeps every line mentioning a selection or provider, with hosts/ranks/numbers folded.
    echo "== selected pml / mtl per run (distinct lines from the verbose log)"
    for f in "$DIR"/run_*.txt; do
        echo "$(basename "$f"):"
        grep -h -i -E "selected|provider" "$f" | sed 's/^\[[^]]*\] *//; s/[0-9]\{3,\}/N/g' \
            | sort | uniq -c | sort -rn | head -8
    done
    echo "== timedTotal per run (ms; min max avg over ranks)"
    grep -H "timedTotal" "$DIR"/run_*.txt | sed "s#$DIR/##"
    echo "== stall-scan.py (iterations over 10x median and 20 ms)"
    python3 "$SCAN" "$DIR" > "$DIR/stall-scan.csv"
    awk -F, 'NR > 1 {n[$1]++} END {for (r in n) print "run " r ": " n[r] " flagged"}' "$DIR/stall-scan.csv" | sort
} > "$DIR/summary.txt" 2>&1
