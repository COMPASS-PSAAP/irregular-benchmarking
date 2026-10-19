#!/bin/bash
# Builds ABCG's pilots branch three times for the campaigns, with the compiler, Open MPI,
# and flags irregular-benchmarking uses on dane: build-mpi without MPI Advance (the MPI
# arm's binaries), build-old against locality_aware 752a19b with the old API (STANDARD and
# LOCALITY), the same library build as ~/cand/steps/old_efb6722, and build-vernier, whose
# benchmark_spmv is the MPI arm with a Vernier pattern around each exchange (pilot.sbatch's
# VERNIER_SPMV), against the Vernier that ~/cand/build_vernier.sh installs.
# ABCG_COMMIT (env): build that commit, archived from the clone ABCG_GIT (default
# ~/benchmark26/ABCG-git), into a new ~/benchmark26/ABCG-<short commit>; the commit is
# compiled into every binary (ABCG's CMakeLists.txt, SOURCE_COMMIT). Without it, ABCG_SRC
# (default ~/benchmark26/ABCG-pilots) is built in place and CMake records what git says
# about it ("unknown (not a git checkout)" for a copied tree).
set -e
ABCG_GIT=${ABCG_GIT:-$HOME/benchmark26/ABCG-git}
if [ -n "${ABCG_COMMIT:-}" ]; then
    full=$(git -C "$ABCG_GIT" rev-parse "$ABCG_COMMIT^{commit}")
    SRC=$HOME/benchmark26/ABCG-${full:0:7}
    [ -e "$SRC" ] && { echo "$SRC already exists; not overwriting it"; exit 1; }
    mkdir -p "$SRC"
    git -C "$ABCG_GIT" archive "$full" | tar -x -C "$SRC"
    source_args=(-DSOURCE_COMMIT=$full)
else
    SRC=${ABCG_SRC:-$HOME/benchmark26/ABCG-pilots}
    source_args=()
fi
GCC=/usr/tce/packages/gcc/gcc-13.3.1/bin
OMPI=/usr/tce/packages/openmpi/openmpi-4.1.2-gcc-13.3.1/bin
LA=$HOME/cand/steps/old_efb6722/la-install
# build_step_bench.sh's commit.txt starts with the library's full commit.
LA_COMMIT=$(head -1 "$(dirname $LA)/commit.txt" | cut -d' ' -f1)
VERNIER=$HOME/cand/vernier/install
# ABCG needs CMake >= 3.24; dane's default cmake is 3.23.1. 3.30.5 is the spack-k5 env's.
CMAKE=/usr/tce/packages/cmake/cmake-3.30.5/bin/cmake
common=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$GCC/g++ -DMPI_CXX_COMPILER=$OMPI/mpicxx
        -DCMAKE_CXX_FLAGS=-march=sapphirerapids "${source_args[@]}")

$CMAKE -S $SRC -B $SRC/build-mpi "${common[@]}" > $SRC/build-mpi.log 2>&1
$CMAKE --build $SRC/build-mpi -j16 >> $SRC/build-mpi.log 2>&1
$CMAKE -S $SRC -B $SRC/build-old "${common[@]}" -DUSE_LOCALITY=ON -DMPIL_OLD_API=ON \
    -DCMAKE_PREFIX_PATH=$LA "-DLOCALITY_AWARE_COMMIT=$LA_COMMIT" > $SRC/build-old.log 2>&1
$CMAKE --build $SRC/build-old -j16 >> $SRC/build-old.log 2>&1
$CMAKE -S $SRC -B $SRC/build-vernier "${common[@]}" -DABCG_VERNIER=ON \
    -Dvernier_DIR=$VERNIER/share/vernier > $SRC/build-vernier.log 2>&1
$CMAKE --build $SRC/build-vernier -j16 --target benchmark_spmv >> $SRC/build-vernier.log 2>&1

for b in build-mpi build-old; do
    for p in benchmark_spmv benchmark_cg_pilot; do
        exe=$SRC/$b/benchmarks/$p
        ldd $exe | grep -q "openmpi-4.1.2-gcc-13.3.1/lib/libmpi.so" || { echo "$b/$p: not Open MPI 4.1.2"; exit 1; }
        echo "$b/$p built"
    done
done
grep -q "$LA" $SRC/build-old/benchmarks/CMakeFiles/benchmark_spmv.dir/link.txt || { echo "build-old does not link $LA"; exit 1; }
grep -rq "MPIL_OLD_API" $SRC/build-old/CMakeFiles/abcg.dir/flags.make || { echo "build-old lacks MPIL_OLD_API"; exit 1; }
if grep -rq "locality_aware" $SRC/build-mpi/benchmarks/CMakeFiles/benchmark_spmv.dir/link.txt; then
    echo "build-mpi links locality_aware"; exit 1
fi
exe=$SRC/build-vernier/benchmarks/benchmark_spmv
ldd $exe | grep -q "openmpi-4.1.2-gcc-13.3.1/lib/libmpi.so" || { echo "build-vernier/benchmark_spmv: not Open MPI 4.1.2"; exit 1; }
ldd $exe | grep -q "$VERNIER/lib/libvernier.so" || { echo "build-vernier does not load $VERNIER's libvernier"; exit 1; }
grep -q "ABCG_VERNIER" $SRC/build-vernier/benchmarks/CMakeFiles/benchmark_spmv.dir/flags.make || { echo "build-vernier lacks ABCG_VERNIER"; exit 1; }
echo "build-vernier/benchmark_spmv built"
echo "build-old links $LA with MPIL_OLD_API; build-mpi has no MPI Advance; build-vernier captures with $(cut -c1-7 $VERNIER/vernier-commit)"
for exe in $SRC/build-mpi/benchmarks/benchmark_{spmv,cg_pilot} $SRC/build-old/benchmarks/benchmark_{spmv,cg_pilot} \
    $SRC/build-vernier/benchmarks/benchmark_spmv; do
    echo "${exe#$SRC/}: $(strings "$exe" | grep -m1 '^PROVENANCE: ')"
done
