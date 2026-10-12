#!/bin/bash
# Builds Vernier for the ABCG SpMV captures (ABCG's ABCG_VERNIER build, pilot.sbatch's
# VERNIER_SPMV) with the compiler and Open MPI the campaigns use on dane. Sources come
# from git archive of ~/benchmark25/Vernier, so the checkout is not modified. The commit
# is written to $PREFIX/vernier-commit, where pilot.sbatch reads it into each capture's
# vernier_settings.
# VERNIER_COMMIT (env): commit to build, default the checkout's HEAD (09d650b, branch
# hist-plot-script; GitHub's two later commits there only add plot scripts).
set -e
SRC=$HOME/benchmark25/Vernier
ROOT=$HOME/cand/vernier
PREFIX=$ROOT/install
GCC=/usr/tce/packages/gcc/gcc-13.3.1/bin
# PATH on dane lists mvapich2/Intel ahead of any module, so pin Open MPI 4.1.2.
OMPI=/usr/tce/packages/openmpi/openmpi-4.1.2-gcc-13.3.1/bin
CMAKE=/usr/tce/packages/cmake/cmake-3.30.5/bin/cmake

COMMIT=$(git -C $SRC rev-parse "${VERNIER_COMMIT:-HEAD}")
rm -rf $ROOT
mkdir -p $ROOT/src
git -C $SRC archive $COMMIT | tar -x -C $ROOT/src
# Vernier needs C++20 (requires-clauses); gcc 13.3.1 has it.
$CMAKE -S $ROOT/src -B $ROOT/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$GCC/gcc -DCMAKE_CXX_COMPILER=$GCC/g++ \
    -DMPI_C_COMPILER=$OMPI/mpicc -DMPI_CXX_COMPILER=$OMPI/mpicxx \
    -DCMAKE_INSTALL_PREFIX=$PREFIX > $ROOT/cmake.log 2>&1
$CMAKE --build $ROOT/build -j16 > $ROOT/build.log 2>&1
$CMAKE --install $ROOT/build > $ROOT/install.log 2>&1
git -C $SRC log -1 --format="%H %ad %s" --date=short $COMMIT > $PREFIX/vernier-commit

lib=$PREFIX/lib/libvernier.so
ldd $lib | grep -q "openmpi-4.1.2-gcc-13.3.1/lib/libmpi.so" || { echo "libvernier does not link Open MPI 4.1.2"; exit 1; }
ldd $lib | grep -q mvapich && { echo "libvernier links mvapich2"; exit 1; }
echo "Vernier $(cut -c1-7 $PREFIX/vernier-commit) built: $lib"
