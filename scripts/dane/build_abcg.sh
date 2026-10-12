#!/bin/bash
# Build ABCG's benchmark_cg_mpil (ABCG-new HEAD, NUMA-group edit) against the
# locality_aware already built for a walk step in ~/cand/steps/<label>/la-install.
# usage: build_abcg.sh <label> [compat|pmpi]
#   compat: m10/m11 API (~/cand/abcg-compat.patch; also PMPI-only)
#   pmpi:   newer API, only the Allreduce PMPI configs (~/cand/abcg-pmpionly.patch)
set -e
LABEL=$1; COMPAT=$2
GCC=/usr/tce/packages/gcc/gcc-13.3.1/bin
# PATH on dane lists mvapich2/Intel ahead of any module, so pin Open MPI 4.1.2.
OMPI=/usr/tce/packages/openmpi/openmpi-4.1.2-gcc-13.3.1/bin
MPI_ARGS="-DMPI_C_COMPILER=$OMPI/mpicc -DMPI_CXX_COMPILER=$OMPI/mpicxx"
SRC=/g/g20/bacon4/benchmark26/ABCG-new
# ABCG needs CMake >= 3.24; the default cmake on dane is 3.23.1.
CMAKE=/usr/tce/packages/cmake/cmake-3.30.5/bin/cmake
LA=$HOME/cand/steps/$LABEL/la-install
ROOT=$HOME/cand/abcg/$LABEL
rm -rf $ROOT; mkdir -p $ROOT/src
git -C $SRC log -1 --format='abcg %H' > $ROOT/info.txt
cat $HOME/cand/steps/$LABEL/commit.txt >> $ROOT/info.txt
git -C $SRC archive HEAD | tar -x -C $ROOT/src
patch -s -p1 -d $ROOT/src < $HOME/cand/abcg-numa.patch
patch -s -p1 -d $ROOT/src < $HOME/cand/abcg-mpi31.patch
patch -s -p1 -d $ROOT/src < $HOME/cand/abcg-virtual.patch
if [ "$COMPAT" = compat ]; then
    patch -s -p1 -d $ROOT/src < $HOME/cand/abcg-compat.patch
    echo "abcg_patch=compat" >> $ROOT/info.txt
elif [ "$COMPAT" = pmpi ]; then
    patch -s -p1 -d $ROOT/src < $HOME/cand/abcg-pmpionly.patch
    echo "abcg_patch=pmpi" >> $ROOT/info.txt
fi
$CMAKE -S $ROOT/src -B $ROOT/build -DUSE_LOCALITY=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$GCC/g++ $MPI_ARGS -DCMAKE_CXX_FLAGS="-march=sapphirerapids" \
  -Dlocality_aware_DIR=$LA/share/locality_aware > $ROOT/cmake.log 2>&1
$CMAKE --build $ROOT/build -j16 > $ROOT/build.log 2>&1
BIN=$ROOT/build/benchmarks/benchmark_cg_mpil
DIR=$ROOT/build/benchmarks/CMakeFiles/benchmark_cg_mpil.dir
grep -q "$LA" $DIR/link.txt || { echo "[$LABEL] not linked to step library"; exit 1; }
grep -q "spack/opt/spack/.*/localityaware-" $DIR/link.txt $DIR/flags.make && { echo "[$LABEL] pulls in the Spack localityaware install"; exit 1; }
grep -q "spack-env/view" $DIR/flags.make && { echo "[$LABEL] includes the Spack env view"; exit 1; }
grep -q "$LA/include" $DIR/flags.make || { echo "[$LABEL] not using step locality_aware headers"; exit 1; }
ldd $BIN | grep -q "openmpi-4.1.2-gcc-13.3.1/lib/libmpi.so" || { echo "[$LABEL] does not link Open MPI 4.1.2"; exit 1; }
ldd $BIN | grep -q mvapich && { echo "[$LABEL] links mvapich2"; exit 1; }
ldd $BIN | grep locality_aware | grep -vq "$LA" && { echo "[$LABEL] loads a locality_aware library outside the step install"; exit 1; }
echo "[$LABEL] ABCG built${COMPAT:+ ($COMPAT)}: $BIN"
