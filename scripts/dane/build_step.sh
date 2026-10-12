#!/bin/bash
# Build the benchmark against one locality_aware commit, for the old->new
# performance walk. Sources come from git archive, so no checkout is modified.
# usage: build_step.sh <label> <locality_aware commit> [locality_aware patch|-] [cabana header patch]
set -e
LABEL=$1; COMMIT=$2; LA_PATCH=$3; CAB_PATCH=$4
[ "$LA_PATCH" = "-" ] && LA_PATCH=
GCC=/usr/tce/packages/gcc/gcc-13.3.1/bin
# PATH on dane lists mvapich2/Intel ahead of any module, so pin the same
# Open MPI build1 uses instead of letting FindMPI pick from PATH.
OMPI=/usr/tce/packages/openmpi/openmpi-4.1.2-gcc-13.3.1/bin
MPI_ARGS="-DMPI_C_COMPILER=$OMPI/mpicc -DMPI_CXX_COMPILER=$OMPI/mpicxx"
FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids"
ROOT=$HOME/cand/steps/$LABEL
ENV=$HOME/spackenvs/spack-k5
BENCH=/g/g20/bacon4/benchmark26/irregular-benchmarking-new
KOKKOS_DIR=$(grep '^Kokkos_DIR:' $BENCH/build1/CMakeCache.txt | cut -d= -f2)
CAB_PREFIX=$(spack -e $ENV location -i cabana)
# Not the env view: it also holds the Spack build's locality_aware.h.
TCLAP_INC=$(spack -e $ENV location -i tclap)/include
JSON_INC=$(spack -e $ENV location -i nlohmann-json)/include
rm -rf $ROOT; mkdir -p $ROOT/la-src $ROOT/bench-src
# LA_EXTRA_CXXFLAGS (env): build-only fixes, e.g. "-include cstring" for steps
# that call memcpy without the header; recorded in commit.txt.
# LA_UNIT_TESTS=ON (env): for commits whose CMake needs tests on (e.g. 54b7aaf).
git -C $ENV/localityaware log -1 --format='%H %ad %s' --date=short $COMMIT > $ROOT/commit.txt
git -C $ENV/localityaware archive $COMMIT | tar -x -C $ROOT/la-src
[ -n "$LA_EXTRA_CXXFLAGS" ] && echo "extra_cxxflags=$LA_EXTRA_CXXFLAGS" >> $ROOT/commit.txt
if [ -n "$LA_PATCH" ]; then
    patch -s -p1 -d $ROOT/la-src < $LA_PATCH
    echo "patch=$LA_PATCH" >> $ROOT/commit.txt
fi
cmake -S $ROOT/la-src -B $ROOT/la-build -DNUMA_H=ON -DENABLE_UNIT_TESTS=${LA_UNIT_TESTS:-OFF} -DBENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$GCC/gcc -DCMAKE_CXX_COMPILER=$GCC/g++ $MPI_ARGS \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS $LA_EXTRA_CXXFLAGS" \
  -DCMAKE_INSTALL_PREFIX=$ROOT/la-install > $ROOT/la-cmake.log 2>&1
cmake --build $ROOT/la-build -j16 > $ROOT/la-build.log 2>&1
cmake --install $ROOT/la-build > $ROOT/la-install.log 2>&1
LA_CFG=$(dirname "$(find $ROOT/la-install -name 'locality_aware*onfig.cmake' | head -1)")
LA_HDR=$(find $ROOT/la-install -name locality_aware.h | head -1)
echo "[$LABEL] locality_aware $COMMIT built"
# The Spack-installed Cabana still carries an old local edit to this header,
# so always overlay the pristine copy from the dev checkout's HEAD.
cp -r $CAB_PREFIX $ROOT/cab-install
CAB_HDR=$(find $ROOT/cab-install/include -name Cabana_CommunicationPlan_LocalityAware.hpp | head -1)
git -C $ENV/cabana show HEAD:core/src/impl/Cabana_CommunicationPlan_LocalityAware.hpp > $CAB_HDR
if [ -n "$CAB_PATCH" ]; then
    patch -s $CAB_HDR < $CAB_PATCH
    echo "cabana_patch=$CAB_PATCH" >> $ROOT/commit.txt
fi
git -C $BENCH archive HEAD | tar -x -C $ROOT/bench-src
if grep -q MPIL_Set_split $LA_HDR; then
    echo "split=yes" >> $ROOT/commit.txt
else
    patch -s -p1 -d $ROOT/bench-src < $HOME/cand/cli-nosplit.patch
    echo "split=no" >> $ROOT/commit.txt
fi
cmake -S $ROOT/bench-src -B $ROOT/bench-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$GCC/gcc -DCMAKE_CXX_COMPILER=$GCC/g++ $MPI_ARGS \
  -DCMAKE_CXX_FLAGS="-march=sapphirerapids" \
  -DCabana_DIR=$ROOT/cab-install/share/cmake/Cabana -DKokkos_DIR=$KOKKOS_DIR \
  -Dlocality_aware_DIR=$LA_CFG -DMPI_Advance_PREFIX=$ROOT/la-install \
  -DTCLAP_INCLUDE_DIR=$TCLAP_INC -DNLOHMANN_JSON_INCLUDE_DIR=$JSON_INC \
  > $ROOT/bench-cmake.log 2>&1
cmake --build $ROOT/bench-build -j16 > $ROOT/bench-build.log 2>&1
grep -q "$ROOT/la-install" $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/link.txt || { echo "[$LABEL] benchmark not linked to step library"; exit 1; }
grep -q "spack/opt/spack/.*/localityaware-" $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/link.txt $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/flags.make && { echo "[$LABEL] benchmark pulls in the Spack localityaware install"; exit 1; }
ldd $ROOT/bench-build/src/irregular-benchmarking | grep locality_aware | grep -vq "$ROOT/la-install" && { echo "[$LABEL] benchmark loads a locality_aware library outside the step install"; exit 1; }
grep -q "$ROOT/la-install/include" $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/flags.make || { echo "[$LABEL] benchmark not using step locality_aware headers"; exit 1; }
grep -q "spack-env/view" $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/flags.make && { echo "[$LABEL] benchmark includes the Spack env view"; exit 1; }
grep -q "$ROOT/cab-install" $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/flags.make || { echo "[$LABEL] benchmark not using step Cabana headers"; exit 1; }
ldd $ROOT/bench-build/src/irregular-benchmarking | grep -q "openmpi-4.1.2-gcc-13.3.1/lib/libmpi.so" || { echo "[$LABEL] benchmark does not link Open MPI 4.1.2"; exit 1; }
ldd $ROOT/bench-build/src/irregular-benchmarking | grep -q mvapich && { echo "[$LABEL] benchmark links mvapich2"; exit 1; }
echo "[$LABEL] benchmark built ($(tail -1 $ROOT/commit.txt)): $ROOT/bench-build/src/irregular-benchmarking"
