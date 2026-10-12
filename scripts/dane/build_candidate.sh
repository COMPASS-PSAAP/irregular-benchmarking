#!/bin/bash
# Build one candidate: locality_aware (dev checkout + optional patch) with
# Spack's flags, Cabana headers (install copy + optional patched headers
# overlaid; the locality-aware Cabana code is header-only), and the benchmark.
# usage: build_candidate.sh <label> [locality_patch|-] [cabana_patch|-]
set -e
LABEL=$1; LA_PATCH=${2:--}; CAB_PATCH=${3:--}
GCC=/usr/tce/packages/gcc/gcc-13.3.1/bin
# PATH on dane lists mvapich2/Intel ahead of any module, so pin the same
# Open MPI build1 uses instead of letting FindMPI pick from PATH.
OMPI=/usr/tce/packages/openmpi/openmpi-4.1.2-gcc-13.3.1/bin
MPI_ARGS="-DMPI_C_COMPILER=$OMPI/mpicc -DMPI_CXX_COMPILER=$OMPI/mpicxx"
FLAGS="-O3 -march=sapphirerapids -mtune=sapphirerapids"
ROOT=$HOME/cand/$LABEL
ENV=$HOME/spackenvs/spack-k5
BENCH=/g/g20/bacon4/benchmark26/irregular-benchmarking-new
VIEW=$ENV/.spack-env/view
KOKKOS_DIR=$(grep '^Kokkos_DIR:' $BENCH/build1/CMakeCache.txt | cut -d= -f2)
CAB_PREFIX=$(spack -e $ENV location -i cabana)

rm -rf $ROOT; mkdir -p $ROOT
rsync -a --exclude 'build-linux-*' --exclude '.spack*' $ENV/localityaware/ $ROOT/la-src/
if [ "$LA_PATCH" != "-" ]; then git -C $ROOT/la-src apply --whitespace=nowarn "$LA_PATCH"; fi
cmake -S $ROOT/la-src -B $ROOT/la-build -DNUMA_H=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$GCC/gcc -DCMAKE_CXX_COMPILER=$GCC/g++ $MPI_ARGS \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" \
  -DCMAKE_INSTALL_PREFIX=$ROOT/la-install > $ROOT/la-cmake.log 2>&1
cmake --build $ROOT/la-build -j16 > $ROOT/la-build.log 2>&1
cmake --install $ROOT/la-build > $ROOT/la-install.log 2>&1
echo "[$LABEL] locality_aware built"

cp -r $CAB_PREFIX $ROOT/cab-install
if [ "$CAB_PATCH" != "-" ]; then
    rsync -a --exclude 'build-linux-*' --exclude '.spack*' $ENV/cabana/ $ROOT/cab-src/
    git -C $ROOT/cab-src apply --whitespace=nowarn "$CAB_PATCH"
    for f in $(git -C $ROOT/cab-src diff --name-only); do
        case "$f" in
            *.hpp|*.h) dest=$(find $ROOT/cab-install/include -name "$(basename $f)" | head -1)
                       [ -z "$dest" ] && { echo "no installed header for $f"; exit 1; }
                       cp $ROOT/cab-src/$f $dest; echo "[$LABEL] overlaid $f";;
            *) echo "[$LABEL] cabana patch changes non-header $f; header overlay cannot apply it"; exit 1;;
        esac
    done
fi

cmake -S $BENCH -B $ROOT/bench-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$GCC/gcc -DCMAKE_CXX_COMPILER=$GCC/g++ $MPI_ARGS \
  -DCMAKE_CXX_FLAGS="-march=sapphirerapids" \
  -DCabana_DIR=$ROOT/cab-install/share/cmake/Cabana -DKokkos_DIR=$KOKKOS_DIR \
  -Dlocality_aware_DIR=$ROOT/la-install/share/locality_aware \
  -DTCLAP_INCLUDE_DIR=$VIEW/include -DNLOHMANN_JSON_INCLUDE_DIR=$VIEW/include \
  > $ROOT/bench-cmake.log 2>&1
cmake --build $ROOT/bench-build -j16 > $ROOT/bench-build.log 2>&1
grep -q "$ROOT/la-install" $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/link.txt || { echo "[$LABEL] benchmark not linked to candidate library"; exit 1; }
grep -q "$ROOT/cab-install" $ROOT/bench-build/src/CMakeFiles/irregular-benchmarking.dir/flags.make || { echo "[$LABEL] benchmark not using candidate Cabana headers"; exit 1; }
ldd $ROOT/bench-build/src/irregular-benchmarking | grep -q "openmpi-4.1.2-gcc-13.3.1/lib/libmpi.so" || { echo "[$LABEL] benchmark does not link Open MPI 4.1.2"; exit 1; }
ldd $ROOT/bench-build/src/irregular-benchmarking | grep -q mvapich && { echo "[$LABEL] benchmark links mvapich2"; exit 1; }
echo "[$LABEL] benchmark built: $ROOT/bench-build/src/irregular-benchmarking"
