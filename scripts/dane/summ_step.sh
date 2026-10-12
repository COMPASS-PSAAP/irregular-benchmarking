#!/bin/bash
# One line per output file of a step: per-apply time (ms, apply total / 1000),
# firstApply and haloTime (ms), all averaged over every sample and pattern,
# plus sample count and verify PASS count.
# Only files from this walk (newer than fp.txt), so older runs with the same tag are skipped.
# usage: summ_step.sh <tag> [tag...]
cd /g/g20/bacon4/benchmark26/irregular-benchmarking-new/build1/src
printf "%-16s %-8s %-15s %10s %10s %10s %4s %4s\n" tag mode started apply/it firstApply haloTime n pass
for TAG in "$@"; do
  for f in $(find . -maxdepth 1 -name "out_1000_*_t2_${TAG}_2*.txt" -newer $HOME/cand/steps/fp.txt | sort); do
    mode=$(echo $f | cut -d_ -f3); ts=$(echo $f | sed 's/.*_\([0-9]*-[0-9]*\)\.txt/\1/')
    awk -f $HOME/summ.awk $f | awk -v t=$TAG -v m=$mode -v ts=$ts -v p=$(grep -c "verify: PASS" $f) '
      $1=="apply"{a=$3/1000; n=$4} $1=="firstApply"{fa=$3} $1=="haloTime"{h=$3}
      END{gsub(/[()]/,"",n); printf "%-16s %-8s %-15s %10.4f %10.3f %10.3f %4s %4s\n", t, m, ts, a, fa, h, n, p}'
  done
done
