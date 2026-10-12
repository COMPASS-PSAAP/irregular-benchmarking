#!/bin/bash
# Allreduce PMPI rows only, per section x SpMV: CG time per solve (ms) mean/min/max
# over all runs of a label, run count, and each run's CG iteration count.
# usage: summ_abcg.sh <label>...
cd /g/g20/bacon4/benchmark26/ABCG-new/results
printf "%-6s %-18s %-9s %10s %10s %10s %4s  %s\n" label section spmv mean_ms min_ms max_ms runs iters
for L in "$@"; do
  awk -v L=$L '
    /^Testing CG with /{sec=$0; sub(/^Testing CG with /,"",sec); sub(/:$/,"",sec)}
    /^Testing with SpMV /{sp=$4; ar=$0; sub(/.*and Allreduce /,"",ar)}
    /^After /{it=$2}
    /^CG takes /{ if (ar=="PMPI") { k=sec"|"sp; t=$3*1000; n[k]++; s[k]+=t;
        if (!(k in mn) || t<mn[k]) mn[k]=t; if (t>mx[k]) mx[k]=t; its[k]=its[k] it " ";
        if (!(k in ord)) { ord[k]=++c; key[c]=k } } }
    END { for (i=1;i<=c;i++) { k=key[i]; split(k,p,"|");
        printf "%-6s %-18s %-9s %10.3f %10.3f %10.3f %4d  %s\n", L, p[1], p[2], s[k]/n[k], mn[k], mx[k], n[k], its[k] } }' \
    $(ls out_abcg_${L}_2*.txt 2>/dev/null)
done
