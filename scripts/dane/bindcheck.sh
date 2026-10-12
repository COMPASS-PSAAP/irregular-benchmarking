#!/bin/bash
# Print, per rank, the CPUs it may run on and their NUMA node, under the same
# srun layout as the benchmark; then count ranks where rank/14 != global NUMA id.
cd $HOME/cand/steps && sbatch -p pdebug --nodes=2 --ntasks=224 --time=0:05:00 -J step_bindcheck -o bindcheck_%j.out --wrap='
srun --nodes=2 --ntasks-per-node=112 bash -c '"'"'c=$(awk "/Cpus_allowed_list/{print \$2}" /proc/self/status); f=${c%%[-,]*}; n=$(ls -d /sys/devices/system/cpu/cpu$f/node* | sed "s/.*node//"); echo "$SLURM_PROCID $SLURM_NODEID $c $n"'"'"' | sort -n > bindmap.txt
awk "{g=\$2*8+\$4; if (int(\$1/14)!=g) bad++; if (\$3 ~ /[-,]/) multi++} END{print \"ranks\", NR, \"rank/14 != numa:\", bad+0, \"multi-cpu masks:\", multi+0}" bindmap.txt'
