#!/usr/bin/env python3
"""Turn the Vernier captures of ABCG's SpMV into benchmark inputs.

CAPTURE_DIR/order.csv (written by pilot.sbatch) lists the captures in the order they
ran: order,matrix,pm_file,pm_sha256,capture_dir,exit. Each capture is one
benchmark_spmv run (MPI arm, one exchange) whose Vernier pattern is named after the
matrix, so every file names its matrix in its own pattern_map.

Writes, in that order:
  OUT_DIR/coarse.json       one coarse_<matrix> entry per capture: rank 0's
                            hist-aggregate histograms without Vernier's
                            [{pattern_map}, {comm_map}, ...] wrapper, the shape
                            the benchmark reads with -f
  OUT_DIR/generated_inputs  coarse.json, then each capture directory (read by
                            the benchmark as the fine pattern fine_<matrix>)
Fails, writing nothing, if a capture exited non-zero, holds another pattern than
its matrix, or lacks a per-rank combined file for any rank.
Usage: vernier-config.py CAPTURE_DIR OUT_DIR
"""
import csv
import json
import os
import sys


def load(path):
    # Vernier writes [{"pattern_map": "name:id,..."}, {"comm_map": "size|..."}, {histograms}].
    with open(path) as f:
        header, comms, histograms = json.load(f)
    names = dict(pair.split(":") for pair in header["pattern_map"].split(","))
    world_size = int(comms["comm_map"].split("|")[0])
    return names, world_size, histograms


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} CAPTURE_DIR OUT_DIR")
    capture_dir, out_dir = sys.argv[1], sys.argv[2]
    with open(os.path.join(capture_dir, "order.csv")) as f:
        rows = sorted(csv.DictReader(f), key=lambda row: int(row["order"]))

    coarse = {}
    fine_dirs = []
    for row in rows:
        matrix, run_dir = row["matrix"], row["capture_dir"]
        if row["exit"] != "0":
            sys.exit(f"capture of {matrix} exited {row['exit']}; see {run_dir}")
        names, world_size, histograms = load(os.path.join(run_dir, "vernier-0-0.hist-aggregate.json"))
        if list(names) != [matrix]:
            sys.exit(f"{run_dir} holds patterns {list(names)}, expected only {matrix}")
        coarse["coarse_" + matrix] = histograms["pattern_" + names[matrix]]
        for rank in range(world_size):
            combined = os.path.join(run_dir, f"vernier-{rank}-0.hist-combined.json")
            if not os.path.isfile(combined):
                sys.exit(f"{run_dir} has no combined file for rank {rank} of {world_size}")
        if os.path.exists(os.path.join(run_dir, f"vernier-{world_size}-0.hist-combined.json")):
            sys.exit(f"{run_dir} has combined files beyond its {world_size} ranks")
        fine_dirs.append(os.path.abspath(run_dir))

    coarse_path = os.path.abspath(os.path.join(out_dir, "coarse.json"))
    with open(coarse_path, "w") as f:
        json.dump(coarse, f, indent=2)
    with open(os.path.join(out_dir, "generated_inputs"), "w") as f:
        f.write("\n".join([coarse_path] + fine_dirs) + "\n")


if __name__ == "__main__":
    main()
