"""Parse every launch of the five-way campaign into one table.

Reads data/campaign/<jobid>-gatherdata-<N>/manifest.csv and each run file it lists, and
writes results/runs.csv (one row per manifest row: identifiers, wall time, the
min/max/avg of every metric, q = timedTotal max / n in ms per iteration, the verify
and check flags) and results/launches.csv (one row per launch: node count, node list,
rows, complete or not). A run file with no manifest row (the run in progress when a
partial launch was pulled) is ignored.

Usage: python load.py
"""
import csv
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "campaign"
RESULTS = HERE.parent / "results"
RUNS_PER_LAUNCH = 126

# Numeric rows only: ABCG also prints "(pattern)  check skipped: ..." lines.
ROW = re.compile(r"^\((\S+)\)\s+(\w+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*$")
KIND_PREFIXES = ["abcgspmvnp_", "abcgspmv_", "abcgcgnp_", "abcgcg_", "fine_", "coarse_"]


def split_pattern(pattern):
    for prefix in KIND_PREFIXES:
        if pattern.startswith(prefix):
            return prefix[:-1], pattern[len(prefix):]
    return "pm", pattern


def parse_run(path):
    metrics, text = {}, path.read_text()
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            for stat, value in zip(("min", "max", "avg"), m.group(3, 4, 5)):
                metrics[f"{m.group(2)}_{stat}"] = float(value)
    flags = {
        "verify_pass": text.count("verify: PASS"),
        "verify_fail": text.count("verify: FAIL"),
        "check_failed": text.count("check FAILED"),
    }
    return metrics, flags


def launch_nodes(folder):
    for prov in folder.glob("provenance_*.txt"):
        if prov.name.count(".") > 1:      # .binding.txt / .fabric.txt
            continue
        for line in prov.read_text().splitlines():
            if line.startswith("nodes:"):
                return line.split(":", 1)[1].split()
    return []


def main():
    RESULTS.mkdir(exist_ok=True)
    rows, launches = [], []
    for folder in sorted(DATA.glob("*-gatherdata-*")):
        launch, _, nodes = folder.name.split("-")
        nodes = int(nodes)
        with open(folder / "manifest.csv") as f:
            manifest = list(csv.DictReader(f))
        node_list = launch_nodes(folder)
        launches.append({"launch": launch, "nodes": nodes, "rows": len(manifest),
                         "complete": len(manifest) == RUNS_PER_LAUNCH,
                         "node_list": " ".join(node_list)})
        for r in manifest:
            metrics, flags = parse_run(folder / r["file"])
            kind, matrix = split_pattern(r["pattern"])
            start = datetime.fromisoformat(r["start"])
            end = datetime.fromisoformat(r["end"])
            n = int(r["n"])
            rows.append({
                "nodes": nodes, "launch": launch, "position": int(r["run"]),
                "pattern": r["pattern"], "kind": kind, "matrix": matrix, "arm": r["arm"],
                "k": int(r["k"]), "n": n, "start": r["start"], "end": r["end"],
                "wall_s": (end - start).total_seconds(), "exit": int(r["exit"]),
                "file": r["file"], "q_ms": metrics["timedTotal_max"] / n,
                **flags, **metrics,
            })
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "runs.csv", index=False)
    pd.DataFrame(launches).to_csv(RESULTS / "launches.csv", index=False)
    print(f"{len(runs)} runs from {len(launches)} launches")
    print(runs.groupby("nodes").agg(runs=("q_ms", "size"), launches=("launch", "nunique")))


if __name__ == "__main__":
    main()
