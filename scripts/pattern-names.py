#!/usr/bin/env python3
"""Print the pattern names in a benchmark input file, one per line.

A JSON pattern config maps each pattern name to its statistics, so the names are
the top-level keys; a .pm matrix is a single pattern named after the file; a Vernier
capture directory (see vernier-config.py) is the single pattern fine_<name>, where
<name> is the pattern Vernier recorded in the capture's own files.
Usage: pattern-names.py CONFIG.json|MATRIX.pm|CAPTURE_DIR
"""
import json
import os
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} CONFIG.json|MATRIX.pm|CAPTURE_DIR")
    path = sys.argv[1]
    # A PETSc binary matrix is one pattern, named after the file, as the benchmark names it.
    if path.endswith(".pm"):
        print(os.path.splitext(os.path.basename(path))[0])
        return
    if os.path.isdir(path):
        with open(os.path.join(path, "vernier-0-0.hist-combined.json")) as f:
            pattern_map = json.load(f)[0]["pattern_map"]
        for pair in pattern_map.split(","):
            print("fine_" + pair.split(":")[0])
        return
    with open(path) as f:
        patterns = json.load(f)
    for name in patterns:
        print(name)


if __name__ == "__main__":
    main()
