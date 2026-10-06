#!/usr/bin/env python3
"""Print the pattern names in a benchmark input file, one per line.

A JSON pattern config maps each pattern name to its statistics, so the names are
the top-level keys; a .pm matrix is a single pattern named after the file.
Usage: pattern-names.py CONFIG.json|MATRIX.pm
"""
import json
import os
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} CONFIG.json|MATRIX.pm")
    path = sys.argv[1]
    # A PETSc binary matrix is one pattern, named after the file, as the benchmark names it.
    if path.endswith(".pm"):
        print(os.path.splitext(os.path.basename(path))[0])
        return
    with open(path) as f:
        patterns = json.load(f)
    for name in patterns:
        print(name)


if __name__ == "__main__":
    main()
