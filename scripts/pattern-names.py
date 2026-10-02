#!/usr/bin/env python3
"""Print the pattern names in a benchmark pattern config file, one per line.

A pattern config maps each pattern name to its statistics, so the names are the
top-level keys. Usage: pattern-names.py CONFIG.json
"""
import json
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} CONFIG.json")
    with open(sys.argv[1]) as f:
        patterns = json.load(f)
    for name in patterns:
        print(name)


if __name__ == "__main__":
    main()
