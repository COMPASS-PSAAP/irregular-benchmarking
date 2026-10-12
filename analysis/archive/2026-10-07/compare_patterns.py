"""Compare each SuiteSparse matrix with the sampled pattern that stands in for it.

The sampled campaign (data-9-30-2026/data-old-locality, patterns drawn from t2.json)
and the matrix campaign (data/, the .pm files) both ran locality_aware 752a19b with
-s NUMA and 40 launches per node count. PATTERN_MAP is NicksBenchmarkDumps'
pattern_map.txt. For each node count and pair it reports the communication structure
(per rank per gather, median over runs), the median time per gather and startup
window of each arm (max over ranks), and N* for LOCALITY vs MPI from each campaign's
nstar.csv. The two campaigns ran on different allocations at different times, so
the comparison is unpaired.

Usage: python compare_patterns.py      Writes results/compare_patterns.csv.
"""
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLED_DATA = HERE.parent / "data-9-30-2026" / "data-old-locality"
SAMPLED_RESULTS = HERE.parent / "data-9-30-2026" / "results-old-locality"
MATRIX_DATA = HERE / "data"
MATRIX_RESULTS = HERE / "results"
PATTERN_MAP = {"pattern_0": "crankseg_2", "pattern_1": "nd12k", "pattern_2": "x104",
               "pattern_3": "crankseg_1", "pattern_4": "m_t1", "pattern_5": "nd24k"}
ARMS = ["MPI", "STANDARD", "LOCALITY"]


def parse_run(path):
    """-> (pattern, arm, n, {row: (min, max, avg)}) or None for a run without results."""
    comm = alltoallv = n = pattern = None
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("-MPI:"):
            comm = line.split(":", 1)[1].strip()
        elif line.startswith("-alltoallv:"):
            alltoallv = line.split(":", 1)[1].strip()
        elif line.startswith("-timed iterations (n):"):
            n = int(line.split(":", 1)[1])
        else:
            m = re.match(r"^\((.+)\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
            if m:
                try:
                    rows[m.group(2)] = tuple(float(m.group(i)) for i in (3, 4, 5))
                    pattern = m.group(1)
                except ValueError:
                    pass
    if "timedTotal" not in rows or not n:
        return None
    arm = "MPI" if comm == "MPI" else alltoallv.rsplit("_", 1)[-1]
    return pattern, arm, n, rows


def load(data_dir, rename):
    """-> {(nodes, name, arm): [rows dict with n]} over every run with results."""
    runs = defaultdict(list)
    for folder in data_dir.glob("*-gatherdata-*"):
        nodes = int(folder.name.rsplit("-", 1)[1])
        for path in folder.glob("run_*.txt"):
            run = parse_run(path)
            if run:
                pattern, arm, n, rows = run
                runs[(nodes, rename.get(pattern, pattern), arm)].append(dict(rows, n=n))
    return runs


def nstar(results_dir, rename):
    out = {}
    with (results_dir / "nstar.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["comparison"] == "LOCALITY vs MPI" and float(r["f"]) == 0.0:
                out[(int(r["nodes"]), rename.get(r["pattern"], r["pattern"]), r["procedure"])] = r["N_star"]
    return out


def summarize(runs, key):
    """Structure from the MPI arm's runs (the pattern is the same on every arm)."""
    mpi = runs[key + ("MPI",)]
    med = lambda f: st.median(f(r) for r in mpi)
    s = {
        "msgs_avg": med(lambda r: r["numberOfmessages"][2]),
        "msgs_max": med(lambda r: r["numberOfmessages"][1]),
        "bytes_avg": med(lambda r: r["data_sent"][2]),
        "bytes_max": med(lambda r: r["data_sent"][1]),
        "off_node_msg_frac": med(lambda r: r["offNodeNeighbors"][2] / r["numberOfmessages"][2]),
        "off_node_byte_frac": med(lambda r: r["offNodeBytes"][2] / r["data_sent"][2]),
        "n": med(lambda r: r["n"]),
    }
    for arm in ARMS:
        rs = runs[key + (arm,)]
        s[f"{arm}_ms_per_gather"] = st.median(r["timedTotal"][1] / r["n"] for r in rs) if rs else float("nan")
        s[f"{arm}_window_ms"] = st.median(r["startupWindow"][1] for r in rs) if rs else float("nan")
    return s


def main():
    sampled = load(SAMPLED_DATA, PATTERN_MAP)
    matrix = load(MATRIX_DATA, {})
    n_sampled = nstar(SAMPLED_RESULTS, PATTERN_MAP)
    n_matrix = nstar(MATRIX_RESULTS, {})
    inverse = {v: k for k, v in PATTERN_MAP.items()}

    rows = []
    for nodes in sorted({k[0] for k in matrix}):
        for name in PATTERN_MAP.values():
            for source, runs, ns in (("sampled", sampled, n_sampled), ("matrix", matrix, n_matrix)):
                if (nodes, name, "MPI") not in runs:
                    continue
                s = summarize(runs, (nodes, name))
                s.update(nodes=nodes, matrix=name, sampled_pattern=inverse[name], source=source,
                         nstar_median=ns.get((nodes, name, "sign_median"), ""),
                         nstar_order=ns.get((nodes, name, "order_statistic"), ""))
                rows.append(s)

    header = ["nodes", "matrix", "sampled_pattern", "source", "n", "msgs_avg", "msgs_max",
              "bytes_avg", "bytes_max", "off_node_msg_frac", "off_node_byte_frac"] + \
             [f"{a}_ms_per_gather" for a in ARMS] + [f"{a}_window_ms" for a in ARMS] + \
             ["nstar_median", "nstar_order"]
    with (MATRIX_RESULTS / "compare_patterns.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, header)
        w.writeheader()
        w.writerows(rows)

    print("per rank per gather, medians over runs; times are max over ranks (ms); "
          "N* is LOCALITY vs MPI at f = 0 (sign test / order statistic)")
    print(f"{'nodes':>5} {'matrix':<11}{'source':<8}{'msgs avg/max':>14}{'bytes avg/max':>16}"
          f"{'off-node msg/B':>16}{'MPI':>9}{'STD':>9}{'LOC':>9}{'LOC wnd':>9}{'N*':>14}")
    for r in rows:
        print(f"{r['nodes']:>5} {r['matrix']:<11}{r['source']:<8}"
              f"{r['msgs_avg']:>7.0f}/{r['msgs_max']:<6.0f}{r['bytes_avg']:>8.0f}/{r['bytes_max']:<7.0f}"
              f"{r['off_node_msg_frac']:>8.2f}/{r['off_node_byte_frac']:<7.2f}"
              f"{r['MPI_ms_per_gather']:>9.3f}{r['STANDARD_ms_per_gather']:>9.3f}"
              f"{r['LOCALITY_ms_per_gather']:>9.3f}{r['LOCALITY_window_ms']:>9.1f}"
              f"{r['nstar_median']:>7}/{r['nstar_order']:<6}")


if __name__ == "__main__":
    main()
