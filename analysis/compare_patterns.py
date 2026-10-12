"""Compare each SuiteSparse matrix with the sampled pattern that stands in for it,
paired inside the combined campaign's launches.

Each launch of the combined campaign (data/, NicksBenchmarkDumps/data-10-7-2026-combined)
ran the six sampled patterns from t<nodes>.json and the six .pm matrices under MPI,
STANDARD, and LOCALITY on one allocation, so every sampled-vs-matrix contrast is formed
inside a launch. PATTERN_MAP is NicksBenchmarkDumps' pattern_map.txt. For each node
count and pair it reports the communication structure (per rank per gather, median
over runs), the median time per gather of each arm (max over ranks), the median over
launches of the matrix/sampled ratio of that time, and N* for LOCALITY vs MPI from
results/nstar.csv (python.py's output).

Usage: python compare_patterns.py      Writes results/compare_patterns.csv.
"""
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
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


def load(data_dir):
    """-> {(nodes, pattern, arm): {launch: rows dict with n}} over every run with results."""
    runs = defaultdict(dict)
    for folder in data_dir.glob("*-gatherdata-*"):
        launch, nodes = folder.name.split("-")[0], int(folder.name.rsplit("-", 1)[1])
        for path in folder.glob("run_*.txt"):
            run = parse_run(path)
            if run:
                pattern, arm, n, rows = run
                runs[(nodes, pattern, arm)][launch] = dict(rows, n=n)
    return runs


def nstar(results_dir):
    out = {}
    with (results_dir / "nstar.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["comparison"] == "LOCALITY vs MPI" and float(r["f"]) == 0.0:
                out[(int(r["nodes"]), r["pattern"], r["procedure"])] = r["N_star"]
    return out


def per_gather(rows):
    return rows["timedTotal"][1] / rows["n"]


def summarize(runs, nodes, pattern):
    """Structure from the MPI arm's runs (the pattern is the same on every arm)."""
    mpi = list(runs[(nodes, pattern, "MPI")].values())
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
        rs = list(runs[(nodes, pattern, arm)].values())
        s[f"{arm}_ms_per_gather"] = st.median(per_gather(r) for r in rs) if rs else float("nan")
        s[f"{arm}_window_ms"] = st.median(r["startupWindow"][1] for r in rs) if rs else float("nan")
    return s


def paired_ratio(runs, nodes, sampled, matrix, arm):
    """Median over launches of matrix / sampled time per gather, within each launch."""
    a, b = runs[(nodes, matrix, arm)], runs[(nodes, sampled, arm)]
    ratios = [per_gather(a[l]) / per_gather(b[l]) for l in a.keys() & b.keys()]
    return (st.median(ratios) if ratios else float("nan")), len(ratios)


def main():
    runs = load(DATA)
    ns = nstar(RESULTS)
    rows = []
    for nodes in sorted({k[0] for k in runs}):
        for sampled, matrix in PATTERN_MAP.items():
            for source, pattern in (("sampled", sampled), ("matrix", matrix)):
                if (nodes, pattern, "MPI") not in runs:
                    continue
                s = summarize(runs, nodes, pattern)
                s.update(nodes=nodes, matrix=matrix, sampled_pattern=sampled, source=source,
                         nstar_median=ns.get((nodes, pattern, "sign_median"), ""),
                         nstar_order=ns.get((nodes, pattern, "order_statistic"), ""))
                for arm in ARMS:
                    ratio, pairs = paired_ratio(runs, nodes, sampled, matrix, arm)
                    s[f"{arm}_matrix_over_sampled"] = ratio if source == "matrix" else ""
                    s["paired_launches"] = pairs
                rows.append(s)

    header = ["nodes", "matrix", "sampled_pattern", "source", "n", "msgs_avg", "msgs_max",
              "bytes_avg", "bytes_max", "off_node_msg_frac", "off_node_byte_frac"] + \
             [f"{a}_ms_per_gather" for a in ARMS] + [f"{a}_window_ms" for a in ARMS] + \
             [f"{a}_matrix_over_sampled" for a in ARMS] + \
             ["paired_launches", "nstar_median", "nstar_order"]
    with (RESULTS / "compare_patterns.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, header)
        w.writeheader()
        w.writerows(rows)

    print("per rank per gather, medians over runs; times are max over ranks (ms); "
          "ratio = median over launches of matrix/sampled time per gather (MPI, LOCALITY); "
          "N* is LOCALITY vs MPI at f = 0 (sign test / order statistic)")
    print(f"{'nodes':>5} {'matrix':<11}{'source':<8}{'msgs avg/max':>14}{'bytes avg/max':>16}"
          f"{'off-node msg/B':>16}{'MPI':>9}{'STD':>9}{'LOC':>9}{'ratio MPI/LOC':>15}{'N*':>14}")
    for r in rows:
        ratios = "" if r["source"] == "sampled" else \
            f"{r['MPI_matrix_over_sampled']:.1f}/{r['LOCALITY_matrix_over_sampled']:.1f}"
        print(f"{r['nodes']:>5} {r['matrix']:<11}{r['source']:<8}"
              f"{r['msgs_avg']:>7.0f}/{r['msgs_max']:<6.0f}{r['bytes_avg']:>8.0f}/{r['bytes_max']:<7.0f}"
              f"{r['off_node_msg_frac']:>8.2f}/{r['off_node_byte_frac']:<7.2f}"
              f"{r['MPI_ms_per_gather']:>9.3f}{r['STANDARD_ms_per_gather']:>9.3f}"
              f"{r['LOCALITY_ms_per_gather']:>9.3f}{ratios:>15}"
              f"{r['nstar_median']:>7}/{r['nstar_order']:<6}")


if __name__ == "__main__":
    main()
