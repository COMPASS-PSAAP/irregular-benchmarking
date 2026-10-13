"""Five-way test: each matrix's SpMV halo exchanged every way, in one launch.

For each matrix (in capture order) and arm it lists, per kind of run:
  <matrix>           irregular-benchmarking on the .pm
  coarse_<matrix>    the aggregate Vernier histograms, sampled
  fine_<matrix>      each rank's own Vernier histograms, sampled
  abcgspmv[np]_      ABCG's SpMV exchange, persistent [np: one call per exchange]
  abcgcg[np]_        ABCG's CG iterations (time is per CG iteration, not per gather)
the time per gather (timedTotal max over ranks / n), the startup window (max over
ranks), messages per rank (avg/max), bytes sent per rank (avg/max), and the time
relative to the benchmark's .pm run on the same arm. It also counts verify PASS/FAIL
lines and ABCG "check FAILED" lines.
Usage: python compare_fiveway.py PILOT_DIR LAUNCH_DIR [OUT.csv]
"""
import csv
import os
import re
import sys

ROW = re.compile(r"^\((\S+)\)\s+(\w+)\s+(\S+)\s+(\S+)\s+(\S+)")
ARMS = ["MPI", "STANDARD", "LOCALITY"]


def read_run(path):
    rows, text = {}, open(path).read()
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            rows[m.group(2)] = tuple(float(x) for x in m.group(3, 4, 5))
    return rows, text


def main():
    pilot, launch = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    with open(os.path.join(pilot, "capture", "order.csv")) as f:
        matrices = [r["matrix"] for r in sorted(csv.DictReader(f), key=lambda r: int(r["order"]))]
    runs, passes, fails, abcg_failed, nonzero = {}, 0, 0, 0, 0
    with open(os.path.join(launch, "manifest.csv")) as f:
        manifest = list(csv.DictReader(f))
    for r in manifest:
        rows, text = read_run(os.path.join(launch, r["file"]))
        runs[(r["pattern"], r["arm"])] = (int(r["n"]), rows)
        passes += text.count("verify: PASS")
        fails += text.count("verify: FAIL")
        abcg_failed += text.count("check FAILED")
        nonzero += r["exit"] != "0"
    print(f"{len(manifest)} runs, {nonzero} non-zero exits, verify PASS {passes} FAIL {fails}, "
          f"ABCG check FAILED {abcg_failed}")

    out = []
    for matrix in matrices:
        print(f"\n{matrix}: ms per gather (abcgcg: per CG iteration); x = relative to {matrix} on that arm")
        print(f"  {'run':24} " + " ".join(f"{a:>15}" for a in ARMS) +
              f" {'LOC window':>10} {'msgs avg/max':>13} {'bytes avg/max':>17}")
        kinds = [matrix, "coarse_" + matrix, "fine_" + matrix, "abcgspmv_" + matrix,
                 "abcgspmvnp_" + matrix, "abcgcg_" + matrix, "abcgcgnp_" + matrix]
        for name in kinds:
            cells = []
            for arm in ARMS:
                n, rows = runs[(name, arm)]
                t = rows["timedTotal"][1] / n
                n0, rows0 = runs[(matrix, arm)]
                rel = t / (rows0["timedTotal"][1] / n0)
                cells.append(f"{t:7.3f} ({rel:4.2f}x)")
                out.append({"matrix": matrix, "run": name, "arm": arm, "n": n, "ms_per_gather": t,
                            "relative_to_pm": rel, "window_ms": rows["startupWindow"][1],
                            "msgs_avg": rows["numberOfmessages"][2], "msgs_max": rows["numberOfmessages"][1],
                            "bytes_avg": rows["data_sent"][2], "bytes_max": rows["data_sent"][1]})
            _, loc = runs[(name, "LOCALITY")]
            _, mpi = runs[(name, "MPI")]
            print(f"  {name:24} " + " ".join(f"{c:>15}" for c in cells) +
                  f" {loc['startupWindow'][1]:10.1f} {mpi['numberOfmessages'][2]:6.1f}/{mpi['numberOfmessages'][1]:<6.0f}"
                  f" {mpi['data_sent'][2]:8.0f}/{mpi['data_sent'][1]:<8.0f}")
    if out_path:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
