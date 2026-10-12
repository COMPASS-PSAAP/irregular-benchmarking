"""Per launch, each matrix beside its old profile pattern (pattern_k), coarse_<matrix> and
fine_<matrix>: ms per gather (timedTotal max / n), message and byte rows, per arm.
Usage: python compare_launch.py PILOT_DIR LAUNCH_DIR [LAUNCH_DIR...]
"""
import csv, os, re, sys

ROW = re.compile(r"^\((\S+)\)\s+(\w+)\s+(\S+)\s+(\S+)\s+(\S+)")


def read_run(path):
    rows = {}
    with open(path) as f:
        for line in f:
            m = ROW.match(line)
            if m:
                rows[m.group(2)] = tuple(float(x) for x in m.group(3, 4, 5))
    return rows


def main():
    pilot, launches = sys.argv[1], sys.argv[2:]
    with open(os.path.join(pilot, "capture", "order.csv")) as f:
        order = sorted(csv.DictReader(f), key=lambda r: int(r["order"]))
    for launch in launches:
        runs = {}
        with open(os.path.join(launch, "manifest.csv")) as f:
            for r in csv.DictReader(f):
                runs[(r["pattern"], r["arm"])] = (int(r["n"]), read_run(os.path.join(launch, r["file"])))
        print(f"\n== {os.path.basename(launch)}")
        print(f"{'matrix':11} {'pattern':18} {'MPI ms':>8} {'STD ms':>8} {'LOC ms':>8} {'LOC win':>8} {'msgs min/max/avg':>18} {'B/rank avg':>10} {'offNode msg':>11}")
        for row in order:
            matrix = row["matrix"]
            for name in [matrix, f"pattern_{int(row['order']) - 1}", "coarse_" + matrix, "fine_" + matrix]:
                ms = []
                for arm in ["MPI", "STANDARD", "LOCALITY"]:
                    n, rows = runs[(name, arm)]
                    ms.append(rows["timedTotal"][1] / n)
                n, rows = runs[(name, "MPI")]
                msgs = rows["numberOfmessages"]
                off = rows["offNodeNeighbors"][2] / msgs[2] if msgs[2] else float("nan")
                loc_win = runs[(name, "LOCALITY")][1]["startupWindow"][1]
                print(f"{matrix:11} {name:18} {ms[0]:8.3f} {ms[1]:8.3f} {ms[2]:8.3f} {loc_win:8.1f} "
                      f"{msgs[0]:5.0f}/{msgs[1]:4.0f}/{msgs[2]:6.1f} {rows['data_sent'][2]:10.0f} {off:11.3f}")


if __name__ == "__main__":
    main()
