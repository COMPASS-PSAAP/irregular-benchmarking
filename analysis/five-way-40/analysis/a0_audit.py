"""Analysis 0: spot-check the established pass checks and take the slow-run census.

Writes tables/audit.csv (per node count: launches, runs, failures, fine == pm count
check, rank spread of timedTotal, n consistency, CG iteration cap) and
tables/slow_runs.csv (every flagged or >15 min run, with its predecessor, position,
and the launch's node list), and prints both. Also tables/stall_quantum.csv and two
figures: see stall_quantum().
"""
import numpy as np
import pandas as pd

from common import (ARM_COLOR, ARMS, NODES, RESULTS, SLOW_FACTOR, load_runs, new_fig, save_fig,
                    write_table)

COUNT_METRICS = [f"{m}_{s}" for m in ("numberOfmessages", "data_sent") for s in ("min", "max", "avg")]


def main():
    runs = load_runs()
    launches = pd.read_csv(RESULTS / "launches.csv")
    audit = []
    for nodes in NODES:
        r = runs[runs.nodes == nodes]
        bench_mpia = r[r.kind.isin(["pm", "fine", "coarse"]) & r.arm.isin(["STANDARD", "LOCALITY"])]
        pm = r[r.kind == "pm"].set_index(["launch", "matrix", "arm"])[COUNT_METRICS]
        fine = r[r.kind == "fine"].set_index(["launch", "matrix", "arm"])[COUNT_METRICS]
        both = pm.join(fine, lsuffix="_pm", rsuffix="_fine", how="inner")
        fine_mismatch = sum((both[f"{m}_pm"] != both[f"{m}_fine"]).sum() for m in COUNT_METRICS)
        spread = (r.timedTotal_max - r.timedTotal_min) / r.timedTotal_max
        n_per_pattern = r.groupby("pattern")["n"].nunique()
        cg = r[r.kind.isin(["abcgcg", "abcgcgnp"])]
        audit.append({
            "nodes": nodes,
            "launches": r.launch.nunique(),
            "complete_launches": int(launches[(launches.nodes == nodes) & launches.complete].shape[0]),
            "runs": len(r),
            "exit_nonzero": int((r.exit != 0).sum()),
            "verify_fail": int(r.verify_fail.sum()),
            "bench_mpia_runs_without_pass": int((bench_mpia.verify_pass < 1).sum()),
            "abcg_check_failed": int(r.check_failed.sum()),
            "fine_vs_pm_pairs": len(both),
            "fine_vs_pm_mismatched_values": int(fine_mismatch),
            "rank_spread_timedTotal_max": float(spread.max()),
            "rank_spread_timedTotal_median": float(spread.median()),
            "patterns_with_varying_n": int((n_per_pattern > 1).sum()),
            "cg_runs": len(cg),
            "cg_runs_reference_iters_500": int((cg.cgReferenceIters_max == 500).sum()),
            "cg_runs_no_solve_finished": int((cg.cgItersPerSolve_max == 0).sum()),
            "slow_runs": int(r.slow.sum()),
            "runs_over_15min": int((r.wall_s > 900).sum()),
        })
    audit = pd.DataFrame(audit)
    write_table(audit, "audit.csv")
    print(audit.T.to_string())

    runs = runs.sort_values(["launch", "position"])
    runs["prev_pattern"] = runs.groupby("launch")["pattern"].shift()
    runs["prev_arm"] = runs.groupby("launch")["arm"].shift()
    cell_median = runs.groupby(["nodes", "pattern", "arm"])["q_ms"].transform("median")
    runs["q_over_cell_median"] = runs.q_ms / cell_median
    flagged = runs[runs.slow | (runs.wall_s > 900)].merge(
        launches[["launch", "node_list"]], on="launch")
    cols = ["nodes", "launch", "position", "pattern", "arm", "n", "q_ms", "q_over_cell_median",
            "wall_s", "slow", "prev_pattern", "prev_arm", "start", "node_list"]
    flagged = flagged[cols].sort_values(["nodes", "launch", "position"])
    write_table(flagged, "slow_runs.csv")
    print(f"\nflagged runs (q > {SLOW_FACTOR:g}x cell median, or wall > 15 min):")
    print(flagged.drop(columns="node_list").to_string(index=False))
    stall_quantum(runs)


def stall_quantum(runs):
    """Found while auditing, not pre-stated: in many cells the run-to-run differences in
    total timed time are integer multiples of one constant. Per (nodes, pattern, arm)
    cell whose timedTotal spans >= 0.3 s, scan periods Q in [0.08, 0.30] s and record the
    phase coherence |mean exp(2 pi i dT / Q)| over pairwise differences dT < 2 s (1 =
    every difference is a multiple of Q; unstructured data gives ~0.1)."""
    periods = np.linspace(0.08, 0.30, 441)
    rows = []
    for (nodes, pattern, arm), d in runs.groupby(["nodes", "pattern", "arm"]):
        t = np.sort(d.timedTotal_max.to_numpy() / 1000)
        if t[-1] - t[0] < 0.3:
            continue
        dt = np.abs(t[:, None] - t[None, :])[np.triu_indices(len(t), 1)]
        dt = dt[dt < 2]
        if len(dt) < 50:
            continue
        power = np.abs(np.exp(2j * np.pi * dt[None, :] / periods[:, None]).mean(axis=1))
        j = power.argmax()
        rows.append({"nodes": nodes, "pattern": pattern, "kind": d.kind.iloc[0], "matrix": d.matrix.iloc[0],
                     "arm": arm, "n": int(d.n.iloc[0]), "runs": len(t), "timedTotal_range_s": t[-1] - t[0],
                     "best_period_s": periods[j], "coherence": power[j],
                     "coherence_at_0.160": power[np.abs(periods - 0.160).argmin()],
                     "median_coherence": float(np.median(power)),
                     "q_step_ms_if_0.160": 160.0 / d.n.iloc[0]})
    table = pd.DataFrame(rows).sort_values("coherence", ascending=False)
    write_table(table, "stall_quantum.csv")
    strong = table[table.coherence > 0.5]
    print(f"\nstall quantum: {len(table)} cells span >= 0.3 s; {len(strong)} have coherence > 0.5, "
          f"best period median {strong.best_period_s.median():.4f} s "
          f"(IQR {strong.best_period_s.quantile(.25):.4f}-{strong.best_period_s.quantile(.75):.4f})")
    print(pd.crosstab(strong.kind, strong.arm))

    best = table.iloc[0]
    d = runs[(runs.nodes == best.nodes) & (runs.pattern == best.pattern) & (runs.arm == best.arm)]
    excess = (d.timedTotal_max - d.timedTotal_max.min()) / 1000 / 0.160
    fig, ax = new_fig("Slow runs lose whole 0.160-second stalls",
                      "extra time compared with the fastest launch (in units of 0.160 s)", "number of launches",
                      caption=f"{best.pattern}, {best.arm} arm, {best.nodes} nodes: the cell where the pattern is "
                              f"clearest. Each bar counts launches by how much longer this run's timed loop took than "
                              f"the fastest launch's. The bars sit on whole numbers, so runs lose a whole number of "
                              f"0.160 s stalls rather than running uniformly slower. {len(d)} launches, {best.n} timed "
                              f"iterations per run.")
    ax.hist(excess, bins=np.arange(-0.05, excess.max() + 0.15, 0.1), color="#333333")
    ax.set_xticks(np.arange(0, np.ceil(excess.max()) + 1, max(1, int(excess.max() // 15))))
    save_fig(fig, f"stall_quantum_excess_{best.pattern}_{best.arm}_{best.nodes}nodes.png")

    fig, ax = new_fig("Where the 0.160-second stall step shows up",
                      "spread of the timed total across launches (s)", "fit to 0.160 s steps",
                      caption=f"One point per node count, pattern, and arm whose timed total varies by at least "
                              f"0.3 s across launches ({len(table)} of them; {table.runs.min()}-{table.runs.max()} launches each). The "
                              f"vertical axis measures how well the launch-to-launch differences line up on multiples "
                              f"of 0.160 s: 1 means perfectly, about 0.1 means no pattern. Points above the dashed "
                              f"line show the stall step clearly, and nearly all of them are MPI or STANDARD.")
    for arm in ARMS:
        s = table[table.arm == arm]
        ax.scatter(s.timedTotal_range_s, s["coherence_at_0.160"], s=14, label=arm, color=ARM_COLOR[arm])
    ax.set_xscale("log")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.legend()
    save_fig(fig, "stall_quantum_coherence_by_cell.png")


if __name__ == "__main__":
    main()
