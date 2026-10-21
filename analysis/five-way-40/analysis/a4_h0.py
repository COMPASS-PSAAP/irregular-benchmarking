"""Analysis 4 (H0): does the application's best arm change across workloads, and does
per-workload selection beat the best single fixed arm?

Reference kinds: abcgspmv and abcgcg (primary), abcgspmvnp and abcgcgnp (secondary).
  best-arm probability  per node count x matrix: the share of launch-bootstrap
                        resamples in which each arm has the lowest median q.
  selection gain        workloads = the 6 matrices at one node count (H0 as stated), or
                        the 4 node counts of one matrix. Each repeat resamples every
                        workload's launches, splits them in half, picks the per-workload
                        arm and the single fixed arm (lowest summed log median over the
                        workloads) on one half, and scores both on the other half:
                        gain = geometric mean over workloads of q(fixed) / q(selected) - 1.
                        Cross-fitting keeps the selection from being scored on the data
                        it was picked from. Reported: median and 2.5-97.5% over repeats,
                        plus the fixed arm's worst-workload regret on the held-out half.
Slow runs kept and dropped. Writes tables/h0_best_arm.csv, tables/h0_selection_gain.csv,
figures h0_best_arm_<reference>_<N>nodes.png and h0_selection_gain_<reference>.png.
"""
import numpy as np
import pandas as pd

from common import HUMAN, ARM_COLOR, ARMS, B, MATRICES, NODES, REFERENCES, SEED, launch_resamples, load_runs, new_fig, q_table, save_fig, write_table

REPEATS = 2000


def best_arm_probabilities(R):
    idx = launch_resamples(len(R))
    best = np.median(R[idx], axis=1).argmin(axis=1)
    return [float(np.mean(best == a)) for a in range(3)]


def selection_gain(tables, rng):
    """tables: list of (L_i, 3) arrays, one per workload."""
    gains, fixed_worst, fixed_choice = [], [], []
    for _ in range(REPEATS):
        train, test = [], []
        for R in tables:
            Rb = R[rng.integers(0, len(R), len(R))]
            perm = rng.permutation(len(Rb))
            half = len(Rb) // 2
            train.append(np.log(np.median(Rb[perm[:half]], axis=0)))
            test.append(np.log(np.median(Rb[perm[half:]], axis=0)))
        train, test = np.array(train), np.array(test)              # (workloads, 3)
        selected = train.argmin(axis=1)
        fixed = train.sum(axis=0).argmin()
        w = np.arange(len(tables))
        gains.append(np.exp(np.mean(test[w, fixed] - test[w, selected])) - 1)
        fixed_worst.append(np.exp(np.max(test[w, fixed] - test.min(axis=1))) - 1)
        fixed_choice.append(fixed)
    gains, fixed_worst = np.array(gains), np.array(fixed_worst)
    counts = np.bincount(fixed_choice, minlength=3) / REPEATS
    return {"gain_median": np.median(gains), "gain_lo": np.quantile(gains, 0.025),
            "gain_hi": np.quantile(gains, 0.975), "p_gain_positive": float(np.mean(gains > 0)),
            "fixed_worst_regret_median": np.median(fixed_worst),
            "fixed_worst_regret_lo": np.quantile(fixed_worst, 0.025),
            "fixed_worst_regret_hi": np.quantile(fixed_worst, 0.975),
            **{f"fixed_is_{arm}": counts[a] for a, arm in enumerate(ARMS)}}


def main():
    runs = load_runs()
    rng = np.random.default_rng(SEED)
    best_rows, gain_rows = [], []
    for reference in REFERENCES:
        for drop in (False, True):
            label = "dropped" if drop else "kept"
            tables = {}
            for nodes in NODES:
                for matrix in MATRICES:
                    T = q_table(runs, nodes, matrix, [reference], drop_slow=drop)
                    R = T[reference][ARMS].to_numpy()
                    tables[(nodes, matrix)] = R
                    med = np.median(R, axis=0)
                    probs = best_arm_probabilities(R)
                    best_rows.append({"reference": reference, "slow_runs": label, "nodes": nodes, "matrix": matrix,
                                      "launches": len(R), "best_arm": ARMS[int(med.argmin())],
                                      **{f"median_q_ms_{arm}": med[a] for a, arm in enumerate(ARMS)},
                                      **{f"p_best_{arm}": probs[a] for a, arm in enumerate(ARMS)}})
            for nodes in NODES:
                g = selection_gain([tables[(nodes, m)] for m in MATRICES], rng)
                gain_rows.append({"reference": reference, "slow_runs": label, "across": "matrices",
                                  "fixed_at": f"{nodes} nodes", **g})
            for matrix in MATRICES:
                g = selection_gain([tables[(n, matrix)] for n in NODES], rng)
                gain_rows.append({"reference": reference, "slow_runs": label, "across": "node counts",
                                  "fixed_at": matrix, **g})
    best, gain = pd.DataFrame(best_rows), pd.DataFrame(gain_rows)
    write_table(best, "h0_best_arm.csv")
    write_table(gain, "h0_selection_gain.csv")

    for reference in ["abcgspmv", "abcgcg"]:
        for nodes in NODES:
            t = best[(best.reference == reference) & (best.nodes == nodes) & (best.slow_runs == "kept")]
            t = t.set_index("matrix").reindex(MATRICES)
            fig, ax = new_fig(f"Which arm is fastest for {HUMAN[reference]}, {nodes} nodes",
                              "matrix", "probability of being fastest",
                              caption=f"Bar segments: the share of {B:,} launch-bootstrap resamples in which each arm "
                                      f"has the lowest median time. A single solid color is a clear winner; mixed "
                                      f"colors mean the data cannot separate the arms. Launches per matrix: "
                                      f"{', '.join(str(v) for v in t.launches)}; slow runs kept.")
            bottom = np.zeros(len(t))
            for arm in ARMS:
                ax.bar(MATRICES, t[f"p_best_{arm}"], bottom=bottom, color=ARM_COLOR[arm], label=arm)
                bottom += t[f"p_best_{arm}"].to_numpy()
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, loc="lower right")
            save_fig(fig, f"h0_best_arm_{reference}_{nodes}nodes.png")

        t = gain[(gain.reference == reference) & (gain.slow_runs == "kept")]
        fig, ax = new_fig(f"Choosing the arm per workload vs. one fixed arm, {HUMAN[reference]}",
                          "set of workloads compared", "speedup over the best fixed arm (%)", width=9,
                          caption=f"How much faster {HUMAN[reference]} runs if each workload gets its own best arm "
                                  f"than if every workload uses the single best arm (geometric mean; 0% = no "
                                  f"benefit). Left four: the 6 matrices at one node count. Right six: the 4 node "
                                  f"counts of one matrix. Arms are chosen on half the launches and scored on the "
                                  f"other half; points are medians and bars 95% ranges over {REPEATS:,} bootstrap "
                                  f"splits. Slow runs kept.")
        x = np.arange(len(t))
        ax.errorbar(x, t.gain_median * 100, yerr=np.vstack([(t.gain_median - t.gain_lo) * 100,
                                                            (t.gain_hi - t.gain_median) * 100]),
                    fmt="o", capsize=3, color="black")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x, [f"{a}\nat {f}" for a, f in zip(t.across, t.fixed_at)], fontsize=7)
        ax.set_yscale("symlog", linthresh=1)
        save_fig(fig, f"h0_selection_gain_{reference}.png")

    print(best[best.slow_runs == "kept"].pivot_table(index=["nodes", "matrix"], columns="reference",
                                                     values="best_arm", aggfunc="first").to_string())
    print(gain[gain.slow_runs == "kept"][["reference", "across", "fixed_at", "gain_median", "gain_lo", "gain_hi",
                                          "fixed_worst_regret_median", "fixed_is_LOCALITY"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
