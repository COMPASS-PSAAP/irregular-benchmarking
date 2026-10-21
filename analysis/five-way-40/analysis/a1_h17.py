"""Analysis 1 (H17): does a stand-in rank the three arms the way the application does?

Per node count x matrix x stand-in (pm, fine, coarse) x reference (abcgspmv, abcgcg
primary; abcgspmvnp, abcgcgnp secondary), over the launches holding all six runs:

  decision regret     the stand-in's arm with the lowest median over launches, scored on
                      the reference's medians: med_ref[choice] / min_a med_ref[a] - 1.
                      This is the plan's launch-time setting (RESEARCH-PLAN section 1).
                      Interval: percentile, joint launch bootstrap; p_choice_changes is
                      the bootstrap probability that the stand-in's choice differs.
  fixed-arm regret    the same for always-MPI, always-STANDARD, always-LOCALITY.
  per-launch regret   within one launch, the stand-in's fastest arm scored on that
                      launch's reference: median and max over launches.
  noise floor         the reference predicting itself from a different launch (all
                      ordered pairs): median, and the mean over cyclic shifts of the max
                      over launches, which is what a max over L launches compares to.
  agreement           per launch, same argmin / same full order; Wilson intervals.
  pairwise            per arm pair, per launch: agree when the stand-in orders the pair
                      the same way, or when the reference is tied (within TIE). Reported
                      with ties counted as agreement and over untied launches only.

Each is computed with slow runs kept and with them dropped (slow_runs column).
Writes tables/h17_design_points.csv, tables/h17_pairwise.csv, tables/h17_prediction.csv,
and figures regret_<stand-in>_vs_<reference>_<N>nodes.png,
pairwise_agreement_<stand-in>_vs_<reference>.png, worst_case_regret.png.
"""
import itertools

import numpy as np
import pandas as pd

from common import (HUMAN, ARMS, B, KIND_LABEL, MATRICES, NODES, REFERENCES, STAND_INS, TIE, launch_resamples,
                    load_runs, new_fig, percentile_interval, q_table, save_fig, wilson, write_table)

PAIRS = [("MPI", "STANDARD"), ("MPI", "LOCALITY"), ("STANDARD", "LOCALITY")]
PRIMARY = ["abcgspmv", "abcgcg"]


def ordering_stats(S, R):
    """S, R: (L, 3) q per launch for stand-in and reference, arms in ARMS order."""
    L = len(S)
    out = {"launches": L}
    med_s, med_r = np.median(S, axis=0), np.median(R, axis=0)
    choice = int(med_s.argmin())
    out["stand_in_choice"] = ARMS[choice]
    out["reference_best"] = ARMS[int(med_r.argmin())]
    out["decision_regret"] = med_r[choice] / med_r.min() - 1
    for a, arm in enumerate(ARMS):
        out[f"fixed_{arm}_regret"] = med_r[a] / med_r.min() - 1

    idx = launch_resamples(L)
    bs, br = np.median(S[idx], axis=1), np.median(R[idx], axis=1)          # (B, 3)
    bchoice = bs.argmin(axis=1)
    bregret = br[np.arange(B), bchoice] / br.min(axis=1) - 1
    out["decision_regret_lo"], out["decision_regret_hi"] = percentile_interval(bregret)
    out["p_choice_changes"] = float(np.mean(bchoice != choice))

    lchoice = S.argmin(axis=1)
    lregret = R[np.arange(L), lchoice] / R.min(axis=1) - 1
    out["per_launch_regret_median"] = float(np.median(lregret))
    out["per_launch_regret_median_lo"], out["per_launch_regret_median_hi"] = percentile_interval(
        np.median(lregret[idx], axis=1))
    out["per_launch_regret_max"] = float(lregret.max())

    shifts = []
    all_pairs = []
    for k in range(1, L):
        other = np.roll(np.arange(L), -k)
        r = R[np.arange(L), R[other].argmin(axis=1)] / R.min(axis=1) - 1
        shifts.append(r.max())
        all_pairs.append(r)
    out["noise_floor_regret_median"] = float(np.median(np.concatenate(all_pairs))) if L > 1 else np.nan
    out["noise_floor_regret_max"] = float(np.mean(shifts)) if L > 1 else np.nan

    agree_min = int(np.sum(lchoice == R.argmin(axis=1)))
    agree_order = int(np.sum(np.all(np.argsort(S, axis=1) == np.argsort(R, axis=1), axis=1)))
    out["argmin_agree"] = agree_min / L
    out["argmin_agree_lo"], out["argmin_agree_hi"] = wilson(agree_min, L)
    out["order_agree"] = agree_order / L
    out["order_agree_lo"], out["order_agree_hi"] = wilson(agree_order, L)
    return out


def pairwise_rows(S, R):
    """Per arm pair: agreement per launch (ties in the reference count as agreement)."""
    rows, per_launch = [], {}
    for a, b in PAIRS:
        i, j = ARMS.index(a), ARMS.index(b)
        tied = np.maximum(R[:, i], R[:, j]) / np.minimum(R[:, i], R[:, j]) < 1 + TIE
        same = np.sign(S[:, i] - S[:, j]) == np.sign(R[:, i] - R[:, j])
        agree = same | tied
        per_launch[(a, b)] = agree
        L, untied = len(S), int((~tied).sum())
        k_untied = int(same[~tied].sum())
        rows.append({"pair": f"{a}-{b}", "launches": L, "reference_tied": int(tied.sum()),
                     "agree_with_ties": agree.mean(), **dict(zip(("agree_with_ties_lo", "agree_with_ties_hi"),
                                                               wilson(int(agree.sum()), L))),
                     "untied_launches": untied, "agree_untied": k_untied / untied if untied else np.nan,
                     **dict(zip(("agree_untied_lo", "agree_untied_hi"), wilson(k_untied, untied)))})
    return rows, per_launch


def prediction_difference(per_launch):
    """Bootstrap the pre-stated prediction: agreement on MPI-STANDARD minus the mean
    agreement on the two pairs involving LOCALITY (ties count as agreement)."""
    ms = per_launch[("MPI", "STANDARD")].astype(float)
    loc = (per_launch[("MPI", "LOCALITY")].astype(float) + per_launch[("STANDARD", "LOCALITY")]) / 2
    diff = ms - loc
    idx = launch_resamples(len(diff), seed_offset=1)
    return diff.mean(), *percentile_interval(diff[idx].mean(axis=1))


def compare(runs, stand_in, reference, drop_slow):
    points, pairs, prediction = [], [], []
    for nodes in NODES:
        for matrix in MATRICES:
            T = q_table(runs, nodes, matrix, [stand_in, reference], drop_slow=drop_slow)
            if len(T) < 5:
                continue
            S, R = T[stand_in][ARMS].to_numpy(), T[reference][ARMS].to_numpy()
            key = {"nodes": nodes, "matrix": matrix, "stand_in": stand_in, "reference": reference,
                   "slow_runs": "dropped" if drop_slow else "kept"}
            points.append({**key, **ordering_stats(S, R)})
            prows, per_launch = pairwise_rows(S, R)
            pairs += [{**key, **r} for r in prows]
            d, lo, hi = prediction_difference(per_launch)
            prediction.append({**key, "launches": len(S), "agree_MPI-STANDARD_minus_LOCALITY_pairs": d,
                               "lo": lo, "hi": hi})
    return points, pairs, prediction


def plot_regret(points, stand_in, reference, nodes):
    p = points[(points.stand_in == stand_in) & (points.reference == reference) & (points.nodes == nodes)
               & (points.slow_runs == "kept")].set_index("matrix").reindex(MATRICES).dropna(subset=["launches"])
    if p.empty:
        return
    x = np.arange(len(p))
    pct = 100
    fig, ax = new_fig(f"Cost of letting {HUMAN[stand_in]} pick the arm for {HUMAN[reference]}, {nodes} nodes",
                      "matrix", "slowdown vs. the best arm (%)", height=6.2,
                      caption=f"Regret is how much slower {HUMAN[reference]} runs on the arm {HUMAN[stand_in]} "
                              f"ranks fastest than on its own fastest arm; 0% means the pick was right. Black: the "
                              f"pick made once from medians over launches, with a 95% launch-bootstrap interval "
                              f"(the plan's headline setting). Blue: the pick made separately in each launch "
                              f"(square = median over launches, open triangle = worst launch). Red: noise floor, the "
                              f"application predicting itself from a different launch. Launches per matrix: "
                              f"{', '.join(str(int(v)) for v in p.launches)}. Slow runs kept.")
    err = np.vstack([p.decision_regret - p.decision_regret_lo, p.decision_regret_hi - p.decision_regret]) * pct
    ax.errorbar(x - 0.15, p.decision_regret * pct, yerr=np.clip(err, 0, None), fmt="o", color="black", capsize=3,
                label="decision regret (median-based choice), 95% launch bootstrap")
    err2 = np.vstack([p.per_launch_regret_median - p.per_launch_regret_median_lo,
                      p.per_launch_regret_median_hi - p.per_launch_regret_median]) * pct
    ax.errorbar(x + 0.05, p.per_launch_regret_median * pct, yerr=np.clip(err2, 0, None), fmt="s", color="#1f77b4",
                capsize=3, label="per-launch regret, median over launches")
    ax.scatter(x + 0.05, p.per_launch_regret_max * pct, marker="^", color="#1f77b4", facecolors="none",
               label="per-launch regret, max over launches")
    ax.scatter(x + 0.25, p.noise_floor_regret_median * pct, marker="_", s=200, color="#d62728",
               label="noise floor: reference predicting itself across launches (median)")
    ax.scatter(x + 0.25, p.noise_floor_regret_max * pct, marker="v", color="#d62728", facecolors="none",
               label="noise floor (expected max over launches)")
    ax.set_xticks(x, p.index, rotation=0)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save_fig(fig, f"regret_{stand_in}_vs_{reference}_{nodes}nodes.png")


def plot_pairwise(pairs, stand_in, reference):
    p = pairs[(pairs.stand_in == stand_in) & (pairs.reference == reference) & (pairs.slow_runs == "kept")]
    points = [(n, m) for n in NODES for m in MATRICES if ((p.nodes == n) & (p.matrix == m)).any()]
    fig, ax = new_fig(f"Does {HUMAN[stand_in]} order each pair of arms the way {HUMAN[reference]} does?",
                      "node count / matrix", "share of launches that agree", width=11,
                      caption=f"Each point is the share of launches in which the stand-in puts two arms in the same "
                              f"order as the application, with a 95% Wilson interval. When the application's two arms "
                              f"are within {TIE:.0%} of each other, either order counts as agreement. The plan "
                              f"predicted MPI-STANDARD (grey) would agree best. {p.launches.min()}-{p.launches.max()} launches per point; slow runs "
                              f"kept.")
    for off, (pair, color) in zip((-0.2, 0, 0.2), (("MPI-STANDARD", "#7f7f7f"), ("MPI-LOCALITY", "#2ca02c"),
                                                   ("STANDARD-LOCALITY", "#9467bd"))):
        s = p[p.pair == pair].set_index(["nodes", "matrix"]).reindex(points)
        x = np.arange(len(points)) + off
        ax.errorbar(x, s.agree_with_ties, yerr=np.clip(np.vstack([s.agree_with_ties - s.agree_with_ties_lo,
                                                                    s.agree_with_ties_hi - s.agree_with_ties]), 0, None),
                    fmt="o", ms=3.5, capsize=2, color=color, label=pair)
    ax.set_xticks(np.arange(len(points)), [f"{n}/{m}" for n, m in points], rotation=70, fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    save_fig(fig, f"pairwise_agreement_{stand_in}_vs_{reference}.png")


def plot_worst_case(points):
    p = points[(points.slow_runs == "kept") & points.reference.isin(PRIMARY)]
    combos = [(s, r) for r in PRIMARY for s in STAND_INS]
    fig, ax = new_fig("Worst-case cost of each stand-in's arm choice",
                      "stand-in -> application", "worst slowdown vs. the best arm (%)",
                      caption="Bar: the largest regret over the 24 combinations of matrix and node count when the "
                              "stand-in picks the arm for the application (100% = the application runs 2x slower "
                              "than on its best arm). Markers: the same quantity for always using one fixed arm. "
                              f"{p.launches.min()}-{p.launches.max()} launches per combination; slow runs kept.")
    x = np.arange(len(combos))
    worst = [p[(p.stand_in == s) & (p.reference == r)].decision_regret.max() * 100 for s, r in combos]
    ax.bar(x - 0.2, worst, width=0.4, color="#333333", label="stand-in's choice")
    for k, arm in enumerate(ARMS):
        fixed = [p[(p.stand_in == s) & (p.reference == r)][f"fixed_{arm}_regret"].max() * 100 for s, r in combos]
        ax.scatter(x + 0.2, fixed, marker="os^"[k], label=f"always {arm}", zorder=3)
    ax.set_xticks(x, [f"{KIND_LABEL[s]}\n-> {r}" for s, r in combos], fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(fontsize=8)
    save_fig(fig, "worst_case_regret.png")


def main():
    runs = load_runs()
    points, pairs, prediction = [], [], []
    for stand_in, reference, drop in itertools.product(STAND_INS, REFERENCES, (False, True)):
        a, b, c = compare(runs, stand_in, reference, drop)
        points += a
        pairs += b
        prediction += c
    points, pairs, prediction = pd.DataFrame(points), pd.DataFrame(pairs), pd.DataFrame(prediction)
    write_table(points, "h17_design_points.csv")
    write_table(pairs, "h17_pairwise.csv")
    write_table(prediction, "h17_prediction.csv")
    for stand_in, reference in itertools.product(STAND_INS, PRIMARY):
        for nodes in NODES:
            plot_regret(points, stand_in, reference, nodes)
        plot_pairwise(pairs, stand_in, reference)
    plot_worst_case(points)

    kept = points[points.slow_runs == "kept"]
    print(kept.groupby(["stand_in", "reference"]).agg(
        worst_decision_regret=("decision_regret", "max"), median_decision_regret=("decision_regret", "median"),
        argmin_agree=("argmin_agree", "mean"), worst_per_launch=("per_launch_regret_max", "max"),
        noise_floor_max=("noise_floor_regret_max", "max")).round(3).to_string())


if __name__ == "__main__":
    main()
