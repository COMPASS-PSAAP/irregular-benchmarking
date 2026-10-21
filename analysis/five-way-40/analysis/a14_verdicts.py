"""Analysis 14 (paper): which arm is faster, and does each stand-in track the run above it?

Part 1, verdicts on all 40 launches for every arm pair of every (nodes, kind, matrix):
  sign test on d (primary), paired t on ln q, paired t on q, Wilcoxon signed-rank; BH
  within each kind (72 comparisons). Primary verdict from the sign test and the 95%
  order-statistic interval of the median ratio:
    faster       BH q < 0.05 (flagged "under 5%" when the interval lies inside +/-5%)
    equivalent   not significant and the interval lies inside +/-5%
    inconclusive otherwise
  Where the t-test on ln q and the sign test disagree, the cause is labeled: "estimand"
  (mean and median of d point different ways), else "stalls" (the comparison has stalled
  runs), else "power". Best arm: the arm significantly faster than both others, if any.

Part 2, tracking: stand-in X tracks reference Y on arm pair (A, B) when the per-launch
difference in differences e = ln(q_XA/q_XB) - ln(q_YA/q_YB) has a 90% order-statistic
interval of its median inside +/-10% (a TOST at 5%). Also the sign test on e (does X shift
the pair's ratio?) and the share of launches in which X and Y order the pair the same
way (Wilson interval).

Writes tables/verdicts.csv, verdict_disagreements.csv, best_arm_tests.csv, tracking.csv,
and figures tracking_<x>_vs_<y>.png, tracking_summary.png.
"""
import numpy as np
import pandas as pd

from common import ARMS, MATRICES, NODES, SEED, new_fig, save_fig, wilson, write_table
from paired import (ARM_PAIRS, KINDS, arm_comparisons, bh, hodges_lehmann, jitter, key_of, order_stat_ci,
                    runs_with_stalls, sign_p, t_p, wide, wilcoxon_p)
from a12_calibration import mad

EQ5 = np.log(1.05)
TRACK = np.log(1.10)
TRACK_PAIRS = [("coarse", "fine"), ("fine", "pm"), ("pm", "abcgspmv"), ("abcgspmv", "abcgcg"), ("pm", "abcgcg"),
               ("fine", "abcgcg"), ("coarse", "abcgcg")]
SHORT = {"coarse": "coarse replay", "fine": "fine replay", "pm": ".pm replay", "abcgspmv": "ABCG SpMV",
         "abcgcg": "ABCG CG", "abcgspmvnp": "ABCG SpMV (np)", "abcgcgnp": "ABCG CG (np)"}


def main():
    runs = runs_with_stalls()
    rng = np.random.default_rng(SEED + 14)
    rows = []
    for c in arm_comparisons(runs):
        d, x = c["d"], c["qA"] - c["qB"]
        lo, hi = (v[0] for v in order_stat_ci(d[None]))
        dj = jitter(d[None], rng, mad(d))
        rows.append({**key_of(c), "median_lr": np.median(d), "lo": lo, "hi": hi, "mean_lr": d.mean(),
                     "mean_q_diff": x.mean(), "hl_lr": hodges_lehmann(d),
                     "p_sign": sign_p(dj)[0], "p_t_ln": t_p(d[None])[0], "p_t_q": t_p(x[None])[0],
                     "p_wilcoxon": wilcoxon_p(dj)[0], "wins_A": int((d < 0).sum())})
    v = pd.DataFrame(rows)
    for m in ("sign", "t_ln", "t_q", "wilcoxon"):
        v[f"q_{m}"] = v.groupby("kind")[f"p_{m}"].transform(lambda p: pd.Series(bh(p.to_numpy()), index=p.index))
    est = {"sign": "median_lr", "t_ln": "mean_lr", "t_q": "mean_q_diff", "wilcoxon": "hl_lr"}
    for m, e in est.items():
        v[f"v_{m}"] = np.where(v[f"q_{m}"] < 0.05, np.sign(v[e]), 0).astype(int)
    inside = (v.lo > -EQ5) & (v.hi < EQ5)
    v["verdict"] = np.where(v.v_sign != 0, np.where(inside, "faster, under 5%", "faster"),
                            np.where(inside, "equivalent", "inconclusive"))
    v["faster_arm"] = np.where(v.v_sign < 0, v.A, np.where(v.v_sign > 0, v.B, ""))
    write_table(v, "verdicts.csv")

    dis = v[v.v_t_ln != v.v_sign].copy()
    dis["cause"] = np.where(np.sign(dis.mean_lr) != np.sign(dis.median_lr), "estimand",
                            np.where(dis.regime != "none", "stalls", "power"))
    dis["which_rejects"] = np.where(dis.v_sign == 0, "t only", np.where(dis.v_t_ln == 0, "sign only", "opposite"))
    write_table(dis, "verdict_disagreements.csv")

    print(v.groupby("kind").verdict.value_counts().unstack(fill_value=0).to_string())
    print("\nt on ln q vs sign test (rows t, columns sign):")
    print(pd.crosstab(v.v_t_ln, v.v_sign).to_string())
    print("\nt on q vs sign test:")
    print(pd.crosstab(v.v_t_q, v.v_sign).to_string())
    print("\nWilcoxon vs sign test:")
    print(pd.crosstab(v.v_wilcoxon, v.v_sign).to_string())
    print("\nt-ln vs sign disagreements by cause and which rejects:")
    print(pd.crosstab(dis.cause, dis.which_rejects).to_string())

    best = []
    for (nodes, kind, matrix), g in v.groupby(["nodes", "kind", "matrix"]):
        row = {"nodes": nodes, "kind": kind, "matrix": matrix}
        for m in ("sign", "t_ln", "t_q"):
            beats = {a: 0 for a in ARMS}
            beaten = {a: False for a in ARMS}
            for _, r in g.iterrows():
                if r[f"v_{m}"] < 0:
                    beats[r.A] += 1
                    beaten[r.B] = True
                elif r[f"v_{m}"] > 0:
                    beats[r.B] += 1
                    beaten[r.A] = True
            winner = [a for a in ARMS if beats[a] == 2]
            row[f"best_{m}"] = winner[0] if winner else "/".join(a for a in ARMS if not beaten[a])
        best.append(row)
    best = pd.DataFrame(best)
    best["sign_vs_t_ln_differ"] = best.best_sign != best.best_t_ln
    write_table(best, "best_arm_tests.csv")
    print("\nbest arm differs between sign test and t on ln q:", int(best.sign_vs_t_ln_differ.sum()), "of", len(best))
    print(best[best.sign_vs_t_ln_differ].to_string(index=False))

    track = tracking(runs)
    write_table(track, "tracking.csv")
    summary = track.groupby(["stand_in", "reference"], sort=False).agg(
        points=("tracks", "size"), tracks=("tracks", "sum"), shifted=("shift_sig", "sum"),
        order_agree_median=("order_agree", "median"), order_agree_min=("order_agree", "min"),
        verdict_agree=("verdict_agree", "sum"))
    print("\ntracking (per node count x matrix x arm pair):")
    print(summary.to_string())
    print(track.groupby(["stand_in", "reference", "A", "B"], sort=False).tracks.sum().unstack(["A", "B"]).to_string())
    figures(track, summary)


def tracking(runs):
    rows = []
    for x, y in TRACK_PAIRS:
        for nodes in NODES:
            for matrix in MATRICES:
                q = wide(runs, nodes, [x, y], matrix, "q_ms")
                for a, b in ARM_PAIRS:
                    dx = np.log(q[(x, a)] / q[(x, b)]).to_numpy()
                    dy = np.log(q[(y, a)] / q[(y, b)]).to_numpy()
                    e = dx - dy
                    lo90, hi90 = (v[0] for v in order_stat_ci(e[None], level=0.90))
                    agree = int((np.sign(dx) == np.sign(dy)).sum())
                    wl, wh = wilson(agree, len(e))
                    rx = np.random.default_rng(SEED + 141)
                    px = sign_p(jitter(dx[None], rx, mad(dx)))[0]
                    py = sign_p(jitter(dy[None], rx, mad(dy)))[0]
                    vx = int(np.sign(np.median(dx))) if px < 0.05 else 0
                    vy = int(np.sign(np.median(dy))) if py < 0.05 else 0
                    rows.append({"stand_in": x, "reference": y, "nodes": nodes, "matrix": matrix, "A": a, "B": b,
                                 "median_e": np.median(e), "lo90": lo90, "hi90": hi90,
                                 "tracks": bool(lo90 > -TRACK and hi90 < TRACK),
                                 "p_shift": sign_p(jitter(e[None], rx, mad(e)))[0],
                                 "order_agree": agree / len(e), "order_lo": wl, "order_hi": wh,
                                 "verdict_x": vx, "verdict_y": vy, "verdict_agree": vx == vy})
    t = pd.DataFrame(rows)
    t["q_shift"] = t.groupby(["stand_in", "reference"]).p_shift.transform(
        lambda p: pd.Series(bh(p.to_numpy()), index=p.index))
    t["shift_sig"] = t.q_shift < 0.05
    return t


def figures(track, summary):
    for (x, y), g in track.groupby(["stand_in", "reference"], sort=False):
        fig, ax = new_fig(f"Does the {SHORT[x]} reproduce the {SHORT[y]}'s arm ratios?",
                          "node count / matrix", "ratio of arm ratios (stand-in / reference)", width=11, height=5.0,
                          caption=f"For each arm pair, the per-launch ratio (time A / time B) of the {SHORT[x]} divided "
                                  f"by the same ratio for the {SHORT[y]}; 1 means the stand-in reproduces how much "
                                  f"faster one arm is than the other. Points are medians over 40 launches with 90% "
                                  f"order-statistic intervals; the grey band is +/-10%, and an interval inside it is "
                                  f"a passed equivalence test at 5%.")
        pts = [(n, m) for n in NODES for m in MATRICES]
        colors = {("MPI", "STANDARD"): "#7f7f7f", ("MPI", "LOCALITY"): "#1f77b4", ("STANDARD", "LOCALITY"): "#ff7f0e"}
        for j, (a, b) in enumerate(ARM_PAIRS):
            s = g[(g.A == a) & (g.B == b)].set_index(["nodes", "matrix"]).reindex(pts)
            xs = np.arange(len(pts)) + (j - 1) * 0.22
            med = np.exp(s.median_e)
            ax.errorbar(xs, med, yerr=np.vstack([med - np.exp(s.lo90), np.exp(s.hi90) - med]), fmt="o", ms=3,
                        capsize=2, color=colors[(a, b)], label=f"{a} / {b}")
        ax.axhspan(1 / 1.1, 1.1, color="grey", alpha=0.15)
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(pts)), [f"{n}n {m}" for n, m in pts], rotation=90, fontsize=7)
        ax.legend(fontsize=8)
        save_fig(fig, f"tracking_{x}_vs_{y}.png")

    fig, ax = new_fig("How often each stand-in reproduces the arm ratios of the run above it",
                      "stand-in vs. reference", "share of the 72 node count x matrix x arm-pair points",
                      width=9, height=4.8,
                      caption="Equivalent: the 90% interval of the median ratio of arm ratios lies inside +/-10%. "
                              "Shifted: a sign test (BH within the stand-in pair) finds the stand-in's arm ratio "
                              "systematically different from the reference's. Same verdict: the sign tests on each "
                              "kind's own arm pair reach the same conclusion (A faster, B faster, or neither).")
    xs = np.arange(len(summary))
    ax.bar(xs - 0.25, summary.tracks / summary.points, 0.25, color="#2ca02c", label="equivalent within 10%")
    ax.bar(xs, summary.shifted / summary.points, 0.25, color="#d62728", label="ratio shifted (sign test)")
    ax.bar(xs + 0.25, summary.verdict_agree / summary.points, 0.25, color="#1f77b4", label="same verdict")
    ax.set_xticks(xs, [f"{SHORT[x]}\nvs {SHORT[y]}" for x, y in summary.index], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    save_fig(fig, "tracking_summary.png")


if __name__ == "__main__":
    main()
