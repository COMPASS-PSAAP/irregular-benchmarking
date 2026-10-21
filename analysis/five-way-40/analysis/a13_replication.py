"""Analysis 13 (paper): does each test's verdict survive a second set of launches?

Each method is judged against itself, so no method is the referee. For every arm pair of
every (nodes, kind, matrix), the 40 launches are split into two halves of 20, either at
random (500 splits) or chronologically (the first 20 launches to start against the last
20, the "rerun tomorrow" case). Each method gives a verdict on each half (A faster, B
faster, or none at 5%):
  t_ln (mean of d), t_q (mean of q_A - q_B), sign (median of d), wilcoxon (HL of d).
Reported per method: the share of half-verdicts that are significant, directional
replication (the second half gives the same significant verdict, given the first was
significant), reversals (the second half is significant the other way), and whether the
first half's 95% interval contains the second half's estimate (t interval / mean,
order-statistic interval / median).

Diagnostics for exchangeability: per comparison, Spearman correlation of d with launch
start time, with the run pair's mean position in its launch, and with the number of
concurrently running launches (BH within each covariate). Sensitivity: the percentile
interval of the median of d from a bootstrap over blocks of 5 consecutive launches,
against the ordinary launch bootstrap.

Writes tables/replication.csv, tables/exchangeability.csv, and figures
replication_<split>.png.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import SEED, new_fig, save_fig, write_table
from paired import (ALPHA, arm_comparisons, bh, hl_rows, jitter, key_of, launch_table, order_stat_ci, runs_with_stalls,
                    sign_p, t_ci, t_p, wilcoxon_p)
from a12_calibration import TEST_COLOR, TEST_LABEL, mad

R = 500
METHODS = ["t_ln", "t_q", "sign", "wilcoxon"]


def verdicts(d, x, idx, rng, s):
    """idx: (splits, 20) launch indices. Returns {method: (verdict, estimate, lo, hi)}."""
    D, Xq = d[idx], x[idx]
    Dj = jitter(D, rng, s)
    out = {}
    m = D.mean(axis=1)
    lo, hi = t_ci(D)
    out["t_ln"] = (np.where(t_p(D) < ALPHA, np.sign(m), 0), m, lo, hi)
    mq = Xq.mean(axis=1)
    lo, hi = t_ci(Xq)
    out["t_q"] = (np.where(t_p(Xq) < ALPHA, np.sign(mq), 0), mq, lo, hi)
    med = np.median(D, axis=1)
    lo, hi = order_stat_ci(D)
    out["sign"] = (np.where(sign_p(Dj) < ALPHA, np.sign(med), 0), med, lo, hi)
    hl = hl_rows(D)
    out["wilcoxon"] = (np.where(wilcoxon_p(Dj) < ALPHA, np.sign(hl), 0), hl, None, None)
    return out


def main():
    runs = runs_with_stalls()
    L = launch_table(runs).set_index("launch")
    rng = np.random.default_rng(SEED + 13)
    rows, ex = [], []
    for c in arm_comparisons(runs):
        d, x = c["d"], c["qA"] - c["qB"]
        n = len(d)
        s = mad(d)
        start = L.loc[c["launch"], "start_h"].to_numpy()
        order = np.argsort(start)
        perms = np.argsort(rng.uniform(size=(R, n)), axis=1)
        splits = {"random": (perms[:, :n // 2], perms[:, n // 2:]),
                  "chronological": (order[None, :n // 2], order[None, n // 2:])}
        for split, (ia, ib) in splits.items():
            va, vb = verdicts(d, x, ia, rng, s), verdicts(d, x, ib, rng, s)
            for m in METHODS:
                a, ea, la, ha = va[m]
                b, eb, lb, hb = vb[m]
                # Both directions of the split count (A predicts B and B predicts A).
                first = np.concatenate([a, b])
                second = np.concatenate([b, a])
                sig = first != 0
                row = {**key_of(c), "split": split, "method": m, "halves": len(first),
                       "sig_rate": sig.mean(),
                       "replicated": (second[sig] == first[sig]).sum(), "reversed": (second[sig] == -first[sig]).sum(),
                       "significant": int(sig.sum())}
                if la is not None:
                    contain = np.concatenate([(la <= eb) & (ha >= eb), (lb <= ea) & (hb >= ea)])
                    row["contains_other_estimate"] = contain.mean()
                rows.append(row)
        launch_ids = c["launch"]
        ex_row = {**key_of(c)}
        for name, cov in (("start_h", start), ("position", c["pos"]),
                          ("concurrency", L.loc[launch_ids, "concurrency"].to_numpy())):
            rho, p = stats.spearmanr(cov, d)
            ex_row[f"rho_{name}"], ex_row[f"p_{name}"] = rho, p
        blocks = order.reshape(8, 5)
        bi = blocks[rng.integers(0, 8, (2000, 8))].reshape(2000, -1)
        ii = rng.integers(0, n, (2000, n))
        wb = np.diff(np.quantile(np.median(d[bi], axis=1), [0.025, 0.975]))[0]
        wi = np.diff(np.quantile(np.median(d[ii], axis=1), [0.025, 0.975]))[0]
        ex_row["block_width"], ex_row["iid_width"] = wb, wi
        ex.append(ex_row)
    rep, ex = pd.DataFrame(rows), pd.DataFrame(ex)
    for name in ("start_h", "position", "concurrency"):
        ex[f"q_{name}"] = bh(ex[f"p_{name}"])
    ex["block_over_iid"] = ex.block_width / ex.iid_width.replace(0, np.nan)
    write_table(rep, "replication.csv")
    write_table(ex, "exchangeability.csv")

    summary = rep.groupby(["split", "method"]).agg(sig_rate=("sig_rate", "mean"), significant=("significant", "sum"),
                                                   replicated=("replicated", "sum"), reversed=("reversed", "sum"),
                                                   contains=("contains_other_estimate", "mean"))
    summary["replication"] = summary.replicated / summary.significant
    summary["reversal"] = summary.reversed / summary.significant
    print(summary.round(4).to_string())
    print(rep.groupby(["split", "regime", "method"]).apply(
        lambda g: pd.Series({"sig_rate": g.sig_rate.mean(), "replication": g.replicated.sum() / max(g.significant.sum(), 1),
                             "reversal": g.reversed.sum() / max(g.significant.sum(), 1)}),
        include_groups=False).round(4).to_string())
    for name in ("start_h", "position", "concurrency"):
        q = ex[f"q_{name}"]
        print(f"{name}: {int((q < 0.05).sum())} of {len(ex)} comparisons correlate with d at BH q < 0.05; "
              f"by node count {ex[q < 0.05].groupby('nodes').size().to_dict()}")
    print("block / iid bootstrap width of the median: median %.2f, p90 %.2f" %
          (ex.block_over_iid.median(), ex.block_over_iid.quantile(0.9)))

    for split in ("random", "chronological"):
        s = summary.loc[split]
        fig, ax = new_fig(f"Do significant verdicts hold on the other half of the launches? ({split} split)",
                          "test", "share",
                          caption=f"All 504 arm-pair comparisons, 40 launches split into halves of 20 "
                                  f"({'500 random splits' if split == 'random' else 'first 20 launches to start against the last 20'}"
                                  f"; both directions). Left bar: share of half-verdicts that are significant at 5%. "
                                  f"Middle: of those, the share where the other half is significant the same way. "
                                  f"Right (red): the share where the other half is significant the opposite way.")
        x = np.arange(len(METHODS))
        ax.bar(x - 0.25, s.loc[METHODS, "sig_rate"], 0.25, color="#bbbbbb", label="significant")
        ax.bar(x, s.loc[METHODS, "replication"], 0.25, color=[TEST_COLOR[m] for m in METHODS], label="replicated")
        ax.bar(x + 0.25, s.loc[METHODS, "reversal"], 0.25, color="#d62728", label="reversed")
        for xi, v in zip(x + 0.25, s.loc[METHODS, "reversal"]):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x, [TEST_LABEL[m] for m in METHODS], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        save_fig(fig, f"replication_{split}.png")


if __name__ == "__main__":
    main()
