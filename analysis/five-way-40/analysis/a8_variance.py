"""Analysis 8: how much of the spread in ln q is shared by a whole launch?

Per node count x kind (and per node count over all kinds): ln q ~ cell (pattern x arm)
+ (1 | launch), MixedLM REML. launch_var is the variance of a shift every run in a
launch shares; resid_var is each run's own deviation from its cell mean and its
launch's shift. Each cell runs once per launch, so resid_var is not repeat noise
within a launch: it mixes per-run noise with launch x cell interaction, which this
design cannot separate. ICC = launch_var / (launch_var + resid_var). Slow runs kept and
dropped. Writes tables/variance.csv and figure variance_share_<N>nodes.png.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import KIND_LABEL, NODES, load_runs, new_fig, save_fig, write_table

KINDS = ["pm", "fine", "coarse", "abcgspmv", "abcgspmvnp", "abcgcg", "abcgcgnp"]


def fit(sub):
    sub = sub.assign(cell=sub.pattern + ":" + sub.arm, lnq=np.log(sub.q_ms))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("lnq ~ C(cell)", sub, groups=sub["launch"]).fit(reml=True)
    lv, rv = float(m.cov_re.iloc[0, 0]), float(m.scale)
    return {"rows": len(sub), "launches": sub.launch.nunique(), "converged": bool(m.converged),
            "launch_var": lv, "resid_var": rv, "icc": lv / (lv + rv),
            "launch_sd_pct": 100 * (np.exp(np.sqrt(lv)) - 1), "resid_sd_pct": 100 * (np.exp(np.sqrt(rv)) - 1)}


def main():
    runs = load_runs()
    rows = []
    for nodes in NODES:
        for drop in (False, True):
            base = runs[runs.nodes == nodes]
            if drop:
                base = base[~base.slow]
            label = "dropped" if drop else "kept"
            rows.append({"nodes": nodes, "kind": "all", "slow_runs": label, **fit(base)})
            for kind in KINDS:
                rows.append({"nodes": nodes, "kind": kind, "slow_runs": label, **fit(base[base.kind == kind])})
    table = pd.DataFrame(rows)
    write_table(table, "variance.csv")

    for nodes in NODES:
        t = table[(table.nodes == nodes)]
        fig, ax = new_fig(f"How much variation a whole launch shares, {nodes} nodes",
                          "kind of run", "variance of log time",
                          caption=f"Bar height is the variance of log time per iteration after removing each "
                                  f"pattern-and-arm average. Blue: a shift shared by every run in a launch. Orange: "
                                  f"each run's own deviation. In each pair, the left bar uses all runs and the faded "
                                  f"right bar drops runs over 5x their usual time. {t.launches.min()}-"
                                  f"{t.launches.max()} launches.")
        order = ["all"] + KINDS
        x = np.arange(len(order))
        for off, label, alpha in ((-0.2, "kept", 1.0), (0.2, "dropped", 0.5)):
            s = t[t.slow_runs == label].set_index("kind").reindex(order)
            ax.bar(x + off, s.launch_var, width=0.38, color="#1f77b4", alpha=alpha,
                   label=f"launch (shared), slow {label}")
            ax.bar(x + off, s.resid_var, width=0.38, bottom=s.launch_var, color="#ff7f0e", alpha=alpha,
                   label=f"residual, slow {label}")
        ax.set_xticks(x, ["all"] + [KIND_LABEL[k] for k in KINDS], fontsize=8)
        ax.legend(fontsize=7)
        save_fig(fig, f"variance_share_{nodes}nodes.png")
    print(table[["nodes", "kind", "slow_runs", "converged", "launch_var", "resid_var", "icc",
                 "launch_sd_pct", "resid_sd_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
