"""Analysis 7: order and drift effects, and where the slow runs fall.

Per node count, a linear mixed model (statsmodels MixedLM, REML):
  ln q ~ cell (pattern x arm) + position/100 + previous run's kind + (1 | launch)
fitted with slow runs kept and dropped. RESEARCH-PLAN section 5.4 requires the position
coefficient. The previous kind is "first" for the first run of a launch.

Slow runs (exploratory, not pre-stated): for MPI/STANDARD runs, the slow fraction by
previous kind and by position decile; per launch, a stall score (slow fraction of its
MPI/STANDARD runs) against launch start time (Spearman) and against each node that
appears in >= 3 launches (Mann-Whitney, Bonferroni over nodes).

Writes tables/order_model.csv, tables/slow_by_predecessor.csv,
tables/slow_by_position.csv, tables/launch_stall_score.csv, tables/slow_by_node.csv, and
figures position_effect_<N>nodes.png and launch_stall_score_vs_start.png.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import NODES, RESULTS, load_runs, new_fig, save_fig, write_table


def with_order(runs):
    runs = runs.sort_values(["launch", "position"]).copy()
    runs["prev_kind"] = runs.groupby("launch")["kind"].shift().fillna("first")
    runs["cell"] = runs.pattern + ":" + runs.arm
    runs["pos100"] = runs.position / 100
    runs["lnq"] = np.log(runs.q_ms)
    return runs


def fit(sub):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("lnq ~ C(cell) + pos100 + C(prev_kind, Treatment('first'))", sub,
                        groups=sub["launch"]).fit(reml=True)
    ci = m.conf_int()
    out = {"rows": len(sub), "launches": sub.launch.nunique(), "converged": bool(m.converged),
           "pos100_coef": m.params["pos100"], "pos100_lo": ci.loc["pos100", 0], "pos100_hi": ci.loc["pos100", 1],
           "pos100_p": m.pvalues["pos100"], "launch_var": float(m.cov_re.iloc[0, 0]), "resid_var": m.scale}
    for name in m.params.index:
        if name.startswith("C(prev_kind"):
            k = name.split("[T.")[1].rstrip("]")
            out[f"prev_{k}_coef"] = m.params[name]
            out[f"prev_{k}_lo"], out[f"prev_{k}_hi"] = ci.loc[name, 0], ci.loc[name, 1]
    return out


def main():
    runs = with_order(load_runs())
    rows = []
    for nodes in NODES:
        for drop in (False, True):
            sub = runs[runs.nodes == nodes]
            if drop:
                sub = sub[~sub.slow]
            rows.append({"nodes": nodes, "slow_runs": "dropped" if drop else "kept", **fit(sub)})
    model = pd.DataFrame(rows)
    write_table(model, "order_model.csv")
    print(model[["nodes", "slow_runs", "rows", "converged", "pos100_coef", "pos100_lo", "pos100_hi",
                 "launch_var", "resid_var"]].round(4).to_string(index=False))

    fig, ax = new_fig("Do runs get slower later in a launch?",
                      "node count", "change in log time per 100 runs", width=7,
                      caption="Estimated change in log time per iteration for every 100 positions later in the "
                              "launch's run order (0.1 is about 10% slower), from a mixed model that controls for "
                              "pattern, arm, the previous run, and the launch. Bars: 95% intervals. Black: all runs; "
                              f"red: runs over 5x their usual time removed. {model.launches.min()}-{model.launches.max()} launches per "
                              "node count.")
    for off, label, color in ((-0.1, "kept", "black"), (0.1, "dropped", "#d62728")):
        m = model[model.slow_runs == label]
        x = np.arange(len(NODES)) + off
        ax.errorbar(x, m.pos100_coef, yerr=np.vstack([m.pos100_coef - m.pos100_lo, m.pos100_hi - m.pos100_coef]),
                    fmt="o", capsize=3, color=color, label=f"slow runs {label}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(np.arange(len(NODES)), [str(n) for n in NODES])
    ax.legend(fontsize=8)
    save_fig(fig, "position_effect_by_nodes.png")

    p2p = runs[runs.arm.isin(["MPI", "STANDARD"])]
    pred = p2p.groupby(["nodes", "prev_kind"]).slow.agg(["mean", "sum", "size"]).reset_index()
    write_table(pred, "slow_by_predecessor.csv")
    p2p = p2p.assign(pos_decile=pd.cut(p2p.position, 10, labels=False))
    pos = p2p.groupby(["nodes", "pos_decile"]).slow.agg(["mean", "sum", "size"]).reset_index()
    write_table(pos, "slow_by_position.csv")

    launches = pd.read_csv(RESULTS / "launches.csv")
    score = p2p.groupby(["nodes", "launch"]).agg(stall_score=("slow", "mean"),
                                                start=("start", "min")).reset_index()
    score = score.merge(launches[["launch", "node_list"]], on="launch")
    score["start_h"] = (pd.to_datetime(score.start, utc=True) - pd.to_datetime(score.start, utc=True).min()
                        ).dt.total_seconds() / 3600
    write_table(score, "launch_stall_score.csv")
    fig, ax = new_fig("Stall rate over the campaign", "hours since the first launch started",
                      "share of MPI/STANDARD runs that were slow",
                      caption="One point per launch (159). Vertical: the share of that launch's MPI and STANDARD "
                              "runs that took over 5x their usual time. Horizontal: when the launch started. The "
                              "legend gives the Spearman correlation per node count; none is significant, so the "
                              "stalls do not follow time of day over this roughly one-day campaign.")
    node_rows = []
    for k, nodes in enumerate(NODES):
        s = score[score.nodes == nodes]
        rho, p = stats.spearmanr(s.start_h, s.stall_score)
        ax.scatter(s.start_h, s.stall_score, s=14, marker="osD^"[k], label=f"{nodes} nodes (Spearman {rho:.2f}, p={p:.2g})")
        members = {}
        for launch, nl in zip(s.launch, s.node_list):
            for nd in nl.split():
                members.setdefault(nd, set()).add(launch)
        tested = {nd: l for nd, l in members.items() if len(l) >= 3}
        for nd, ls in tested.items():
            inside = s[s.launch.isin(ls)].stall_score
            outside = s[~s.launch.isin(ls)].stall_score
            u, pu = stats.mannwhitneyu(inside, outside, alternative="two-sided")
            node_rows.append({"nodes": nodes, "node": nd, "launches_with": len(ls), "mean_with": inside.mean(),
                              "mean_without": outside.mean(), "p": pu, "p_bonferroni": min(1, pu * len(tested))})
    ax.legend(fontsize=7)
    save_fig(fig, "launch_stall_score_vs_start.png")
    node_table = pd.DataFrame(node_rows).sort_values("p")
    write_table(node_table, "slow_by_node.csv")
    print("\nslow fraction by previous kind (MPI/STANDARD runs):")
    print(pred.pivot(index="prev_kind", columns="nodes", values="mean").round(3).to_string())
    print("\nnodes tested:", len(node_table), " smallest Bonferroni p:", node_table.p_bonferroni.min().round(3))
    print(node_table.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
