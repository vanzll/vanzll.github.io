"""Render cached, same-point force diagnostics without re-querying W&B."""

import argparse
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ns = "probe/force_conversion/"
    specifications = [
        ("26rilzmp", "NFT + EMA", None, ["guidance", "restore"]),
        ("adnxitig", "NFT: second update", 1, ["guidance", "restore"]),
        ("75pf0fok", "Fast: partial", None, ["guidance", "reference_mse"]),
    ]
    fields = ["output/individual_l2_mean", "parameter/individual_l2_mean",
              "parameter/aggregate_mean_l2"]
    colors = {"guidance": "#087F8C", "restore": "#CF526F", "reference_mse": "#756B98"}
    markers = {"guidance": "o", "restore": "s", "reference_mse": "D"}
    rows, sources = [], {}
    for rid, label, window, components in specifications:
        source = args.source / (rid + ".json")
        data = json.loads(source.read_text())
        sources[rid] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "commit": data["commit"], "state": data["state"],
                        "url": "https://wandb.ai/vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo/runs/" + rid}
        index = lambda component: {(r[ns + "context/rollout"], r[ns + "context/window"]): r
                                   for r in data["history"][component]}
        indexed = {component: index(component) for component in components}
        paired = set.intersection(*(set(values) for values in indexed.values()))
        paired = {key for key in paired if (window is None or key[1] == window)
                  and all(indexed[c][key][ns + c + "/" + fields[0]] > 0 for c in components)}
        for component in components:
            selected = [indexed[component][key] for key in sorted(paired)]
            means = [st.mean(r[ns + component + "/" + field] for r in selected)
                     for field in fields]
            rows.append(dict(run=rid, group=label, component=component,
                             windows=len(selected), states_per_window=selected[0][ns + component + "/sample_count"],
                             output_norm=means[0], individual_gradient_norm=means[1],
                             aggregate_gradient_norm=means[2], gain=means[1] / means[0],
                             retention=means[2] / means[1]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": "#243238", "axes.labelcolor": "#243238",
                         "xtick.color": "#69777D", "axes.edgecolor": "#C9D2D5",
                         "axes.linewidth": .7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    fig.subplots_adjust(left=.24, right=.96, top=.89, bottom=.24, wspace=.33)
    positions = [5.4, 4.6, 3.4, 2.6, 1.4, .6]
    for row, y in zip(rows, positions):
        c = row["component"]
        for axis, metric in zip(axes, ["gain", "retention"]):
            value = row[metric]
            axis.scatter(value, y, color=colors[c], marker=markers[c], s=34, zorder=3)
            axis.annotate(f"{value:.1f}" if metric == "gain" else f"{value:.1%}",
                          (value, y), xytext=(7, 0), textcoords="offset points",
                          va="center", fontsize=9)
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#E7ECEE", linewidth=.6)
        axis.set_axisbelow(True)
        axis.set(ylim=(0, 6), yticks=[5, 3, 1], yticklabels=[s[1] for s in specifications])
    axes[0].set(xscale="log", xlim=(2, 400), xticks=[3, 10, 30, 100, 300],
                xticklabels=["3", "10", "30", "100", "300"], xlabel="Mapping gain (log scale)")
    axes[1].set(xlim=(0, 1), xticks=[0, .25, .5, .75, 1],
                xticklabels=["0", "25%", "50%", "75%", "100%"], xlabel="Aggregation retention")
    legend = [Line2D([], [], color=colors[c], marker=markers[c], linestyle="None", markersize=5,
                     label=label) for c, label in [("guidance", "Guidance"),
                                                  ("restore", "Restoration"),
                                                  ("reference_mse", "Reference MSE (unweighted)")]]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .015), fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(sources=sources, statistics=rows, queried_on="2026-10-09",
                    aggregation="Ratio of window-mean norms; identical paired windows with nonzero output forces for both components.",
                    uncertainty="Single seed; no confidence interval or smoothing.",
                    limitations=["NFT has 80 sampled states/window; Fast has 24 and different noise positions.",
                                 "Cloud histories contain 19 EMA, 40 NFT-disjoint, 33 Fast windows; local coverage is 20, 40, 34.",
                                 "Reference MSE is diagnostic only; applied KL coefficient is zero.",
                                 "Runtime commit differs from supplied dfc49cd; runtime diff not yet verified."])
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
