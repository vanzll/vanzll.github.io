"""Render three force-diagnostic figures from frozen local W&B snapshots only."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import signal
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

PROJECT = "vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo"
FPA = ["output/individual_l2_mean", "parameter/individual_l2_mean",
       "parameter/aggregate_mean_l2"]
COLORS = {"guidance": "#087F8C", "restore": "#CF526F", "reference_mse": "#756B98"}
MARKERS = {"guidance": "o", "restore": "s", "reference_mse": "D"}
SEEDS = [314159, 271828, 161803]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def public_metadata(data, repo, public_output):
    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str) and (re.search(r"/(Users|m2v_intern2?|root|private|tmp)/", value)
                                       or value.startswith("/")):
            return "[private path omitted]"
        return value
    result = scrub(data)
    for rid, source in data["sources"].items():
        path = Path(source["path"])
        result["sources"][rid]["path"] = str(path.relative_to(repo))
        result["sources"][rid].pop("runtime_metadata", None)
        result["sources"][rid]["summary"] = {
            key: value for key, value in result["sources"][rid].get("summary", {}).items()
            if key in source["queried_keys"] or key == "_step"}
    result["renderer"] = str(Path(data["renderer"]).relative_to(repo))
    result["output_base"] = str((public_output / data["figure"]).relative_to(repo))
    assert not re.search(r"/(Users|m2v_intern2?|root|private|tmp)/", json.dumps(result))
    return result


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_info(path, data):
    return dict(path=str(path.resolve()), sha256=digest(path),
                commit=data.get("commit"), state=data["state"],
                url=data.get("url", f"https://wandb.ai/{PROJECT}/runs/{data['id']}"),
                fetched_at=data.get("fetched_at"), cutoff_step=data.get("cutoff_step"),
                config=data.get("config"), summary=data.get("summary"),
                runtime_metadata=data.get("runtime_metadata"),
                queried_keys=sorted({key for entries in data.get("history", {}).values()
                                     for row in entries for key in row if key != "_step"}))


def load_new(root, rid):
    path = root / f"force_mechanisms_source_{rid}.json"
    data = json.loads(path.read_text())
    merged = {}
    for key, entries in data["history"].items():
        for entry in entries:
            step = int(entry["_step"])
            assert key not in merged.setdefault(step, {}), (rid, key, step)
            merged[step][key] = entry[key]
    return data, [(step, merged[step]) for step in sorted(merged)], source_info(path, data)


def read_context(rid, step, row, prefix):
    if "insight_followup" in prefix:
        rollout = row.get(prefix + "context/epoch")
        microbatch = row.get(prefix + "context/microbatch")
        if rollout is None or microbatch is None:
            raise ValueError(f"Missing ART/Naive context: {rid}/{step}")
        assert microbatch in (0, 8), (rid, step, microbatch)
        window = int(microbatch) // 8
    else:
        rollout = row.get(prefix + "context/rollout")
        window = row.get(prefix + "context/window")
        microbatch = None
        if rollout is None or window is None:
            raise ValueError(f"Missing NFT context: {rid}/{step}")
    return dict(logged_step=step, rollout=int(rollout), window=int(window), microbatch=microbatch,
                optimizer_step=row.get(prefix + "context/optimizer_step"))


def read_component(rid, step, row, prefix, component):
    stem = prefix + component + "/"
    values = [row.get(stem + f) for f in FPA]
    if not all(finite(v) for v in values):
        return None
    context = read_context(rid, step, row, prefix)
    valid = bool(row.get(prefix + "valid", False))
    gain_valid = valid and values[0] > 0 and bool(row.get(stem + "gain_valid", True))
    retention_valid = (valid and values[0] > 0 and values[1] > 0
                       and bool(row.get(stem + "retention_valid", True)))
    equal_count = row.get(stem + "joint/equal_state_gain/count")
    equal = row.get(stem + "joint/equal_state_gain/mean")
    equal_valid = (gain_valid and finite(equal_count) and equal_count > 0
                   and bool(row.get(stem + "joint/equal_state_gain/valid", False))
                   and finite(equal))
    return dict(run=rid, source_url=f"https://wandb.ai/{PROJECT}/runs/{rid}",
                **context, component=component,
                training_seed=42, sample_count=row.get(stem + "sample_count"),
                valid=valid, gain_valid=gain_valid, retention_valid=retention_valid,
                F=values[0], P=values[1], A=values[2],
                gain=values[1] / values[0] if gain_valid else None,
                retention=values[2] / values[1] if retention_valid else None,
                equal_state_gain=equal if equal_valid else None,
                equal_state_valid_count=equal_count,
                zero_force_count=row.get(stem + "zero_force_count"))


def aggregate(rows, group, component):
    assert rows and all(r["valid"] and r["F"] > 0 and r["P"] > 0 for r in rows)
    means = {k: st.mean(r[k] for r in rows) for k in ("F", "P", "A")}
    counts = sorted({r["sample_count"] for r in rows if r["sample_count"] is not None})
    return dict(run=rows[0]["run"], group=group, component=component,
                windows=len({(r["rollout"], r["window"]) for r in rows}),
                observations=len(rows), sample_counts=json.dumps(counts), training_seed=42,
                **means, gain=means["P"] / means["F"], retention=means["A"] / means["P"])


def base_metadata(name, sources):
    return dict(figure=name, sources=sources,
                renderer=str(Path(__file__).resolve()), renderer_sha256=digest(Path(__file__)),
                uncertainty="One training seed (42); sampled states are not independent training seeds; no CI.",
                force_definition="F = mean_i ||f_i||; P = mean_i ||J_i^T f_i||; A = ||mean_i J_i^T f_i||, raw LoRA VJP before Adam/clip.",
                smoothing="None. No interpolation, forward-fill or imputation.",
                runtime_diff_verified=False,
                limitations=["Algorithms use their natural state/noise populations, not a common Jacobian.",
                             "Runtime commits are metadata evidence; full runtime diff has not been audited."])


def finish(fig, root, name, rows, metadata):
    out = root / name
    root.mkdir(parents=True, exist_ok=True)
    table(out.with_suffix(".csv"), rows)
    metadata["dimensions_inches"] = list(fig.get_size_inches())
    metadata["png_dpi"] = 600
    metadata["matplotlib_version"] = matplotlib.__version__
    metadata["preview_dpi"] = 150
    metadata["output_base"] = str(out.resolve())
    metadata["csv_sha256"] = digest(out.with_suffix(".csv"))
    fig.savefig(out.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out.with_name(name + "_preview").with_suffix(".png"), dpi=150, facecolor="white")
    dump(out.with_suffix(".json"), metadata)
    plt.close(fig)


def polish(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E7ECEE", linewidth=.6)
    axis.set_axisbelow(True)


def dot_axes(groups, height):
    fig, axes = plt.subplots(1, 2, figsize=(8, height), sharey=True)
    fig.subplots_adjust(left=.235, right=.975, top=.91, bottom=.25, wspace=.32)
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=10)
        axis.grid(axis="x", color="#E7ECEE", linewidth=.6)
        axis.set_axisbelow(True)
        axis.set(ylim=(-.6, len(groups) - .4), yticks=list(range(len(groups))),
                 yticklabels=list(reversed(groups)))
    axes[0].set(xscale="log", xlim=(2, 850), xticks=[3, 10, 30, 100, 300],
                xticklabels=["3", "10", "30", "100", "300"], xlabel="Mapping gain  P / F")
    axes[1].set(xlim=(0, .68), xticks=[0, .2, .4, .6], xlabel="Aggregation retention  A / P")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_title("(a) Mapping", loc="left", fontsize=10, pad=12)
    axes[1].set_title("(b) Aggregation", loc="left", fontsize=10, pad=12)
    return fig, axes


def dot(axis, value, y, color, marker, retention=False):
    axis.scatter(value, y, color=color, marker=marker, s=30, zorder=3)
    axis.annotate(f"{value:.1%}" if retention else f"{value:.1f}", (value, y),
                  xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)


def mechanisms(source, output):
    rows, windows, sources, paired_guidance = [], [], {}, []
    specs = [("9mau8hwi", "Flow-GRPO", ["guidance"]),
             ("4w87ez9o", "DiffusionART", ["art_guidance"]),
             ("26rilzmp", "NFT + EMA", ["guidance", "restore"]),
             ("adnxitig", "NFT: second update", ["guidance", "restore"]),
             ("75pf0fok", "Fast (partial)", ["guidance", "reference_mse"])]
    for rid, label, components in specs:
        if rid in ("9mau8hwi", "4w87ez9o"):
            data, merged, info = load_new(source, rid)
            selected = [read_component(rid, step, r, "probe/insight_followup/", components[0])
                        for step, r in merged if "probe/insight_followup/context/epoch" in r]
            assert len(selected) == 40 and all(r and r["gain_valid"] for r in selected)
            assert {(r["rollout"], r["window"]) for r in selected} == {
                (e, w) for e in range(20) for w in (0, 1)}
            sources[rid] = info
            rows.append(aggregate(selected, label, "guidance"))
            windows.extend(dict(r, group=label, population="all_40") for r in selected)
            for component in (["reference_mse", "art_restore"] if rid == "4w87ez9o" else ["reference_mse"]):
                component_rows = [read_component(rid, step, r, "probe/insight_followup/", component)
                                  for step, r in merged if "probe/insight_followup/context/epoch" in r]
                paired = { (r["rollout"], r["window"]): r for r in component_rows
                           if r and r["gain_valid"] and r["retention_valid"]
                           and (component != "art_restore" or r["window"] == 1)}
                assert len(paired) == (20 if component == "art_restore" else 39)
                matched_guidance = [r for r in selected if (r["rollout"], r["window"]) in paired]
                assert len(matched_guidance) == len(paired)
                canonical = "restore" if component == "art_restore" else component
                rows.append(aggregate(list(paired.values()), label, canonical))
                paired_guidance.append(dict(aggregate(matched_guidance, label, "guidance"), paired_with=canonical))
                windows.extend(dict(r, group=label, population=canonical + "_pair") for r in paired.values())
                windows.extend(dict(r, group=label, population=canonical + "_pair") for r in matched_guidance)
            continue
        path = source / (rid + ".json")
        data = json.loads(path.read_text())
        sources[rid] = source_info(path, data)
        prefix = "probe/force_conversion/"
        indexed = {c: {(int(r[prefix + "context/rollout"]), int(r[prefix + "context/window"])): r
                       for r in data["history"][c]} for c in components}
        paired = set.intersection(*(set(indexed[c]) for c in components))
        paired = {key for key in paired if (rid != "adnxitig" or key[1] == 1)
                  and all(indexed[c][key][prefix + c + "/" + FPA[0]] > 0 for c in components)}
        expected = {"26rilzmp": 18, "adnxitig": 20, "75pf0fok": 32}[rid]
        assert len(paired) == expected
        for c in components:
            selected = []
            for key in sorted(paired):
                r = indexed[c][key]
                r = {**r, prefix + "valid": True}
                entry = read_component(rid, r.get("_step"), r, prefix, c)
                assert entry and entry["gain_valid"]
                selected.append(entry)
            rows.append(aggregate(selected, label, c))
            windows.extend(dict(r, group=label) for r in selected)
    # Reconstruct and compare the existing Figure 12 estimates, not a fresh
    # independently filtered population of each component.
    previous = json.loads((source / "three_layer_force_probe.json").read_text())
    for old in previous["statistics"]:
        new = next(r for r in rows if r["run"] == old["run"] and r["component"] == old["component"])
        for metric in ("gain", "retention"):
            assert math.isclose(new[metric], old[metric], rel_tol=1e-10)
    fig, axes = dot_axes([label for _, label, _ in specs], 4.3)
    for index, (_, label, _) in enumerate(specs):
        group_rows = [r for r in rows if r["group"] == label]
        offsets = [0] if len(group_rows) == 1 else [.17, -.17] if len(group_rows) == 2 else [.28, -.28, 0]
        for offset, row in zip(offsets, group_rows):
            y = len(specs) - 1 - index + offset
            c = row["component"]
            for axis, metric in zip(axes, ("gain", "retention")):
                dot(axis, row[metric], y, COLORS[c], MARKERS[c], metric == "retention")
                matched = next((r for r in paired_guidance if r["run"] == row["run"] and r["paired_with"] == c), None)
                if matched:
                    axis.plot([matched[metric], row[metric]], [y, y], color="#CAD4D7", linewidth=.7, zorder=1)
                    axis.scatter(matched[metric], y, facecolors="white", edgecolors=COLORS["guidance"],
                                 marker="o", s=20, linewidth=.9, zorder=3)
    fig.legend(handles=[Line2D([], [], color=COLORS[c], marker=MARKERS[c], linestyle="None",
                               markersize=5, label=label) for c, label in
                        [("guidance", "Guidance"), ("restore", "Restoration"),
                         ("reference_mse", "Raw reference MSE")]] +
               [Line2D([], [], color=COLORS["guidance"], marker="o", markerfacecolor="white", linestyle="None",
                       markersize=4, label="Guidance: paired subset")],
               loc="lower center", bbox_to_anchor=(.5, .008), ncol=2, frameon=False, columnspacing=2.2)
    metadata = base_metadata("force_mechanisms", sources)
    metadata.update(statistics=rows, aggregation="Mean window F/P/A first, then P/F and A/P. Paired nonzero component windows within NFT/Fast groups.",
                    groups_in_display_order=[label for _, label, _ in specs],
                    component_scope="Filled Flow-GRPO/ART guidance uses all 40 windows. Their reference MSE uses 39 nonzero pairs; ART restoration uses 20 second-window pairs. Hollow guidance points and connectors compare exactly the same subset as each added component. NFT/Fast components use common paired populations. No value-based ordering.",
                    paired_guidance_statistics=paired_guidance,
                    cached_figure12_metadata_sha256=digest(source / "three_layer_force_probe.json"),
                    selected_contexts={rid: sorted({(r["rollout"], r["window"]) for r in windows if r["run"] == rid}) for rid, _, _ in specs})
    metadata["limitations"] += ["Fast is partial (32 nonzero pairs, 33 cached cloud windows, 34 local); 24 states/window vs 80 for the other groups.",
                                 "ART first-window restore is near-zero numerical residual and excluded, not interpreted as meaningful restoration.",
                                 "Reference MSE is unweighted diagnostic geometry, not applied KL (coefficient zero).",
                                 "NFT EMA 18 paired nonzero windows; dual-update NFT only 20 second windows."]
    table(output / "force_mechanisms_windows.csv", windows)
    finish(fig, output, "force_mechanisms", rows, metadata)
    return rows


def direction(source, output):
    rid = "bqrf8nn7"
    data, merged, info = load_new(source, rid)
    prefix = "probe/force_conversion/"
    components = ["restore_original"] + [f"restore_{kind}_seed_{seed}"
                    for kind in ("permuted", "random") for seed in SEEDS]
    by_component = {c: {} for c in components}
    for step, r in merged:
        if prefix + "context/rollout" not in r:
            continue
        for c in components:
            entry = read_component(rid, step, r, prefix, c)
            if entry and entry["gain_valid"] and entry["retention_valid"]:
                key = (entry["rollout"], entry["window"])
                assert key not in by_component[c]
                by_component[c][key] = entry
    paired = set.intersection(*(set(v) for v in by_component.values()))
    assert len(paired) == 19 and {k[0] for k in paired} == set(range(1, 20))
    windows, rows = [], []
    for kind, label in [("original", "Original"), ("permuted", "Same-timestep permutation"),
                        ("random", "Random direction")]:
        names = ["restore_original"] if kind == "original" else [f"restore_{kind}_seed_{s}" for s in SEEDS]
        selected = [dict(by_component[c][key], control_seed=None if kind == "original" else int(c.rsplit("_", 1)[1]),
                         direction=kind) for key in sorted(paired) for c in names]
        windows.extend(selected)
        rows.append(aggregate(selected, label, kind))
    assert max(r["F"] for r in rows) / min(r["F"] for r in rows) - 1 < 1e-7
    # Published ledger values are regression checks, never the data source.
    for row, gain, retention in zip(rows, [138.649, 73.7448, 3.19233], [.3216, .4969, .1293]):
        assert math.isclose(row["gain"], gain, rel_tol=1e-4)
        assert math.isclose(row["retention"], retention, abs_tol=1e-4)
    fig, axes = dot_axes([r["group"] for r in rows], 2.85)
    fig.subplots_adjust(left=.30, bottom=.26, top=.84, wspace=.33)
    colors = [COLORS["restore"], COLORS["guidance"], COLORS["reference_mse"]]
    for index, (row, color) in enumerate(zip(rows, colors)):
        for axis, metric in zip(axes, ("gain", "retention")):
            dot(axis, row[metric], 2 - index, color, ["s", "D", "o"][index], metric == "retention")
    metadata = base_metadata("force_direction_control", {rid: info})
    metadata.update(statistics=rows, control_seeds=SEEDS,
                    aggregation="Identical 19 nonzero paired windows. Mean F/P/A over windows and, for controls, 3 diagnostic seeds; then P/F and A/P. Original is not triplicated.",
                    selected_contexts=sorted(paired),
                    control_definition="Same state/Jacobian and receiving state force norm. Permute unit direction only within rank and timestep among nonzero endpoints; random unit direction rescaled to original norm. Training is unchanged.")
    metadata["limitations"] = ["Three direction-control seeds are not independent training seeds.",
                               "Larger diagnostic population than old NFT EMA: do not reuse its retention as an algorithm constant.",
                               "Full runtime diff not audited."]
    table(output / "force_direction_control_windows.csv", windows)
    finish(fig, output, "force_direction_control", rows, metadata)
    return rows


def broken_line(axis, rows, metric, **kwargs):
    xs, ys = [], []
    previous = None
    for row in rows:
        x = row["rollout"]
        if previous is not None and x != previous + 1:
            xs.append(float("nan"))
            ys.append(float("nan"))
        xs.append(x)
        value = row.get(metric)
        if (metric == "equal_state_gain" and finite(row.get("equal_state_valid_count"))
                and row["equal_state_valid_count"] < row["sample_count"]):
            value = None
        ys.append(value if finite(value) else float("nan"))
        previous = x
    axis.plot(xs, ys, **kwargs)


def stability(source, output):
    sources, windows, rewards, coverage = {}, [], [], {}
    specs = [("2xi0cj0u", "NFT", "probe/force_conversion/", "guidance"),
             ("f9la5w7l", "DiffusionART", "probe/insight_followup/", "art_guidance")]
    for rid, label, prefix, component in specs:
        data, merged, info = load_new(source, rid)
        sources[rid] = info
        train = data["config"]["train"]
        sample = data["config"]["sample"]
        assert train["learning_rate"] == .0003 and not train["cfg"] and train["beta"] == 0
        assert sample["num_image_per_prompt"] == 16 and data["config"]["seed"] == 42
        selected = []
        contexts = {}
        for step, r in merged:
            if prefix + ("context/epoch" if label == "DiffusionART" else "context/rollout") in r:
                contexts[step] = read_context(rid, step, r, prefix)
                entry = read_component(rid, step, r, prefix, component)
                if entry is None:
                    entry = dict(run=rid, source_url=info["url"], **contexts[step],
                                 component=component, training_seed=42, sample_count=None,
                                 valid=False, gain_valid=False, retention_valid=False,
                                 F=None, P=None, A=None, gain=None, retention=None,
                                 equal_state_gain=None, equal_state_valid_count=None, zero_force_count=None)
                entry["method"] = label
                selected.append(entry)
        assert len({(r["rollout"], r["window"]) for r in selected}) == len(selected)
        if label == "NFT":
            assert {r["window"] for r in selected} == {0}
        else:
            assert {r["microbatch"] for r in selected} == {0, 8}
        windows.extend(selected)
        for r in data["history"]["train/reward/avg"]:
            context = contexts.get(int(r["_step"]))
            if context is None:
                # Do not infer reward x from logged index or a nearby probe.
                raise ValueError(f"Unpaired reward context {rid}/{r['_step']}")
            rewards.append(dict(run=rid, method=label, logged_step=int(r["_step"]),
                                rollout=context["rollout"], window=context["window"],
                                optimizer_step=context["optimizer_step"], training_seed=42,
                                component="train_ocr", train_ocr=r["train/reward/avg"],
                                source_url=info["url"]))
        coverage[rid] = dict(probe_windows=len(selected), rollout_range=[min(r["rollout"] for r in selected), max(r["rollout"] for r in selected)],
                             invalid_probe_windows=sum(not r["valid"] for r in selected),
                             zero_force_windows=sum(r["F"] == 0 for r in selected),
                             equal_gain_windows=sum(r["equal_state_gain"] is not None for r in selected),
                             retention_windows=sum(r["retention"] is not None for r in selected),
                             sample_counts=sorted({r["sample_count"] for r in selected if r["sample_count"] is not None}),
                             equal_state_valid_counts=sorted({r["equal_state_valid_count"] for r in selected if r["equal_state_valid_count"] is not None}),
                             train_ocr_points=sum(r["run"] == rid for r in rewards))
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.9), sharex=True)
    fig.subplots_adjust(left=.10, right=.97, top=.93, bottom=.23, hspace=.52, wspace=.29)
    series = [("2xi0cj0u", 0, "NFT", COLORS["restore"], "o", "-"),
              ("f9la5w7l", 0, "ART: first window", COLORS["guidance"], "o", "-"),
              ("f9la5w7l", 1, "ART: second window", "#756B98", "s", "--")]
    for rid, window, label, color, marker, linestyle in series:
        selected = sorted([r for r in windows if r["run"] == rid and r["window"] == window], key=lambda r: r["rollout"])
        for axis, metric in zip(axes.flat, ("F", "equal_state_gain", "retention")):
            broken_line(axis, selected, metric, color=color, marker=marker, markersize=2.2,
                        linewidth=.85, linestyle=linestyle, label=label)
        partial = [r for r in selected if finite(r["equal_state_gain"])
                   and r["equal_state_valid_count"] < r["sample_count"]]
        axes[0, 1].scatter([r["rollout"] for r in partial], [r["equal_state_gain"] for r in partial],
                           facecolors="white", edgecolors=color, marker="D", s=17,
                           linewidth=.8, zorder=4)
    for rid, label, _, _ in specs:
        selected = sorted([r for r in rewards if r["run"] == rid], key=lambda r: r["rollout"])
        broken_line(axes[1, 1], selected, "train_ocr", color=COLORS["restore"] if label == "NFT" else COLORS["guidance"],
                    marker="o", markersize=2.2, linewidth=.85, label=label)
    titles = ["(a) Raw guidance force  F", "(b) Equal-state mapping gain", "(c) Aggregation retention", "(d) Train OCR"]
    x_max = max(r["rollout"] for r in windows)
    for axis, title in zip(axes.flat, titles):
        polish(axis)
        axis.set_title(title, loc="left", fontsize=10, pad=8)
        axis.set_xlim(-2, x_max + 2)
    axes[0, 0].set_yscale("symlog", linthresh=1e-6, linscale=.4)
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Mean output L2 (symlog)", fontsize=9)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Mean valid-state  P$_i$ / F$_i$", fontsize=9)
    axes[1, 0].set(ylim=(-.015, 1.015), ylabel="A / P")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1, 1].set(ylim=(-.02, 1.02), ylabel="OCR score")
    axes[1, 1].set_yticks([0, .5, 1])
    for axis in axes[1]:
        axis.set_xlabel("Rollout epoch (logged context)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#52636C", marker="D", markerfacecolor="white",
                          linestyle="None", markersize=4))
    labels.append("Gain: fewer valid states")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .025), ncol=2, frameon=False)
    metadata = base_metadata("force_stability", sources)
    extrema = {rid: {metric: max((r for r in windows if r["run"] == rid and finite(r.get(metric))), key=lambda r: r[metric], default=None)
                     for metric in ("F", "equal_state_gain", "retention")} for rid, _, _, _ in specs}
    metadata.update(coverage=coverage, extrema=extrema,
                    reward_endpoints={rid: [[r for r in rewards if r["run"] == rid][0],
                                            [r for r in rewards if r["run"] == rid][-1]] for rid, _, _, _ in specs},
                    plotted_components="Guidance only. ART restoration is not plotted; ART windows stay separate.",
                    gain_definition="Per-window mean_i(P_i/F_i) over valid nonzero-force states, not P/F of means. Valid count saved per window; no equal weighting over invalid states.",
                    partial_gain_windows=[dict(run=r["run"], rollout=r["rollout"], window=r["window"],
                                               equal_state_gain=r["equal_state_gain"],
                                               valid_count=r["equal_state_valid_count"], sample_count=r["sample_count"])
                                          for r in windows if finite(r["equal_state_gain"])
                                          and r["equal_state_valid_count"] < r["sample_count"]],
                    partial_gain_display="Fewer valid states than sampled states: isolated hollow diamonds, omitted from connecting lines. Invalid/zero-force ratios remain absent, never zero or stale summary values.",
                    retention_definition="Per-window A/P only when probe valid, F>0 and P>0.",
                    x_definition={"NFT": "probe/force_conversion/context/rollout", "ART": "probe/insight_followup/context/epoch"},
                    ART_window_definition="context/microbatch=0 first, 8 second; inner_epoch does not distinguish windows.",
                    reward_context="Exact same logged-step probe context, never nearest context. Reward sampled/logged before same-index update, not after it; no same-index gain -> reward causality.",
                    force_scale="Symlog, linear threshold 1e-6, linscale .4; zero forces remain visible.",
                    common_settings=dict(learning_rate=.0003, K=16, training_seed=42, cfg=False, kl=0),
                    freeze="Fetch-start summary cutoff per run; all targeted history requests bounded by cutoff. Rendering never queries W&B.")
    metadata["limitations"] += ["One update/rollout for NFT; two disjoint updates/rollout for ART. Runs remain at their frozen observed endpoints.",
                                 "Running status is cloud metadata, not full rank-health or budget-completion verification."]
    finish(fig, output, "force_stability", windows + rewards, metadata)
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=repo / "_draft_assets/diffusion-rl")
    parser.add_argument("--output", type=Path, default=repo / "_draft_assets/diffusion-rl")
    parser.add_argument("--public-output", type=Path, default=repo / "assets/blog/diffusion-rl")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    signal.alarm(args.timeout)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": "#243238", "axes.labelcolor": "#243238",
                         "xtick.color": "#69777D", "ytick.color": "#69777D",
                         "axes.edgecolor": "#C9D2D5", "axes.linewidth": .7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    stats = mechanisms(args.source, args.output)
    control = direction(args.source, args.output)
    coverage = stability(args.source, args.output)
    args.public_output.mkdir(parents=True, exist_ok=True)
    for name in ("force_mechanisms", "force_direction_control", "force_stability"):
        for path in args.output.glob(name + "*"):
            if path.suffix in (".png", ".pdf", ".csv", ".json") and "source_" not in path.name and "inventory_" not in path.name:
                if path.suffix == ".json":
                    dump(args.public_output / path.name,
                         public_metadata(json.loads(path.read_text()), repo, args.public_output))
                else:
                    shutil.copy2(path, args.public_output / path.name)
    signal.alarm(0)
    print(json.dumps(dict(mechanisms=stats, direction_control=control, stability_coverage=coverage), indent=2))


if __name__ == "__main__":
    main()
