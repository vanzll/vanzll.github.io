"""Read-only W&B snapshots; plotting is a separate offline operation."""

import argparse
import json
import math
import signal
from datetime import datetime, timezone
from pathlib import Path

import wandb
import requests

PROJECT = "vanzl3386-chinese-university-of-hong-kong-shenzhen/flow-grpo-algo"
RUNS = ["9mau8hwi", "4w87ez9o", "bqrf8nn7", "2xi0cj0u", "f9la5w7l"]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=RUNS)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--supplement", action="store_true", help="Add raw reference and ART restore fields to an existing frozen snapshot")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=20)
    for rid in args.runs:
        signal.alarm(args.timeout)
        path = args.output / ("force_mechanisms_source_" + rid + ".json")
        if path.exists() and not args.inventory and not args.supplement:
            print(rid, "using frozen snapshot", flush=True)
            continue
        run = api.run(f"{PROJECT}/{rid}")
        summary = dict(run.summary._json_dict)
        keys = sorted(k for k in summary if k.startswith("probe/"))
        item = dict(id=rid, url=run.url, name=run.name, state=run.state,
                    commit=run._attrs.get("commit"), config=run.config,
                    summary=summary, available_probe_keys=keys,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    cutoff_step=int(summary.get("_step", 0)), history={})
        if args.supplement:
            item = json.loads(path.read_text())
            prefix = "probe/insight_followup/"
            components = ["reference_mse"] + (["art_restore"] if rid == "4w87ez9o" else [])
            supplement_keys = [prefix + "context/" + k for k in ("epoch", "microbatch", "optimizer_step")]
            supplement_keys += [prefix + "valid"]
            for component in components:
                supplement_keys += [prefix + component + "/" + k for k in
                                    ("output/individual_l2_mean", "parameter/individual_l2_mean",
                                     "parameter/aggregate_mean_l2", "sample_count", "gain_valid", "retention_valid")]
            rows = list(run.scan_history(keys=["_step", *supplement_keys], page_size=500,
                                         max_step=item["cutoff_step"] + 1))
            assert len(rows) == 40
            for key in supplement_keys:
                item["history"][key] = [{"_step": row["_step"], key: row[key]} for row in rows]
            item["supplement_fetched_at"] = datetime.now(timezone.utc).isoformat()
            save(path, item)
            print(rid, "supplemented", len(rows), "windows", flush=True)
            continue
        if args.inventory:
            save(args.output / ("force_mechanisms_inventory_" + rid + ".json"), item)
            print(rid, "state", run.state, "step", item["cutoff_step"],
                  "commit", item["commit"], "keys", json.dumps(keys), flush=True)
            continue
        # Separate always-logged fields from optional ratios so zero-force
        # windows survive sparse queries. Never join by nearest timestamp.
        components = (["restore_original", *[f"restore_{kind}_seed_{seed}"
                       for kind in ("permuted", "random") for seed in (314159, 271828, 161803)]]
                      if rid == "bqrf8nn7" else
                      ["art_guidance"] if rid in ("4w87ez9o", "f9la5w7l") else ["guidance"])
        selected = [k for k in keys if "context/" in k or "reconstruction/" in k
                    or k.endswith("conversion/valid") or k.endswith("followup/valid")
                    or (k.split("/")[2] in components and any(token in k for token in
                        ("valid", "individual_l2_mean", "aggregate_mean_l2",
                         "aggregation_retention", "equal_state_gain/", "sample_count",
                         "zero_force_count", "norm_weighted_gain")))]
        selected += [k for k in summary if k == "train/reward/avg"]
        batches = [[k for k in selected if "context/" in k],
                   [k for k in selected if k.endswith("conversion/valid") or k.endswith("followup/valid")]]
        for component in components:
            stem = "probe/" + ("force_conversion/" if rid in ("bqrf8nn7", "2xi0cj0u") else "insight_followup/") + component + "/"
            fields = [k for k in selected if k.startswith(stem)]
            batches += [[k for k in fields if "/joint/" not in k and not k.endswith("aggregation_retention")],
                        [k for k in fields if k.endswith("equal_state_gain/count") or k.endswith("equal_state_gain/valid")],
                        [k for k in fields if "/equal_state_gain/" in k and not k.endswith(("/count", "/valid"))],
                        [k for k in fields if k.endswith("norm_weighted_gain")],
                        [k for k in fields if k.endswith("aggregation_retention")]]
        batched = {k for batch in batches for k in batch}
        batches += [[k] for k in selected if k not in batched]
        if rid in ("2xi0cj0u", "f9la5w7l"):
            optional = [k for k in selected if k.endswith(("equal_state_gain/mean", "aggregation_retention", "norm_weighted_gain"))]
            selected = [k for k in selected if "reconstruction/" not in k
                        and not k.endswith(("equal_state_gain/std", "equal_state_gain/min", "equal_state_gain/max"))]
            core = [k for k in selected if k not in optional and k != "train/reward/avg"]
            contexts = [k for k in core if "context/" in k]
            validity = [k for k in core if k.endswith("conversion/valid") or k.endswith("followup/valid")]
            core = [k for k in core if k not in contexts + validity]
            batches = [contexts, validity, core, *[[k] for k in optional], ["train/reward/avg"]]
        if rid == "bqrf8nn7":
            core = [k for k in selected if "context/" in k or k.endswith("conversion/valid")
                    or any(k.endswith("/" + f) for f in
                           ("output/individual_l2_mean", "parameter/individual_l2_mean",
                            "parameter/aggregate_mean_l2", "sample_count", "gain_valid", "retention_valid"))]
            batches = [core]
        for index, batch in enumerate(batches):
            if not batch:
                continue
            rows = list(run.scan_history(keys=["_step", *batch], page_size=500,
                                         max_step=item["cutoff_step"] + 1))
            for key in batch:
                item["history"][key] = [{"_step": row["_step"], key: row[key]} for row in rows]
            print(rid, index, "/", len(batches), "fields", len(batch), "rows", len(rows), flush=True)
        item["fetch_completed_at"] = datetime.now(timezone.utc).isoformat()
        metadata_file = run.file("wandb-metadata.json")
        if metadata_file is not None:
            response = requests.get(metadata_file.url, timeout=20)
            response.raise_for_status()
            runtime = response.json()
            item["runtime_metadata"] = {k: runtime.get(k) for k in
                                        ("git", "codePath", "codePathLocal", "host", "startedAt")}
            item["commit"] = runtime.get("git", {}).get("commit", item["commit"])
        save(path, item)
        print(rid, "saved", str(path), flush=True)
        signal.alarm(0)


if __name__ == "__main__":
    main()
