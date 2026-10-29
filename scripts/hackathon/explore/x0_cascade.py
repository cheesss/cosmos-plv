#!/usr/bin/env python3
"""EXPLORATORY x0_cascade: local Cosmos3-Edge first, escalate to Nebius Nemotron only when needed.

Policies compared on the same eval episodes (scored by both backends):
  local_only, nebius_only, cascade_escalate_on_local_success (local says fail -> fail;
  local says success -> ask Nemotron), cascade_escalate_on_disagreement is not possible
  without calling both, so it is not included.
The rule is fixed a priori from the asymmetry measured on the DEV split
(local FN rare, FP common); dev numbers are reported next to eval numbers.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hackathon"))
from summarize_diagnostics import wilson  # noqa: E402


def load(p):
    return {(r["checkpoint_id"], r["env_seed"]): r for r in map(json.loads, open(p)) if r.get("scored")}


def evaluate(L, N):
    keys = sorted(set(L) & set(N))
    out = {}
    for name in ("local_only", "nebius_only", "cascade"):
        correct = calls = tp = tn = fp = fn = 0
        lat = 0.0
        for k in keys:
            t = bool(L[k]["truth_success"])
            ls = bool(L[k]["cosmos_task_success"])
            lat_l, lat_n = L[k]["latency_s"] or 0, N[k]["latency_s"] or 0
            if name == "local_only":
                p = ls; lat += lat_l
            elif name == "nebius_only":
                p = bool(N[k]["cosmos_task_success"]); calls += 1; lat += lat_n
            else:
                lat += lat_l
                if ls:
                    p = bool(N[k]["cosmos_task_success"]); calls += 1; lat += lat_n
                else:
                    p = False
            correct += p == t
            tp += p and t; tn += (not p) and (not t); fp += p and (not t); fn += (not p) and t
        n = len(keys)
        out[name] = {"n": n, "accuracy": round(correct / n, 3), "ci95": wilson(correct, n),
                     "balanced_accuracy": round(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), 3),
                     "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                     "nebius_calls": calls, "nebius_call_fraction": round(calls / n, 3),
                     "mean_latency_s": round(lat / n, 2)}
    return out


C = ROOT / "data/outputs/hackathon_cosmos"
dev_rows = [json.loads(l) for l in open(C / "dev_prompt_selection.jsonl")]
dL = {(r["checkpoint_id"], r["env_seed"]): r for r in dev_rows if r["backend"] == "local" and r["variant"] == "v6_perception_last" and r.get("scored")}
dN = {(r["checkpoint_id"], r["env_seed"]): r for r in dev_rows if r["backend"] == "nebius" and r["variant"] == "v2_evidence_first" and r.get("scored")}
res = {"label": "EXPLORATORY (post-hoc)", "dev": evaluate(dL, dN),
       "eval": evaluate(load(C / "eval_local_v6.jsonl"), load(C / "eval_nebius_v2.jsonl"))}
(ROOT / "results/hackathon/explore/x0_cascade.json").write_text(json.dumps(res, indent=2))
for split in ("dev", "eval"):
    for k, v in res[split].items():
        print(split, k, v)
