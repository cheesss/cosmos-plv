#!/usr/bin/env python3
"""Experiment 2: compare Cosmos3-Edge (local) and Nemotron-3.5-Super-VL (Nebius)
against verified simulator labels on the evaluation split.

Writes results/hackathon/experiment2_backend_comparison.json and prints a summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hackathon"))

from summarize_diagnostics import summarize, wilson  # noqa: E402

FT = ("none", "no_grasp", "dropped", "not_placed", "other")


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def confusion(rows):
    m = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("scored") and r.get("cosmos_failure_type") is not None:
            m[r["truth_failure_type"]][r["cosmos_failure_type"]] += 1
    return {t: dict(m[t]) for t in FT if t in m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-success", default="data/outputs/hackathon_cosmos/eval_local_v6.jsonl")
    ap.add_argument("--local-failure", default="data/outputs/hackathon_cosmos/eval_local_v7.jsonl")
    ap.add_argument("--nebius", default="data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl")
    ap.add_argument("--out", default="results/hackathon/experiment2_backend_comparison.json")
    a = ap.parse_args()
    files = {k: ROOT / v for k, v in (("local_success", a.local_success), ("local_failure", a.local_failure),
                                       ("nebius", a.nebius)) if (ROOT / v).exists()}
    data = {k: load([p]) for k, p in files.items()}
    out = {"files": {k: str(p.relative_to(ROOT)) for k, p in files.items()}, "overall": {}, "by_task": {},
           "failure_type_confusion": {}, "agreement": None}
    for k, rows in data.items():
        out["overall"][k] = summarize(rows)
        out["by_task"][k] = {t: summarize([r for r in rows if r["task"] == t]) for t in sorted({r["task"] for r in rows})}
        if k in ("local_failure", "nebius"):
            out["failure_type_confusion"][k] = confusion(rows)
    if "local_success" in data and "nebius" in data:
        L = {(r["checkpoint_id"], r["env_seed"]): r for r in data["local_success"] if r.get("scored")}
        N = {(r["checkpoint_id"], r["env_seed"]): r for r in data["nebius"] if r.get("scored")}
        both = sorted(set(L) & set(N))
        agree = [k for k in both if bool(L[k]["cosmos_task_success"]) == bool(N[k]["cosmos_task_success"])]
        dis = [k for k in both if k not in agree]
        acc_agree = sum(bool(L[k]["cosmos_task_success"]) == bool(L[k]["truth_success"]) for k in agree)
        out["agreement"] = {
            "episodes_scored_by_both": len(both),
            "agree": len(agree), "agree_rate": round(len(agree) / max(1, len(both)), 3),
            "accuracy_when_agree": round(acc_agree / max(1, len(agree)), 3), "ci95_when_agree": wilson(acc_agree, len(agree)),
            "disagree": len(dis),
            "local_right_when_disagree": sum(bool(L[k]["cosmos_task_success"]) == bool(L[k]["truth_success"]) for k in dis),
            "nebius_right_when_disagree": sum(bool(N[k]["cosmos_task_success"]) == bool(N[k]["truth_success"]) for k in dis),
        }
    Path(ROOT / a.out).write_text(json.dumps(out, indent=2))
    for k, s in out["overall"].items():
        for r in s:
            print(f"{k:14s} {r['variant']:22s} n={r['episodes']:3d} acc_all={r['success_acc_all']} "
                  f"bal_all={r['balanced_acc_all']} ci95={r['success_acc_ci95']} tp={r['tp']} tn={r['tn']} fp={r['fp']} "
                  f"fn={r['fn']} unparsed={r['unparsed']} ft_on_failures={r['failure_type_acc_on_failures']} "
                  f"lat={r['mean_latency_s']}s")
    for k, bt in out["by_task"].items():
        for t, s in bt.items():
            for r in s:
                print(f"  {k:12s} {t:6s} n={r['episodes']:3d} acc_all={r['success_acc_all']} bal_all={r['balanced_acc_all']} "
                      f"fp={r['fp']} fn={r['fn']} ft_on_failures={r['failure_type_acc_on_failures']}")
    print("agreement:", json.dumps(out["agreement"]))
    for k, cm in out["failure_type_confusion"].items():
        print(f"confusion {k} (rows=truth, cols=predicted):")
        for t, c in cm.items():
            print(f"   {t:10s} {c}")


if __name__ == "__main__":
    main()
