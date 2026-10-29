#!/usr/bin/env python3
"""Summarize diagnostic JSONL files: accuracy vs verified simulator labels per (backend, variant, reasoning)."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(c - h, 3), round(c + h, 3))


def summarize(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["backend"], r["variant"], bool(r["reasoning"]))].append(r)
    out = []
    for (backend, variant, reasoning), rs in sorted(by.items()):
        sc = [r for r in rs if r.get("scored")]
        n = len(sc)
        tp = sum(1 for r in sc if r["cosmos_task_success"] and r["truth_success"])
        tn = sum(1 for r in sc if (not r["cosmos_task_success"]) and (not r["truth_success"]))
        fp = sum(1 for r in sc if r["cosmos_task_success"] and not r["truth_success"])
        fn = sum(1 for r in sc if (not r["cosmos_task_success"]) and r["truth_success"])
        # all-episode view: an unparsed output counts as a wrong prediction
        uns = [r for r in rs if not r.get("scored")]
        u_fail = sum(1 for r in uns if not r["truth_success"])
        u_succ = sum(1 for r in uns if r["truth_success"])
        bal_all = 0.5 * (tp / max(1, tp + fn + u_succ) + tn / max(1, tn + fp + u_fail))
        ft = [r for r in sc if "failure_type_correct" in r]
        fails = [r for r in ft if not r["truth_success"]]
        row = {
            "backend": backend, "variant": variant, "reasoning": reasoning,
            "episodes": len(rs), "scored": n, "unparsed": len(rs) - n,
            "success_acc": round((tp + tn) / n, 3) if n else None,
            "success_acc_ci95": wilson(tp + tn, n),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "balanced_acc": round(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), 3) if n else None,
            "balanced_acc_all": round(bal_all, 3),
            "success_acc_all": round((tp + tn) / len(rs), 3),
            "failure_type_acc": round(sum(r["failure_type_correct"] for r in ft) / len(ft), 3) if ft else None,
            "failure_type_acc_on_failures": round(sum(r["failure_type_correct"] for r in fails) / len(fails), 3) if fails else None,
            "n_failures_scored": len(fails),
            "mean_latency_s": round(sum(r["latency_s"] or 0 for r in rs) / len(rs), 2),
            "tasks": sorted({r["task"] for r in rs}),
        }
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--json-out")
    ap.add_argument("--task", default="")
    a = ap.parse_args()
    rows = []
    for p in a.jsonl:
        rows += [json.loads(l) for l in open(p)]
    if a.task:
        rows = [r for r in rows if r["task"] == a.task]
    res = summarize(rows)
    for r in res:
        print(json.dumps(r))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
