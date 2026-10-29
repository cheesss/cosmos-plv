#!/usr/bin/env python3
"""EXPLORATORY verification of x1_adaptive_budget: compare x1's published JSON with
(a) the exact re-run on the frozen snapshot (repro_frozen.json) and
(b) the re-run on the data present now (rerun_current.json).
Writes results/hackathon/explore/x1_adaptive_budget_verification_compare.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCR = ROOT / "data/outputs/hackathon_explore/x1_adaptive_budget_verification"
X1 = json.loads((ROOT / "results/hackathon/explore/x1_adaptive_budget.json").read_text())
OUT = ROOT / "results/hackathon/explore/x1_adaptive_budget_verification_compare.json"

KEY_STRATS = ["uniform", "successive_halving", "ucb1", "thompson", "top_two_thompson",
              "late_epoch_top30+top_two_thompson", "val_loss_top30+top_two_thompson",
              "random_top30+top_two_thompson", "train_action_mse_top30+top_two_thompson"]
KEY_JUDGES = ["perfect", "nemotron_nebius_v2", "cosmos3_edge_local_v6", "cascade_local_then_nebius",
              "cosmos3_edge_local_v6_ftype", "nemotron_nebius_v2_ftype"]


def index(d):
    return {(r["judge"], r["strategy"], r["budget_mult"]): r for r in d["primary"]}


def thr(d, table="thresholds_primary"):
    return {(t["judge"], t["strategy"]): t for t in d[table]}


def compare(other, label):
    a, b = index(X1), index(other)
    keys = sorted(set(a) & set(b), key=str)
    diffs = [abs(a[k]["mean_regret"] - b[k]["mean_regret"]) for k in keys]
    ta, tb = thr(X1), thr(other)
    sa = {(t["judge"], t["strategy"]): t for t in X1["random_splits"]["thresholds"]}
    sb = {(t["judge"], t["strategy"]): t for t in other["random_splits"]["thresholds"]}
    rows = []
    for j in KEY_JUDGES:
        for st in KEY_STRATS:
            if (j, st) not in ta or (j, st) not in tb:
                continue
            rows.append({"judge": j, "strategy": st,
                         "x1_excess_le_0.02": ta[(j, st)]["mult_excess_vs_full_pool_le_thr"],
                         f"{label}_excess_le_0.02": tb[(j, st)]["mult_excess_vs_full_pool_le_thr"],
                         "x1_abs_le_0.02": ta[(j, st)]["mult_mean_regret_le_thr"],
                         f"{label}_abs_le_0.02": tb[(j, st)]["mult_mean_regret_le_thr"],
                         "x1_resplit_excess": sa.get((j, st), {}).get("mult_excess_vs_mean_full_pool_floor_le_thr"),
                         f"{label}_resplit_excess": sb.get((j, st), {}).get("mult_excess_vs_mean_full_pool_floor_le_thr"),
                         **{f"x1_{m}N": round(a[(j, st, m)]["mean_regret"], 4) for m in (1, 4, 25)},
                         **{f"{label}_{m}N": round(b[(j, st, m)]["mean_regret"], 4) for m in (1, 4, 25)}})
    ctx = [{"run": c["run"], "N": c["n_checkpoints"], "floor": round(c["full_pool_uniform_regret_noise_floor"], 4),
            "heldout_best": c["heldout_best"],
            "spearman": {p: round(v["spearman_vs_heldout"], 3) for p, v in c["priors"].items()}}
           for c in other["context"]]
    jc = {j: {t: (v["fp"], v["n_fail"], v["fn"], v["n_succ"]) for t, v in dd.items() if t != "by_failure_type"}
          | {"by_failure_type": dd["by_failure_type"]} for j, dd in other["judge_counts"].items()}
    return {"n_cells_compared": len(keys), "max_abs_diff_mean_regret": max(diffs) if diffs else None,
            "n_cells_identical": sum(1 for x in diffs if x < 1e-12),
            "runtime_s": other.get("runtime_s"), "context": ctx, "judge_counts": jc,
            "dropped_share": other["dropped_share_eval_rollouts"],
            "split_floor_mean": sum(other["random_splits"]["full_pool_noise_floor_per_split"]) /
                                len(other["random_splits"]["full_pool_noise_floor_per_split"]),
            "thresholds": rows}


res = {"name": "x1_adaptive_budget_verification_compare", "exploratory": True}
for label, fn in (("repro_frozen", "repro_frozen.json"), ("rerun_current", "rerun_current.json")):
    p = SCR / fn
    if p.is_file():
        res[label] = compare(json.loads(p.read_text()), label)
OUT.write_text(json.dumps(res, indent=1))
for label in ("repro_frozen", "rerun_current"):
    if label in res:
        r = res[label]
        print(label, "cells", r["n_cells_compared"], "identical", r["n_cells_identical"], "max diff", r["max_abs_diff_mean_regret"])
        print(" ctx", r["context"])
        print(" dropped", r["dropped_share"], "split floor", round(r["split_floor_mean"], 4))
        print(" judge by_ftype", {j: v["by_failure_type"] for j, v in r["judge_counts"].items()})
        for t in r["thresholds"]:
            print("  ", t)
