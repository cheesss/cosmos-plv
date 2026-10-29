#!/usr/bin/env python3
"""Experiment 1 analysis: join original-script metrics with training-time success,
compute within-run oriented Spearman / Regret@30 and write the results files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cosmos_plv.checkpoint_adapter import discover_checkpoints  # noqa: E402
from cosmos_plv.provenance import git_commit, repo_relative, utc_now, write_json  # noqa: E402
from cosmos_plv.ranking import evaluate_metric  # noqa: E402

# (column, label, lower_is_better, level)
METRICS = [
    ("epoch", "Training epoch (free baseline: later = better)", False, "B"),
    ("val_loss_logged", "Validation loss (objective)", True, "L0"),
    ("val_action_l1", "Action L1", True, "L1"),
    ("open_loop_smoothness", "Open-loop smoothness (teacher-forced)", True, "L2"),
    ("plv1_smoothness", "PLV-1 smoothness (model feedback)", True, "L2"),
    ("boundary_jump_h16", "Boundary jump (model feedback, H=16, median of 4 seeds)", True, "L3"),
    ("replan_overlap_h16", "Replan overlap (model feedback, H=16, median of 4 seeds)", True, "L3"),
    ("boundary_jump_h10", "Boundary jump (model feedback, H=10)", True, "L3"),
    ("replan_overlap_h10", "Replan overlap (model feedback, H=10)", True, "L3"),
]


def key(p: str) -> str:
    return repo_relative(p)


def load(metrics_dir: Path, pool: Path) -> pd.DataFrame:
    recs = pd.DataFrame([r.to_dict() for r in discover_checkpoints(pool)])
    recs["key"] = recs.checkpoint_path.map(key)
    recs["run_id"] = recs.task + "_seed" + recs.seed.astype(str)
    recs = recs.rename(columns={"train_rollout_success": "success"})
    df = recs[["key", "checkpoint_id", "checkpoint_path", "run_id", "task", "seed", "epoch", "success",
               "n_train_rollouts", "val_loss", "train_loss", "size_bytes"]].copy()
    cat = lambda name: pd.concat([pd.read_csv(f) for f in sorted(metrics_dir.glob(f"*/{name}"))], ignore_index=True)
    plv = cat("plv1.csv")
    plv["key"] = plv.checkpoint_path.map(key)
    plv = plv.rename(columns={"smoothness": "plv1_smoothness"})
    df = df.merge(plv[["key", "val_action_l1", "val_loss_logged", "plv1_smoothness", "support_drift",
                       "dynamics_uncertainty", "teacher_forced_latent_mse"]], on="key", how="left")
    ol = cat("open_loop.csv")
    ol["key"] = ol.checkpoint_path.map(key)
    df = df.merge(ol[["key", "open_loop_smoothness"]], on="key", how="left")
    ch = cat("chunk.csv")
    ch["key"] = ch.checkpoint_path.map(key)
    for h in (10, 16):
        sub = ch[ch.chunk_action_horizon == h].groupby("key")[["boundary_jump_l1_norm", "replan_overlap_l1_norm"]].median()
        sub.columns = [f"boundary_jump_h{h}", f"replan_overlap_h{h}"]
        df = df.merge(sub.reset_index(), on="key", how="left")
    return df.sort_values(["run_id", "epoch"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="data/outputs/hackathon_metrics")
    ap.add_argument("--pool", default="data/outputs/hackathon_pool")
    ap.add_argument("--out-dir", default="results/hackathon")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df = load(ROOT / a.metrics_dir, ROOT / a.pool)
    df.to_csv(out / "baseline_metrics.csv", index=False)  # all checkpoints (metrics NaN where not yet computed)
    rows = []
    # Compare every metric on the SAME checkpoints: those that have the cheap metrics.
    full = df
    if "val_action_l1" in df:
        df = df[df.val_action_l1.notna()].copy()
    for col, label, lib, level in METRICS:
        if col not in df or df[col].isna().all():
            continue
        for r in evaluate_metric(df, col, lib, "success", "run_id", qs=(0.3,), n_boot=a.n_boot):
            r.update({"label": label, "level": level})
            rows.append(r)
    # Outcome-free "late half" subset (second half of each run's checkpoints by epoch):
    # approximates the plateau regime of the paper's long runs, where epoch order is
    # less informative. Reported separately; nothing is selected on outcomes.
    late = df[df.epoch >= df.groupby("run_id").epoch.transform("median")]
    for col, label, lib, level in METRICS:
        if col not in late or late[col].isna().all():
            continue
        for r in evaluate_metric(late, col, lib, "success", "run_id", qs=(0.3,), n_boot=a.n_boot):
            r.update({"label": label, "level": level, "subset": "late_half"})
            rows.append(r)
    rank = pd.DataFrame(rows)
    rank["subset"] = rank.get("subset", pd.Series(index=rank.index, dtype=object)).fillna("all")
    rank.to_csv(out / "baseline_ranking.csv", index=False)
    write_json(out / "baseline_summary.json", {
        "generated_utc": utc_now(), "git_commit": git_commit(), "n_checkpoints": int(len(df)),
        "runs": df.groupby("run_id").agg(n=("epoch", "size"), min_epoch=("epoch", "min"), max_epoch=("epoch", "max"),
                                         min_success=("success", "min"), max_success=("success", "max")).reset_index().to_dict("records"),
        "missing_metric_rows": {c: int(df[c].isna().sum()) for c, *_ in METRICS if c in df},
        "ranking": rows,
    })
    print(rank[rank.run_id == "mean_over_runs"][["subset", "label", "n_checkpoints", "spearman", "regret@30"]].to_string(index=False))


if __name__ == "__main__":
    main()
