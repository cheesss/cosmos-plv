#!/usr/bin/env python3
"""Experiments 2 + 3: VLM failure assessment and hybrid checkpoint screening.

Inputs (all produced earlier in the pipeline):
  results/hackathon/baseline_metrics.csv             cheap offline metrics + S_i
  data/outputs/hackathon_rollouts_eval/*/summary.json K=10 eval rollouts per ckpt
  data/outputs/hackathon_cosmos/eval_*.jsonl          frozen-variant VLM diagnostics
Outputs:
  results/hackathon/cosmos_diagnostics.csv   one row per (episode, backend)
  results/hackathon/hybrid_screening.csv     one row per eval checkpoint
  results/hackathon/hybrid_ranking.csv       within-run Spearman / Regret per score
  results/hackathon/experiment2_summary.json, experiment3_summary.json
No score is fitted on any outcome; the target S_i is only used for evaluation.
Run in Environment A (needs h5py + RotationTransformer for action scaling).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hackathon"))

from cosmos_plv.cheap_metrics import executed_smoothness, replan_metrics_from_plans  # noqa: E402
from cosmos_plv.provenance import git_commit, utc_now, write_json  # noqa: E402
from cosmos_plv.ranking import evaluate_metric, regret_at_q  # noqa: E402
from summarize_diagnostics import summarize  # noqa: E402

CHEAP = [("val_action_l1", "Action L1"), ("open_loop_smoothness", "Open-loop smoothness"),
         ("plv1_smoothness", "PLV-1 smoothness"), ("replan_overlap_h16", "Replan overlap (model, H=16)")]


def action_scale(task: str) -> np.ndarray:
    """Per-dimension std of demonstration actions in the 10-D policy space."""
    import h5py

    from diffusion_policy.model.common.rotation_transformer import RotationTransformer

    tf = RotationTransformer("axis_angle", "rotation_6d")
    with h5py.File(ROOT / f"data/robomimic/datasets/{task}/ph/low_dim_abs.hdf5", "r") as f:
        a = np.concatenate([f[f"data/{k}/actions"][:] for k in f["data"]])
    a10 = np.concatenate([a[:, :3], tf.forward(a[:, 3:6]), a[:, 6:]], axis=1)
    return a10.std(axis=0) + 1e-6


def rank_average(*cols):
    """Mean of within-run percentile ranks (higher = better). No fitted weights."""
    ranks = [pd.Series(c).rank(pct=True).values for c in cols]
    return np.mean(ranks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/hackathon/baseline_metrics.csv")
    ap.add_argument("--rollouts", default="data/outputs/hackathon_rollouts_eval")
    # skip the <name>.orig.jsonl backups that refresh_truth.py keeps next to each file
    ap.add_argument("--diag", nargs="+", default=sorted(str(p) for p in (ROOT / "data/outputs/hackathon_cosmos").glob("eval_*.jsonl")
                                                        if not p.name.endswith(".orig.jsonl")))
    ap.add_argument("--out-dir", default="results/hackathon")
    ap.add_argument("--q", type=float, default=0.3)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = ROOT / a.out_dir
    base = pd.read_csv(ROOT / a.baseline)

    # ---- eval rollouts: instrumented K-episode success + real-feedback replanning
    scales, ep_rows = {}, []
    for summ in sorted((ROOT / a.rollouts).glob("*/summary.json")):
        s = json.loads(summ.read_text())
        for ep in s["episodes"]:
            task = ep["task"]
            if task not in scales:
                scales[task] = action_scale(task)
            d = np.load(ROOT / ep["episode_npz"])
            rp = replan_metrics_from_plans(d["action_pred"], d["plan_t"], ep["horizon"], ep["n_obs_steps"],
                                           ep["n_action_steps"], scales[task])
            ep_rows.append({**{k: ep[k] for k in ("checkpoint_id", "env_seed", "task", "success", "failure_type",
                                                  "batch_wall_s", "policy_calls", "n_steps", "video")},
                            **rp, "executed_smoothness": executed_smoothness(d["action"])})
    eps = pd.DataFrame(ep_rows)
    per_ck = eps.groupby("checkpoint_id").agg(
        k_episodes=("success", "size"), k_sim_success=("success", "mean"),
        sim_replan_overlap=("sim_replan_overlap_l1", "mean"), sim_boundary_jump=("sim_boundary_jump_l1", "mean"),
        executed_smoothness=("executed_smoothness", "mean"),
        rollout_wall_s_per_batch=("batch_wall_s", "first")).reset_index()

    # ---- VLM diagnostics (frozen variants)
    diag_rows = []
    for p in a.diag:
        diag_rows += [json.loads(l) for l in open(p)]
    diag = pd.DataFrame(diag_rows)
    exp2 = summarize(diag_rows) if diag_rows else []
    if len(diag):
        keep = ["checkpoint_id", "env_seed", "task", "backend", "variant", "reasoning", "model_id", "model_revision",
                "served_model", "request_id", "video", "truth_success", "truth_failure_type", "truth_grasped",
                "truth_lifted", "truth_dropped", "parse_status", "parse_errors", "cosmos_task_success",
                "cosmos_failure_type", "cosmos_object_lifted", "cosmos_object_dropped", "cosmos_contact_observed",
                "cosmos_failure_time_s", "cosmos_explanation", "scored", "success_correct", "failure_type_correct",
                "latency_s", "prompt_tokens", "completion_tokens", "backend_error", "started_utc", "git_commit"]
        diag[[c for c in keep if c in diag]].to_csv(out / "cosmos_diagnostics.csv", index=False)
        judged = diag[diag.scored.fillna(False).astype(bool)].copy()
        judged["pred"] = judged.cosmos_task_success.astype(bool).astype(float)
        vlm = judged.pivot_table(index="checkpoint_id", columns="backend", values="pred", aggfunc="mean")
        vlm.columns = [f"vlm_success_{c}" for c in vlm.columns]
        n_sc = judged.pivot_table(index="checkpoint_id", columns="backend", values="pred", aggfunc="size")
        n_sc.columns = [f"vlm_scored_{c}" for c in n_sc.columns]
        lat = diag.pivot_table(index="checkpoint_id", columns="backend", values="latency_s", aggfunc="sum")
        lat.columns = [f"vlm_latency_s_{c}" for c in lat.columns]
        per_ck = per_ck.merge(vlm.reset_index(), on="checkpoint_id", how="left") \
                       .merge(n_sc.reset_index(), on="checkpoint_id", how="left") \
                       .merge(lat.reset_index(), on="checkpoint_id", how="left")

    df = base.merge(per_ck, on="checkpoint_id", how="inner").sort_values(["run_id", "epoch"]).reset_index(drop=True)

    # ---- scores (higher = better), all outcome-free
    scores = {"epoch": ("Training epoch (free baseline)", False),
              "k_sim_success": ("K=10 instrumented sim success (privileged)", False)}
    for col, label in CHEAP:
        scores[col] = (label, True)
    scores["sim_replan_overlap"] = ("Replan overlap on real sim feedback (K=10)", True)
    vlm_cols = [c for c in df.columns if c.startswith("vlm_success_")]
    for c in vlm_cols:
        scores[c] = (f"VLM-judged success, K=10 ({c.replace('vlm_success_', '')})", False)
    for run, g in df.groupby("run_id"):
        idx = g.index
        for c in vlm_cols:
            b = c.replace("vlm_success_", "")
            df.loc[idx, f"hybrid_rankavg_l1_{b}"] = rank_average(-g.val_action_l1.values, g[c].values)
            # two-stage: keep top-q by Action L1, then order by VLM success (ties -> Action L1)
            k = max(1, math.ceil(a.q * len(g)))
            order = g.sort_values(["val_action_l1", "checkpoint_path"]).index[:k]
            st = np.full(len(g), -np.inf)
            pos = {ix: i for i, ix in enumerate(idx)}
            for ix in order:
                st[pos[ix]] = g.loc[ix, c] - 1e-6 * g.loc[ix, "val_action_l1"]
            df.loc[idx, f"hybrid_twostage_l1_{b}"] = st
    for c in vlm_cols:
        b = c.replace("vlm_success_", "")
        scores[f"hybrid_rankavg_l1_{b}"] = (f"Hybrid rank-average(Action L1, VLM {b})", False)
        scores[f"hybrid_twostage_l1_{b}"] = (f"Two-stage: top-30% Action L1 -> VLM {b}", False)

    rows = []
    for col, (label, lib) in scores.items():
        if col not in df or df[col].replace([-np.inf], np.nan).isna().all():
            continue
        tmp = df.copy()
        tmp[col] = tmp[col].replace([-np.inf], -1e9)
        for r in evaluate_metric(tmp, col, lib, "success", "run_id", qs=(0.1, 0.3), n_boot=a.n_boot):
            r["label"] = label
            rows.append(r)
        # Regret@1 of the single final pick, per run
        for run, g in tmp.groupby("run_id"):
            s = -g[col].values if lib else g[col].values
            rr = regret_at_q(s, g.success.values, 1e-9, g.checkpoint_path.values)
            rows.append({"metric": col, "label": label, "run_id": run, "pick_regret@1": rr["regret"],
                         "n_checkpoints": rr["n"]})
    rank = pd.DataFrame(rows)
    df.to_csv(out / "hybrid_screening.csv", index=False)
    rank.to_csv(out / "hybrid_ranking.csv", index=False)
    write_json(out / "experiment2_summary.json", {"generated_utc": utc_now(), "git_commit": git_commit(),
                                                  "n_diagnostic_rows": len(diag_rows), "by_backend_variant": exp2})
    write_json(out / "experiment3_summary.json", {
        "generated_utc": utc_now(), "git_commit": git_commit(), "n_checkpoints": int(len(df)),
        "runs": sorted(df.run_id.unique().tolist()), "q": a.q, "ranking": rows,
        "episodes": int(len(eps)), "eval_success_overall": float(eps.success.mean()) if len(eps) else None})
    m = rank[rank.run_id == "mean_over_runs"]
    if len(m):
        print(m[["label", "spearman", "regret@30"]].to_string(index=False))
    p = rank[rank.get("pick_regret@1").notna()] if "pick_regret@1" in rank else pd.DataFrame()
    if len(p):
        print(p.groupby("label")["pick_regret@1"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
