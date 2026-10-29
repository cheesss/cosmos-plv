#!/usr/bin/env python3
"""Chunk-aware PLV metrics for Diffusion Policy checkpoints.

The original PLV implementation replans after executing only the first action
from every predicted chunk. This entrypoint preserves that baseline and adds an
execution-matched rollout that advances the learned dynamics for
``cfg.n_action_steps`` primitive actions before replanning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.scripts.plv_lowdim_checkpoint_metrics import (  # noqa: E402
    build_dynamics_ensemble,
    build_mlp_dynamics_ensemble,
    checkpoint_metadata,
    discover_checkpoints,
    evaluate_action_metrics,
    get_policy_action_start,
    predict_with_uncertainty_components,
    rollout_plv_lite,
    stack_samples,
    summarize_logs,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def set_sampling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_value(cfg, key: str, default=""):
    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


def load_policy_with_dataset_cache(
        ckpt_path: Path,
        device: torch.device,
        use_ema: bool,
        dataset_cache: Dict[str, object]):
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)
    dataset_key = OmegaConf.to_yaml(cfg.task.dataset, resolve=True, sort_keys=True)
    if dataset_key not in dataset_cache:
        dataset_cache[dataset_key] = hydra.utils.instantiate(cfg.task.dataset)
    dataset = dataset_cache[dataset_key]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    state_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.to(device)
    policy.eval()
    del payload
    return cfg, policy, dataset, state_key


def checkpoint_execution_config(ckpt_path: Path, cfg, policy, state_key: str) -> Dict[str, object]:
    policy_target = str(cfg.policy._target_)
    return {
        "checkpoint_path": str(ckpt_path),
        "policy_target": policy_target,
        "state_key": state_key,
        "horizon": int(cfg_value(cfg, "horizon", getattr(policy, "horizon", -1))),
        "n_obs_steps": int(cfg_value(cfg, "n_obs_steps", getattr(policy, "n_obs_steps", -1))),
        "n_action_steps": int(cfg_value(cfg, "n_action_steps", getattr(policy, "n_action_steps", -1))),
        "policy_n_action_steps": int(getattr(policy, "n_action_steps", -1)),
        "n_latency_steps": int(cfg_value(cfg, "n_latency_steps", 0)),
        "pred_action_steps_only": bool(
            cfg_value(cfg, "pred_action_steps_only", getattr(policy, "pred_action_steps_only", False))
        ),
        "oa_step_convention": bool(
            cfg_value(cfg, "policy.oa_step_convention", getattr(policy, "oa_step_convention", False))
        ),
        "action_start": int(get_policy_action_start(cfg, policy_target)),
        "abs_action_runner": bool(cfg_value(cfg, "task.env_runner.abs_action", False)),
        "dataset_path": str(cfg_value(cfg, "task.dataset.dataset_path", "")),
        "train_filter_key": str(cfg_value(cfg, "task.dataset.train_filter_key", "")),
        "val_filter_key": str(cfg_value(cfg, "task.dataset.val_filter_key", "")),
    }


def mean_or_nan(values: List[np.ndarray]) -> float:
    if not values:
        return float("nan")
    arr = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in values])
    return float(np.mean(arr)) if arr.size else float("nan")


def max_per_start_mean(arr: np.ndarray) -> float:
    return float(np.max(arr, axis=1).mean())


def rollout_chunk_aware(
        policy,
        cfg,
        dataset,
        dynamics,
        device: torch.device,
        n_starts: int,
        action_horizon: int,
        sampling_seed: int,
        trace: bool = False) -> tuple[Dict[str, float], List[Dict[str, object]]]:
    """Roll out a predicted action chunk before asking the policy to replan."""
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")

    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        val_dataset = dataset
    batch = stack_samples(val_dataset, n_starts)
    n_obs_steps = int(cfg.n_obs_steps)
    obs_hist = batch["obs"][:, :n_obs_steps].to(device)
    action_scale = np.asarray(dynamics.action_std, dtype=np.float32)
    action_scale = np.maximum(action_scale, 1e-6)

    actions: List[np.ndarray] = []
    support_dists: List[np.ndarray] = []
    progresses: List[np.ndarray] = []
    uncertainties: List[np.ndarray] = []
    uncertainties_epi: List[np.ndarray] = []
    uncertainties_alea: List[np.ndarray] = []
    uncertainties_total: List[np.ndarray] = []
    boundary_l1_raw: List[np.ndarray] = []
    boundary_l2_raw: List[np.ndarray] = []
    boundary_l1_norm: List[np.ndarray] = []
    boundary_l2_norm: List[np.ndarray] = []
    overlap_l1_raw: List[np.ndarray] = []
    overlap_l2_raw: List[np.ndarray] = []
    overlap_l1_norm: List[np.ndarray] = []
    overlap_l2_norm: List[np.ndarray] = []
    traces: List[Dict[str, object]] = []

    set_sampling_seed(sampling_seed)
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(dynamics, "prime_context"):
        dynamics.prime_context(obs_hist)
    elif hasattr(dynamics, "reset_state"):
        dynamics.reset_state(obs_hist.shape[0])

    previous_plan: Optional[np.ndarray] = None
    previous_exec_len = 0
    previous_last_action: Optional[np.ndarray] = None
    primitive_step = 0
    policy_calls = 0
    action_start = get_policy_action_start(cfg, str(cfg.policy._target_))
    predicted_horizon = -1
    executable_horizon = -1
    started = time.perf_counter()

    with torch.no_grad():
        while primitive_step < action_horizon:
            result = policy.predict_action({"obs": obs_hist})
            policy_calls += 1
            executable = result["action"].detach().cpu().numpy().astype(np.float32)
            full_plan = result.get("action_pred", result["action"]).detach().cpu().numpy().astype(np.float32)
            predicted_horizon = int(full_plan.shape[1])
            executable_horizon = int(executable.shape[1])
            if executable_horizon < 1:
                raise RuntimeError("policy returned an empty action chunk")

            remaining = action_horizon - primitive_step
            execute_len = min(executable_horizon, remaining)
            chunk = executable[:, :execute_len]

            if previous_plan is not None:
                old_start = action_start + previous_exec_len
                old_available = max(0, previous_plan.shape[1] - old_start)
                new_available = max(0, full_plan.shape[1] - action_start)
                overlap_len = min(old_available, new_available)
                if overlap_len > 0:
                    old_suffix = previous_plan[:, old_start:old_start + overlap_len]
                    new_prefix = full_plan[:, action_start:action_start + overlap_len]
                    diff = new_prefix - old_suffix
                    diff_norm = diff / action_scale[None, None]
                    overlap_l1_raw.append(np.mean(np.abs(diff), axis=(1, 2)))
                    overlap_l2_raw.append(np.sqrt(np.mean(diff ** 2, axis=(1, 2))))
                    overlap_l1_norm.append(np.mean(np.abs(diff_norm), axis=(1, 2)))
                    overlap_l2_norm.append(np.sqrt(np.mean(diff_norm ** 2, axis=(1, 2))))
                if previous_last_action is not None:
                    jump = chunk[:, 0] - previous_last_action
                    jump_norm = jump / action_scale[None]
                    boundary_l1_raw.append(np.mean(np.abs(jump), axis=1))
                    boundary_l2_raw.append(np.sqrt(np.mean(jump ** 2, axis=1)))
                    boundary_l1_norm.append(np.mean(np.abs(jump_norm), axis=1))
                    boundary_l2_norm.append(np.sqrt(np.mean(jump_norm ** 2, axis=1)))
            else:
                overlap_len = 0

            cycle_start = primitive_step
            for local_idx in range(execute_len):
                action = chunk[:, local_idx]
                current_obs = obs_hist[:, -1].detach().cpu().numpy().astype(np.float32)
                next_obs, var, comps = predict_with_uncertainty_components(dynamics, current_obs, action)
                dist, progress = dynamics.nearest_support(next_obs)
                actions.append(action)
                support_dists.append(dist)
                progresses.append(progress)
                uncertainties.append(var)
                uncertainties_epi.append(comps["epi"])
                uncertainties_alea.append(comps["alea"])
                uncertainties_total.append(comps["total"])
                next_obs_t = torch.from_numpy(next_obs).to(device=device, dtype=obs_hist.dtype)[:, None]
                obs_hist = torch.cat([obs_hist[:, 1:], next_obs_t], dim=1)
                primitive_step += 1

            if trace:
                traces.append({
                    "policy_call": policy_calls - 1,
                    "primitive_start": cycle_start,
                    "primitive_end_exclusive": primitive_step,
                    "execute_len": execute_len,
                    "executable_horizon": executable_horizon,
                    "predicted_horizon": predicted_horizon,
                    "action_start": action_start,
                    "overlap_len": overlap_len,
                })
            previous_plan = full_plan
            previous_exec_len = execute_len
            previous_last_action = chunk[:, -1]

    runtime = time.perf_counter() - started
    action_arr = np.stack(actions, axis=1)
    support_arr = np.stack(support_dists, axis=1)
    progress_arr = np.stack(progresses, axis=1)
    uncertainty_arr = np.stack(uncertainties, axis=1)
    epi_arr = np.stack(uncertainties_epi, axis=1)
    alea_arr = np.stack(uncertainties_alea, axis=1)
    total_arr = np.stack(uncertainties_total, axis=1)
    init_obs = batch["obs"][:, n_obs_steps - 1].numpy()
    init_dist, init_progress = dynamics.nearest_support(init_obs)

    if action_arr.shape[1] >= 3:
        curvature = action_arr[:, 2:] - 2 * action_arr[:, 1:-1] + action_arr[:, :-2]
        curvature_norm = curvature / action_scale[None, None]
        smooth_raw = float(np.mean(curvature ** 2))
        smooth_norm = float(np.mean(curvature_norm ** 2))
    else:
        smooth_raw = float("nan")
        smooth_norm = float("nan")

    progress_delta = progress_arr.mean(axis=1) - init_progress
    progress_final = progress_arr[:, -1] - init_progress
    metrics = {
        "chunk_support_drift": float(support_arr.mean()),
        "chunk_support_drift_initial": float(init_dist.mean()),
        "chunk_support_drift_final": float(support_arr[:, -1].mean()),
        "chunk_support_drift_max": max_per_start_mean(support_arr),
        "chunk_progress_delta": float(progress_delta.mean()),
        "chunk_progress_delta_final": float(progress_final.mean()),
        "chunk_dynamics_uncertainty": float(uncertainty_arr.mean()),
        "chunk_dynamics_uncertainty_final": float(uncertainty_arr[:, -1].mean()),
        "chunk_dynamics_uncertainty_max": max_per_start_mean(uncertainty_arr),
        "chunk_dynamics_uncertainty_epi": float(epi_arr.mean()),
        "chunk_dynamics_uncertainty_epi_max": max_per_start_mean(epi_arr),
        "chunk_dynamics_uncertainty_alea": float(alea_arr.mean()),
        "chunk_dynamics_uncertainty_alea_max": max_per_start_mean(alea_arr),
        "chunk_dynamics_uncertainty_total": float(total_arr.mean()),
        "chunk_dynamics_uncertainty_total_max": max_per_start_mean(total_arr),
        "chunk_curvature_raw": smooth_raw,
        "chunk_curvature_norm": smooth_norm,
        "boundary_jump_l1_raw": mean_or_nan(boundary_l1_raw),
        "boundary_jump_l2_raw": mean_or_nan(boundary_l2_raw),
        "boundary_jump_l1_norm": mean_or_nan(boundary_l1_norm),
        "boundary_jump_l2_norm": mean_or_nan(boundary_l2_norm),
        "replan_overlap_l1_raw": mean_or_nan(overlap_l1_raw),
        "replan_overlap_l2_raw": mean_or_nan(overlap_l2_raw),
        "replan_overlap_l1_norm": mean_or_nan(overlap_l1_norm),
        "replan_overlap_l2_norm": mean_or_nan(overlap_l2_norm),
        "chunk_policy_calls": int(policy_calls),
        "chunk_dynamics_calls": int(action_horizon),
        "chunk_action_horizon": int(action_horizon),
        "chunk_execution_stride": int(executable_horizon),
        "chunk_predicted_horizon": int(predicted_horizon),
        "chunk_action_start": int(action_start),
        "chunk_boundaries": int(len(boundary_l1_raw)),
        "chunk_replan_overlaps": int(len(overlap_l1_raw)),
        "chunk_rollout_starts": int(action_arr.shape[0]),
        "chunk_sampling_seed": int(sampling_seed),
        "chunk_runtime_seconds": float(runtime),
    }
    return metrics, traces


CHUNK_COMPONENTS = {
    "chunk_support_drift_max": 1.0,
    "chunk_dynamics_uncertainty_max": 1.0,
    "chunk_curvature_norm": 1.0,
    "boundary_jump_l1_norm": 1.0,
    "replan_overlap_l1_norm": 1.0,
}


def robust_normalize_chunk_scores(rows: List[Dict[str, object]], group_keys: Iterable[str]) -> None:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in group_keys), []).append(row)

    for group_rows in groups.values():
        stats = {}
        for key in CHUNK_COMPONENTS:
            vals = np.asarray([
                float(row[key]) for row in group_rows
                if row.get(key, "") != "" and np.isfinite(float(row[key]))
            ])
            if vals.size:
                med = float(np.median(vals))
                iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                stats[key] = (med, iqr if iqr > 1e-9 else float(np.std(vals) + 1e-9))

        def component(row, key):
            if key not in stats or not np.isfinite(float(row.get(key, np.nan))):
                return np.nan
            med, scale = stats[key]
            return float(np.clip((float(row[key]) - med) / scale, -5.0, 5.0))

        def combine(row, keys):
            vals = [component(row, key) for key in keys]
            vals = [x for x in vals if np.isfinite(x)]
            return float(np.mean(vals)) if vals else np.nan

        for row in group_rows:
            row["chunk_plv_smoothness_score_norm"] = combine(row, ["chunk_curvature_norm"])
            row["chunk_plv_uncertainty_score_norm"] = combine(row, ["chunk_dynamics_uncertainty_max"])
            row["chunk_plv_u_s_score_norm"] = combine(
                row, ["chunk_dynamics_uncertainty_max", "chunk_curvature_norm"]
            )
            row["chunk_plv_s_s_score_norm"] = combine(
                row, ["chunk_support_drift_max", "chunk_curvature_norm"]
            )
            row["chunk_boundary_score_norm"] = combine(row, ["boundary_jump_l1_norm"])
            row["chunk_replan_score_norm"] = combine(row, ["replan_overlap_l1_norm"])
            row["chunk_plv_replan_score_norm"] = combine(
                row, ["chunk_curvature_norm", "replan_overlap_l1_norm"]
            )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    parser.add_argument("--trace-output")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--rollout-starts", type=int, default=32)
    parser.add_argument("--action-horizons", nargs="+", type=int, default=[10])
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--dyn-type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--dyn-models", type=int, default=5)
    parser.add_argument("--dyn-max-transitions", type=int, default=30000)
    parser.add_argument("--dyn-episode-split", choices=["train", "all"], default="train")
    parser.add_argument("--dyn-seed", type=int, default=0)
    parser.add_argument("--dyn-ridge", type=float, default=1e-3)
    parser.add_argument("--dyn-hidden-dim", type=int, default=256)
    parser.add_argument("--dyn-epochs", type=int, default=8)
    parser.add_argument("--dyn-batch-size", type=int, default=1024)
    parser.add_argument("--dyn-lr", type=float, default=1e-3)
    parser.add_argument("--progress-mode", choices=["semantic", "support_time"], default="semantic")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-latest", action="store_true")
    parser.add_argument("--include-one-step", action="store_true")
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--checkpoint-regex")
    parser.add_argument("--trace-checkpoints", type=int, default=0)
    parser.add_argument(
        "--normalize-group-keys", nargs="*",
        default=["chunk_action_horizon", "chunk_sampling_seed", "task", "dataset_type"],
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpts = discover_checkpoints([Path(x) for x in args.roots], skip_latest=args.skip_latest)
    if args.checkpoint_regex:
        import re
        pattern = re.compile(args.checkpoint_regex)
        ckpts = [path for path in ckpts if pattern.search(str(path))]
    if args.max_checkpoints is not None:
        ckpts = ckpts[:args.max_checkpoints]
    if not ckpts:
        raise SystemExit("No checkpoints matched the requested roots and filters")

    rows: List[Dict[str, object]] = []
    manifests: List[Dict[str, object]] = []
    trace_records: List[Dict[str, object]] = []
    dynamics_cache = {}
    dataset_cache = {}

    for ckpt_idx, ckpt_path in enumerate(ckpts):
        cfg, policy, dataset, state_key = load_policy_with_dataset_cache(
            ckpt_path, device=device, use_ema=args.use_ema, dataset_cache=dataset_cache
        )
        if "DiffusionUnetLowdimPolicy" not in str(cfg.policy._target_):
            print(f"Skipping non-Diffusion Policy checkpoint: {ckpt_path}")
            continue
        _, by_epoch = summarize_logs(ckpt_path.parent.parent / "logs.json.txt")
        base = checkpoint_metadata(ckpt_path, cfg, by_epoch)
        base["state_key"] = state_key
        manifest = checkpoint_execution_config(ckpt_path, cfg, policy, state_key)
        manifests.append({**base, **manifest})

        dataset_key = "|".join([
            str(Path(cfg.task.dataset.dataset_path)),
            f"dyn_type={args.dyn_type}",
            f"dyn_split={args.dyn_episode_split}",
            f"dyn_seed={args.dyn_seed}",
            f"train_filter={cfg_value(cfg, 'task.dataset.train_filter_key', '')}",
            f"val_filter={cfg_value(cfg, 'task.dataset.val_filter_key', '')}",
            f"dataset_seed={cfg_value(cfg, 'task.dataset.seed', '')}",
        ])
        if dataset_key not in dynamics_cache:
            common = dict(
                dataset=dataset,
                n_models=args.dyn_models,
                max_transitions=args.dyn_max_transitions,
                seed=args.dyn_seed,
                task_name=str(cfg.task.task_name),
                progress_mode=args.progress_mode,
                episode_split=args.dyn_episode_split,
            )
            if args.dyn_type == "mlp":
                dynamics_cache[dataset_key] = build_mlp_dynamics_ensemble(
                    **common,
                    device=device,
                    hidden_dim=args.dyn_hidden_dim,
                    epochs=args.dyn_epochs,
                    batch_size=args.dyn_batch_size,
                    lr=args.dyn_lr,
                )
            else:
                dynamics_cache[dataset_key] = build_dynamics_ensemble(
                    **common, ridge=args.dyn_ridge
                )
        dynamics = dynamics_cache[dataset_key]
        base.update({
            "plv_progress_mode": args.progress_mode,
            "dyn_type": args.dyn_type,
            "dyn_seed": args.dyn_seed,
            "dyn_episode_split": dynamics.episode_split,
            "dyn_support_episodes": dynamics.n_support_episodes,
            "dyn_support_steps": dynamics.n_support_steps,
            "dyn_train_transitions": dynamics.n_dynamics_transitions,
            **{k: v for k, v in manifest.items() if k != "checkpoint_path"},
        })
        base.update(evaluate_action_metrics(policy, cfg, dataset, device, args.max_eval_samples))

        one_step = {}
        if args.include_one_step:
            set_sampling_seed(args.sampling_seeds[0])
            one_step = {
                f"one_step_{key}": value for key, value in rollout_plv_lite(
                    policy, cfg, dataset, dynamics, device, args.rollout_starts, max(args.action_horizons)
                ).items()
            }

        for action_horizon in args.action_horizons:
            for sampling_seed in args.sampling_seeds:
                metrics, traces = rollout_chunk_aware(
                    policy=policy,
                    cfg=cfg,
                    dataset=dataset,
                    dynamics=dynamics,
                    device=device,
                    n_starts=args.rollout_starts,
                    action_horizon=action_horizon,
                    sampling_seed=sampling_seed,
                    trace=ckpt_idx < args.trace_checkpoints,
                )
                row = {**base, **one_step, **metrics}
                rows.append(row)
                print(json.dumps(row, sort_keys=True, allow_nan=True))
                for item in traces:
                    trace_records.append({
                        "checkpoint_path": str(ckpt_path),
                        "action_horizon": action_horizon,
                        "sampling_seed": sampling_seed,
                        **item,
                    })

    robust_normalize_chunk_scores(rows, args.normalize_group_keys)
    write_csv(Path(args.output), rows)
    manifest_path = Path(args.manifest_output) if args.manifest_output else Path(args.output).with_name(
        Path(args.output).stem + "_manifest.csv"
    )
    write_csv(manifest_path, manifests)
    if args.trace_output:
        trace_path = Path(args.trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w") as f:
            for record in trace_records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote {len(manifests)} checkpoint configs to {manifest_path}")


if __name__ == "__main__":
    main()
