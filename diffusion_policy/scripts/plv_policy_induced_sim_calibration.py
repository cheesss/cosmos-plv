#!/usr/bin/env python3
"""Small policy-induced simulator calibration for PLV.

This is intentionally not a new policy-training experiment. It selects a small
set of already-trained Diffusion Policy checkpoints, starts from held-out
demonstration states, and compares short-horizon learned-dynamics predictions
against real robosuite transitions under the same checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from diffusion_policy.common.robomimic_util import set_absolute_controller_config
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.env_runner.robomimic_lowdim_runner import make_robosuite_metadata_compatible
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.scripts.plv_lowdim_checkpoint_metrics import (
    build_mlp_dynamics_ensemble,
    load_policy_and_dataset,
)


DEFAULT_INPUT = ROOT / "data/robomimic/plv_assets/plv_final_merged_checkpoint_analysis.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data/robomimic/plv_assets/policy_induced_sim_calibration"


FOCUS_GROUPS = [
    ("square", "mh", 1),
    ("transport", "ph", 1),
    ("tool_hang", "ph", 1),
    ("can", "mh", 1),
]


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return np.nan
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return np.nan
    result = spearmanr(x_arr, y_arr)
    return float(getattr(result, "statistic", getattr(result, "correlation", result[0])))


def sorted_demo_keys(demos) -> list[str]:
    return sorted(demos.keys(), key=lambda name: int(name.split("_")[-1]))


def read_mask_keys(file: h5py.File, key: str | None) -> list[str]:
    if not key or "mask" not in file or key not in file["mask"]:
        return []
    out = []
    for value in file["mask"][key][:]:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        out.append(str(value))
    return out


def obs_at(demo, obs_keys: list[str], t: int) -> np.ndarray:
    return np.concatenate([demo["obs"][key][t] for key in obs_keys], axis=0).astype(np.float32)


def obs_history(demo, obs_keys: list[str], t: int, n_obs_steps: int) -> np.ndarray:
    start = t - n_obs_steps + 1
    return np.stack([obs_at(demo, obs_keys, i) for i in range(start, t + 1)], axis=0)


def sample_validation_starts(
    dataset_path: Path,
    obs_keys: list[str],
    n_obs_steps: int,
    horizon: int,
    n_starts: int,
    val_filter_key: str | None,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    starts = []
    with h5py.File(dataset_path, "r") as file:
        demos = file["data"]
        keys = read_mask_keys(file, val_filter_key)
        if not keys:
            keys = sorted_demo_keys(demos)
        candidates = []
        for demo_key in keys:
            demo = demos[demo_key]
            T = int(demo["actions"].shape[0])
            lo = n_obs_steps - 1
            hi = max(lo, T - horizon - 1)
            if hi <= lo:
                continue
            # Use a few evenly-spaced starts per demo to avoid only measuring
            # the initial reset state.
            for t in np.linspace(lo, hi, num=min(4, hi - lo + 1), dtype=int):
                candidates.append((demo_key, int(t)))
        if not candidates:
            raise RuntimeError(f"no valid starts in {dataset_path}")
        chosen = rng.choice(len(candidates), size=min(n_starts, len(candidates)), replace=False)
        for idx in chosen:
            demo_key, t = candidates[int(idx)]
            demo = demos[demo_key]
            starts.append(
                {
                    "demo_key": demo_key,
                    "t": int(t),
                    "state": demo["states"][t].astype(np.float64),
                    "obs_hist": obs_history(demo, obs_keys, t, n_obs_steps),
                }
            )
    return starts


def create_single_env(dataset_path: Path, obs_keys: list[str], abs_action: bool) -> RobomimicLowdimWrapper:
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    env_meta = make_robosuite_metadata_compatible(env_meta)
    if abs_action:
        set_absolute_controller_config(env_meta["env_kwargs"]["controller_configs"])
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    return RobomimicLowdimWrapper(env=env, obs_keys=obs_keys, init_state=None)


def undo_abs_action(action: np.ndarray, transformer: RotationTransformer | None) -> np.ndarray:
    if transformer is None:
        return action
    raw_shape = action.shape
    work = action
    if work.shape[-1] == 20:
        work = work.reshape(-1, 2, 10)
    d_rot = work.shape[-1] - 4
    pos = work[..., :3]
    rot = work[..., 3 : 3 + d_rot]
    gripper = work[..., [-1]]
    rot = transformer.inverse(rot)
    out = np.concatenate([pos, rot, gripper], axis=-1)
    if raw_shape[-1] == 20:
        out = out.reshape(*raw_shape[:-1], 14)
    return out


def predict_first_action(policy, obs_hist: np.ndarray, device: torch.device, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    obs = torch.from_numpy(obs_hist[None].astype(np.float32)).to(device)
    with torch.no_grad():
        result = policy.predict_action({"obs": obs})
    return result["action"][0, 0].detach().cpu().numpy().astype(np.float32)


def normalized_mse(a: np.ndarray, b: np.ndarray, obs_std: np.ndarray) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) / obs_std.astype(np.float32)
    return float(np.mean(diff**2))


def select_checkpoints(metrics: pd.DataFrame, max_groups: int) -> pd.DataFrame:
    rows = []
    for task, dataset_type, seed in FOCUS_GROUPS[:max_groups]:
        sub = metrics[
            (metrics["policy"] == "diffusion_policy")
            & (metrics["task"] == task)
            & (metrics["dataset_type"] == dataset_type)
            & (metrics["seed"] == seed)
        ].copy()
        sub = sub[sub["checkpoint_path"].map(lambda p: Path(str(p)).exists())]
        if sub.empty:
            continue
        sub = sub.sort_values("test_mean_score")
        picks = [sub.iloc[0], sub.iloc[len(sub) // 2], sub.iloc[-1]]
        for tag, row in zip(["low", "mid", "high"], picks):
            out = row.to_dict()
            out["calibration_pick"] = tag
            rows.append(out)
    return pd.DataFrame(rows)


def run_checkpoint(row: pd.Series, device: torch.device, args: argparse.Namespace) -> list[dict]:
    ckpt_path = ROOT / str(row["checkpoint_path"])
    if not ckpt_path.exists():
        ckpt_path = Path(str(row["checkpoint_path"]))
    cfg, policy, dataset, state_key = load_policy_and_dataset(ckpt_path, device=device, use_ema=True)
    dataset_path = ROOT / str(cfg.task.dataset.dataset_path)
    if not dataset_path.exists():
        dataset_path = Path(str(cfg.task.dataset.dataset_path))
    obs_keys = list(cfg.task.dataset.obs_keys)
    n_obs_steps = int(cfg.n_obs_steps)
    abs_action = bool(OmegaConf.select(cfg, "task.dataset.abs_action", default=False))
    val_filter_key = OmegaConf.select(cfg, "task.dataset.val_filter_key", default=None)

    dynamics = build_mlp_dynamics_ensemble(
        dataset,
        n_models=args.dyn_models,
        max_transitions=args.dyn_max_transitions,
        seed=args.dyn_seed,
        device=device,
        hidden_dim=args.dyn_hidden_dim,
        epochs=args.dyn_epochs,
        batch_size=args.dyn_batch_size,
        lr=args.dyn_lr,
        task_name=str(cfg.task.task_name),
        progress_mode="semantic",
        episode_split="train",
    )
    env = create_single_env(dataset_path, obs_keys, abs_action=abs_action)
    transformer = RotationTransformer("axis_angle", "rotation_6d") if abs_action else None
    starts = sample_validation_starts(
        dataset_path=dataset_path,
        obs_keys=obs_keys,
        n_obs_steps=n_obs_steps,
        horizon=args.horizon,
        n_starts=args.starts,
        val_filter_key=val_filter_key,
        seed=args.start_seed,
    )

    out_rows = []
    for start_idx, start in enumerate(starts):
        env.init_state = start["state"]
        sim_obs = env.reset().astype(np.float32)
        sim_hist = start["obs_hist"].copy().astype(np.float32)
        sim_hist[-1] = sim_obs
        imag_hist = sim_hist.copy()

        step_mse = []
        recursive_mse = []
        one_step_unc = []
        imag_support = []
        sim_support = []
        for k in range(args.horizon):
            seed = args.action_seed + 1000 * start_idx + k
            sim_action = predict_first_action(policy, sim_hist, device=device, seed=seed)
            pred_next_from_sim, unc = dynamics.predict(sim_hist[-1][None], sim_action[None])
            env_action = undo_abs_action(sim_action[None], transformer=transformer)[0]
            next_sim_obs, _, _, _ = env.step(env_action)
            next_sim_obs = next_sim_obs.astype(np.float32)

            imag_action = predict_first_action(policy, imag_hist, device=device, seed=seed)
            next_imag_obs, _ = dynamics.predict(imag_hist[-1][None], imag_action[None])
            next_imag_obs = next_imag_obs[0]

            step_mse.append(normalized_mse(pred_next_from_sim[0], next_sim_obs, dynamics.obs_std))
            recursive_mse.append(normalized_mse(next_imag_obs, next_sim_obs, dynamics.obs_std))
            one_step_unc.append(float(unc[0]))
            imag_dist, _ = dynamics.nearest_support(next_imag_obs[None])
            sim_dist, _ = dynamics.nearest_support(next_sim_obs[None])
            imag_support.append(float(imag_dist[0]))
            sim_support.append(float(sim_dist[0]))

            sim_hist = np.concatenate([sim_hist[1:], next_sim_obs[None]], axis=0)
            imag_hist = np.concatenate([imag_hist[1:], next_imag_obs[None]], axis=0)

        out_rows.append(
            {
                "checkpoint_path": str(ckpt_path.relative_to(ROOT)) if ckpt_path.is_relative_to(ROOT) else str(ckpt_path),
                "run_dir": str(row["run_dir"]),
                "task": str(row["task"]),
                "dataset_type": str(row["dataset_type"]),
                "seed": int(row["seed"]),
                "epoch": int(row["epoch"]),
                "success": float(row["test_mean_score"]),
                "calibration_pick": str(row["calibration_pick"]),
                "state_key": state_key,
                "start_idx": int(start_idx),
                "demo_key": str(start["demo_key"]),
                "demo_t": int(start["t"]),
                "horizon": int(args.horizon),
                "one_step_policy_induced_mse": float(np.mean(step_mse)),
                "one_step_policy_induced_final_mse": float(step_mse[-1]),
                "recursive_imagined_vs_sim_mse": float(np.mean(recursive_mse)),
                "recursive_imagined_vs_sim_final_mse": float(recursive_mse[-1]),
                "one_step_uncertainty": float(np.mean(one_step_unc)),
                "imagined_support_drift": float(np.mean(imag_support)),
                "sim_support_drift": float(np.mean(sim_support)),
                "dyn_train_transitions": int(dynamics.n_dynamics_transitions),
            }
        )
    return out_rows


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ckpt = (
        rows.groupby(["task", "dataset_type", "seed", "epoch", "success", "calibration_pick"])
        .agg(
            starts=("start_idx", "count"),
            one_step_policy_induced_mse=("one_step_policy_induced_mse", "mean"),
            recursive_imagined_vs_sim_mse=("recursive_imagined_vs_sim_mse", "mean"),
            one_step_uncertainty=("one_step_uncertainty", "mean"),
            imagined_support_drift=("imagined_support_drift", "mean"),
            sim_support_drift=("sim_support_drift", "mean"),
        )
        .reset_index()
    )
    group_rows = []
    for keys, sub in ckpt.groupby(["task", "dataset_type"]):
        task, dataset_type = keys
        group_rows.append(
            {
                "task": task,
                "dataset_type": dataset_type,
                "checkpoints": int(len(sub)),
                "mean_one_step_policy_induced_mse": float(sub["one_step_policy_induced_mse"].mean()),
                "mean_recursive_imagined_vs_sim_mse": float(sub["recursive_imagined_vs_sim_mse"].mean()),
                "support_drift_spearman_imag_vs_sim": safe_spearman(sub["imagined_support_drift"], sub["sim_support_drift"]),
                "one_step_mse_vs_success_spearman": safe_spearman(sub["one_step_policy_induced_mse"], sub["success"]),
                "recursive_mse_vs_success_spearman": safe_spearman(sub["recursive_imagined_vs_sim_mse"], sub["success"]),
                "uncertainty_vs_one_step_mse_spearman": safe_spearman(sub["one_step_uncertainty"], sub["one_step_policy_induced_mse"]),
            }
        )
    return ckpt, pd.DataFrame(group_rows)


def write_markdown(path: Path, checkpoint_summary: pd.DataFrame, group_summary: pd.DataFrame) -> None:
    lines = [
        "# Policy-Induced Simulator Calibration",
        "",
        "This diagnostic compares learned-dynamics predictions with short real simulator transitions under already-trained Diffusion Policy checkpoints. It is not new policy training.",
        "",
        "## Group Summary",
        "",
        group_summary.to_markdown(index=False, floatfmt=".4f") if not group_summary.empty else "_No rows._",
        "",
        "## Checkpoint Summary",
        "",
        checkpoint_summary.to_markdown(index=False, floatfmt=".4f") if not checkpoint_summary.empty else "_No rows._",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-groups", type=int, default=2)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=20260707)
    parser.add_argument("--action-seed", type=int, default=707)
    parser.add_argument("--dyn-models", type=int, default=3)
    parser.add_argument("--dyn-seed", type=int, default=0)
    parser.add_argument("--dyn-max-transitions", type=int, default=12000)
    parser.add_argument("--dyn-hidden-dim", type=int, default=256)
    parser.add_argument("--dyn-epochs", type=int, default=4)
    parser.add_argument("--dyn-batch-size", type=int, default=1024)
    parser.add_argument("--dyn-lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.chdir(ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    metrics = pd.read_csv(args.input)
    selected = select_checkpoints(metrics, max_groups=args.max_groups)
    selected.to_csv(output_dir / "selected_checkpoints.csv", index=False)
    if selected.empty:
        raise RuntimeError("no checkpoints selected")

    rows = []
    for _, row in selected.iterrows():
        print(json.dumps({"running": row["checkpoint_path"], "success": row["test_mean_score"]}))
        rows.extend(run_checkpoint(row, device=device, args=args))
        pd.DataFrame(rows).to_csv(output_dir / "policy_induced_sim_calibration_partial.csv", index=False)

    raw = pd.DataFrame(rows)
    checkpoint_summary, group_summary = summarize(raw)
    raw.to_csv(output_dir / "policy_induced_sim_calibration_raw.csv", index=False)
    checkpoint_summary.to_csv(output_dir / "policy_induced_sim_calibration_by_checkpoint.csv", index=False)
    group_summary.to_csv(output_dir / "policy_induced_sim_calibration_summary.csv", index=False)
    write_markdown(output_dir / "policy_induced_sim_calibration_summary.md", checkpoint_summary, group_summary)
    print(f"wrote policy-induced simulator calibration to {output_dir}")


if __name__ == "__main__":
    main()
