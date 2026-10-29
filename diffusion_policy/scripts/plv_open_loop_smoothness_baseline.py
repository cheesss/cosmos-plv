#!/usr/bin/env python3
"""Compute an open-loop policy smoothness baseline on expert validation states."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.scripts.plv_lowdim_checkpoint_metrics import (  # noqa: E402
    checkpoint_metadata,
    discover_checkpoints,
    evaluate_action_metrics,
    load_policy_and_dataset,
    stack_samples,
    summarize_logs,
)


def evaluate_open_loop_smoothness(policy, cfg, dataset, device: torch.device,
        max_samples: int, horizon: int):
    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        val_dataset = dataset
    batch = stack_samples(val_dataset, max_samples)
    batch = dict_apply(batch, lambda x: x.to(device))
    obs_seq = batch["obs"]
    n_obs_steps = int(cfg.n_obs_steps)
    seq_horizon = int(obs_seq.shape[1])
    if seq_horizon < n_obs_steps:
        return {
            "open_loop_smoothness": "",
            "open_loop_action_delta": "",
            "open_loop_action_l2": "",
            "open_loop_horizon": int(horizon),
            "open_loop_steps": 0,
            "open_loop_eval_samples": int(obs_seq.shape[0]),
        }

    actions = []
    n_actions = min(int(horizon), seq_horizon - n_obs_steps + 1)
    if hasattr(policy, "reset"):
        policy.reset()
    with torch.no_grad():
        for t in range(n_obs_steps - 1, n_obs_steps - 1 + n_actions):
            obs_hist = obs_seq[:, t - n_obs_steps + 1:t + 1]
            result = policy.predict_action({"obs": obs_hist})
            actions.append(result["action"][:, 0].detach().cpu().numpy().astype(np.float32))
    action_arr = np.stack(actions, axis=1)
    if action_arr.shape[1] >= 3:
        jerk = action_arr[:, 2:] - 2 * action_arr[:, 1:-1] + action_arr[:, :-2]
        smoothness = float(np.mean(jerk ** 2))
    else:
        smoothness = ""
    if action_arr.shape[1] >= 2:
        delta = action_arr[:, 1:] - action_arr[:, :-1]
        action_delta = float(np.mean(delta ** 2))
    else:
        action_delta = ""
    return {
        "open_loop_smoothness": smoothness,
        "open_loop_action_delta": action_delta,
        "open_loop_action_l2": float(np.linalg.norm(action_arr, axis=-1).mean()),
        "open_loop_horizon": int(horizon),
        "open_loop_steps": int(action_arr.shape[1]),
        "open_loop_eval_samples": int(action_arr.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="Optional common action-sampling seed reset before every checkpoint.",
    )
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--skip-latest", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for ckpt_path in discover_checkpoints([Path(x) for x in args.roots], skip_latest=args.skip_latest):
        cfg, policy, dataset, state_key = load_policy_and_dataset(
            ckpt_path,
            device=device,
            use_ema=args.use_ema,
        )
        if args.policy_seed is not None:
            random.seed(args.policy_seed)
            np.random.seed(args.policy_seed)
            torch.manual_seed(args.policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.policy_seed)
        _, by_epoch = summarize_logs(ckpt_path.parent.parent / "logs.json.txt")
        row = checkpoint_metadata(ckpt_path, cfg, by_epoch)
        row["state_key"] = state_key
        row["policy_eval_seed"] = (
            "" if args.policy_seed is None else int(args.policy_seed)
        )
        row.update(evaluate_action_metrics(
            policy=policy,
            cfg=cfg,
            dataset=dataset,
            device=device,
            max_samples=args.max_eval_samples,
        ))
        row.update(evaluate_open_loop_smoothness(
            policy=policy,
            cfg=cfg,
            dataset=dataset,
            device=device,
            max_samples=args.max_eval_samples,
            horizon=args.horizon,
        ))
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} open-loop smoothness rows to {output_path}")


if __name__ == "__main__":
    main()
