"""Stage B helpers: cheap metrics and execution-matched replanning on REAL feedback.

The paper's offline metrics (Action L1, open-loop smoothness, PLV-1, model-
feedback replanning) are computed by the ORIGINAL scripts in
diffusion_policy/scripts (see scripts/hackathon/compute_baseline_metrics.sh).

This module adds one diagnostic that only becomes available once real rollouts
are recorded (rollout_recorder.py stores every predicted chunk): the same
execution-matched replanning statistics, but where the replacement plan is
conditioned on the TRUE simulator observation after executing the old chunk.
Alignment (paper, plv_chunk_aware_checkpoint_metrics.py:193-213):
    plan index j of a call made at step t predicts the action for step t + j - offset
    offset = n_obs_steps - 1 = 1, executed slice [offset, offset + n_act) = [1, 9)
    old plan suffix [offset + n_act, horizon) = [9, 16)  vs
    new plan prefix [offset, horizon - n_act)    = [1, 8)     (L = 7 steps)
    boundary jump: last executed old action (index 8) vs first new action (index 1)
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def replan_alignment(horizon: int, n_obs_steps: int, n_action_steps: int) -> Dict[str, object]:
    off = n_obs_steps - 1
    old = (off + n_action_steps, horizon)
    new = (off, horizon - n_action_steps)
    L = old[1] - old[0]
    if L <= 0 or (new[1] - new[0]) != L:
        raise ValueError("no replanning overlap for this chunking")
    return {"offset": off, "old_slice": old, "new_slice": new, "overlap": L,
            "boundary_old_index": off + n_action_steps - 1, "boundary_new_index": off}


def replan_metrics_from_plans(action_pred: np.ndarray, plan_t: np.ndarray, horizon: int,
                              n_obs_steps: int, n_action_steps: int,
                              action_scale: np.ndarray | None = None) -> Dict[str, float]:
    """action_pred: (n_calls, horizon, D) consecutive plans of ONE episode."""
    ap = np.asarray(action_pred, dtype=np.float64)
    if ap.ndim != 3 or ap.shape[1] != horizon:
        raise ValueError(f"expected (n, {horizon}, D), got {ap.shape}")
    al = replan_alignment(horizon, n_obs_steps, n_action_steps)
    scale = np.ones(ap.shape[-1]) if action_scale is None else np.asarray(action_scale, dtype=np.float64)
    overlaps, jumps = [], []
    for c in range(len(ap) - 1):
        if int(plan_t[c + 1]) - int(plan_t[c]) != n_action_steps:
            continue  # not a consecutive executed chunk (episode end)
        old = ap[c, al["old_slice"][0]: al["old_slice"][1]] / scale
        new = ap[c + 1, al["new_slice"][0]: al["new_slice"][1]] / scale
        overlaps.append(np.abs(old - new).mean())
        jumps.append(np.abs(ap[c, al["boundary_old_index"]] / scale - ap[c + 1, al["boundary_new_index"]] / scale).mean())
    return {
        "sim_replan_overlap_l1": float(np.mean(overlaps)) if overlaps else float("nan"),
        "sim_boundary_jump_l1": float(np.mean(jumps)) if jumps else float("nan"),
        "sim_replan_pairs": len(overlaps),
    }


def executed_smoothness(actions: np.ndarray) -> float:
    """Mean squared second difference of executed 7-D absolute actions (position + gripper)."""
    a = np.asarray(actions, dtype=np.float64)
    if len(a) < 3:
        return float("nan")
    cols = [0, 1, 2, 6]
    d2 = a[2:, cols] - 2 * a[1:-1, cols] + a[:-2, cols]
    return float((d2 ** 2).sum(axis=1).mean())
