"""Action-space conventions and the (blocked) PLV -> Cosmos action adapter.

PLV / Diffusion Policy (robomimic *_lowdim_abs, robosuite 1.5 OSC_POSE, absolute):
    policy output  : 10-D  [pos(3) m, rotation_6d(6), gripper(1)]   (normalized by
                     the dataset normalizer inside the policy; returned unnormalized)
    env action     : 7-D   [pos(3) m, axis-angle(3) rad, gripper(1) in [-1, 1]]
                     absolute target pose of the end effector in the robot base /
                     world frame, 20 Hz control, executed 8 steps per chunk.

Cosmos3-Edge action-conditioned generation expects (model card / cookbook):
    10-D per frame [dpos(3), rot6d(6), gripper(1)] RELATIVE pose between
    consecutive frames inv(T_k) @ T_{k+1} expressed in the end-effector frame
    after an embodiment-specific camera alignment, quantile-normalized to
    [-1, 1] with DOMAIN statistics, at the domain frame rate (e.g. DROID 15 FPS),
    for supported embodiments (Franka + RobotiQ, WidowX, UR, UMI, ...).

This module implements the geometric part of that conversion (absolute OSC
targets -> per-step relative rot6d poses) with round-trip tests.  The remaining
pieces required for a FAITHFUL adapter are NOT available for robosuite:
    * the frame alignment convention between the robosuite end-effector frame and
      the Cosmos domain frame (undocumented),
    * the domain normalization statistics for a robosuite Panda domain (the LIBERO
      domain exists in the framework registry but is not listed as supported by
      the Edge checkpoint),
    * gripper sign/scale (Panda parallel gripper vs RobotiQ in DROID).
Therefore ``ADAPTER_STATUS`` is "blocked" and the evaluation pipeline uses the
visual-diagnostic path (Mode A) only.  Padding 7-D actions to 10-D is never done.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

ADAPTER_STATUS = "blocked"
ADAPTER_BLOCKERS = (
    "robosuite EEF -> Cosmos domain frame alignment undocumented",
    "no domain normalization statistics for robosuite Panda in Cosmos3-Edge",
    "gripper encoding differs (Panda vs RobotiQ)",
    "commanded OSC targets are not achieved poses; achieved poses require simulation",
)

POLICY_ACTION_DIM = 10
ENV_ACTION_DIM = 7
COSMOS_ACTION_DIM = 10


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-12, 1.0, theta)
    k = rotvec / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1).reshape(rotvec.shape[:-1] + (3, 3))
    th = theta[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    R = eye + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return np.where(theta[..., None] < 1e-12, eye, R)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns, row-major flattened (pytorch3d convention: rows of R[:2])."""
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def validate_env_actions(actions: np.ndarray) -> Dict[str, object]:
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != ENV_ACTION_DIM:
        raise ValueError(f"expected (T, {ENV_ACTION_DIM}) absolute OSC actions, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite actions")
    ang = np.linalg.norm(a[:, 3:6], axis=1)
    return {
        "T": int(a.shape[0]),
        "pos_range_m": [float(a[:, :3].min()), float(a[:, :3].max())],
        "max_rotvec_angle_rad": float(ang.max()),
        "gripper_values": sorted(set(np.round(a[:, 6], 3).tolist()))[:10],
    }


def absolute_to_relative_rot6d(actions: np.ndarray) -> np.ndarray:
    """(T,7) absolute targets -> (T-1,10) relative poses inv(T_k) T_{k+1} in the EEF frame.

    Geometry only; does NOT apply any Cosmos domain alignment or normalization.
    """
    validate_env_actions(actions)
    a = np.asarray(actions, dtype=np.float64)
    R = axis_angle_to_matrix(a[:, 3:6])
    p = a[:, :3]
    R_rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])  # R_k^T R_{k+1}
    p_rel = np.einsum("tji,tj->ti", R[:-1], p[1:] - p[:-1])  # R_k^T (p_{k+1}-p_k)
    return np.concatenate([p_rel, matrix_to_rot6d(R_rel), a[1:, 6:7]], axis=1)


def compose_relative(start: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Inverse check: integrate relative poses from an absolute 7-D start action."""
    R = axis_angle_to_matrix(start[3:6])
    p = start[:3].astype(np.float64)
    out = []
    for r in rel:
        p = p + R @ r[:3]
        R = R @ rot6d_to_matrix(r[3:9])
        out.append(np.concatenate([p, R.reshape(-1)]))
    return np.asarray(out)


def to_cosmos_action(actions: np.ndarray) -> np.ndarray:
    """Refuses: see ADAPTER_BLOCKERS.  Kept so callers fail loudly, not silently."""
    raise NotImplementedError(
        "PLV -> Cosmos action conditioning is blocked: " + "; ".join(ADAPTER_BLOCKERS)
    )
