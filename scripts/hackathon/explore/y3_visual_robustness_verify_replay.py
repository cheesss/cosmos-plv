#!/usr/bin/env python3
"""Verifier for y3_visual_robustness Step 1 (replay determinism), CPU ONLY (no renderer).

Independent re-implementation: env built with rollout_recorder._make_env(..., render=False)
(no offscreen context; run with MUJOCO_GL=osmesa so no EGL context can be created), then
env.seed(seed); env.reset(); one env.step per recorded EXECUTED action; the privileged probe
is compared with the npz trace.  Two negative controls on the first episode show the check
has power: (a) one action component perturbed by +1e-3 at step 20, (b) the wrong env seed.

Episodes: every 10th entry of y3_visual_robustness_episodes.json (fixed, outcome-blind).

  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash -lc 'source scripts/hackathon/env.sh && \
     python scripts/hackathon/explore/y3_visual_robustness_verify_replay.py'
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OUT = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness_verification/replay_cpu.jsonl"


def replay(ep, seed=None, perturb=None):
    from cosmos_plv.rollout_recorder import _make_env
    from cosmos_plv.sim_probe import make_probe

    rec = np.load(ROOT / ep["episode_npz"])
    acts = np.array(rec["action"], copy=True)
    if perturb is not None:
        step, dim, eps = perturb
        acts[step, dim] += eps
    env = _make_env(ROOT / f"data/robomimic/datasets/{ep['task']}/ph/low_dim_abs.hdf5", OBS_KEYS, True, False,
                    (448, 448), "agentview")
    env.seed(int(ep["env_seed"] if seed is None else seed))
    env.reset()
    rs = env.env.env
    assert getattr(rs, "has_offscreen_renderer", False) is False
    probe = make_probe(ep["task"], rs)
    tr = {k: [v] for k, v in probe.sample().items()}
    for a in acts:
        env.step(a)
        for k, v in probe.sample().items():
            tr[k].append(v)
    tr = {k: np.asarray(v) for k, v in tr.items()}
    eef = np.abs(tr["eef_pos"].astype(np.float32) - rec["eef_pos"])
    obj = np.abs(tr["object_pos"].astype(np.float32) - rec["object_pos"])
    eef64 = np.abs(tr["eef_pos"].astype(np.float64) - rec["eef_pos"].astype(np.float64))
    return {"checkpoint_id": ep["checkpoint_id"], "env_seed": ep["env_seed"], "task": ep["task"],
            "replay_seed": int(ep["env_seed"] if seed is None else seed), "perturb": perturb,
            "n_actions": int(acts.shape[0]), "shapes_match": tr["eef_pos"].shape == rec["eef_pos"].shape,
            "max_abs_eef_err_m_f32": float(eef.max()), "max_abs_obj_err_m_f32": float(obj.max()),
            "max_abs_eef_err_m_vs_f64_replay": float(eef64.max()),
            "first_step_eef_err_gt_0": int(np.argmax(eef.max(axis=1) > 0)) if eef.max() > 0 else None,
            "grasped_agree": float(np.mean(tr["grasped"] == rec["grasped"])),
            "success_agree": float(np.mean(tr["success"] == rec["success"])),
            "success_recorded": bool(ep["success"]), "success_replay": bool(tr["success"].any())}


def main():
    eps = json.loads((ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json").read_text())["episodes"]
    pick = eps[::10]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(ep, None, None) for ep in pick] + [(pick[0], None, (20, 0, 1e-3)), (pick[0], int(pick[0]["env_seed"]) + 1, None)]
    for ep, seed, pert in jobs:
        t0 = time.time()
        r = replay(ep, seed, pert)
        r["wall_s"] = round(time.time() - t0, 2)
        rows.append(r)
        print(json.dumps(r), flush=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
