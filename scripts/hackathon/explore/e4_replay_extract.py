#!/usr/bin/env python3
"""Experiment 4 (exploratory), step 1 — Environment A.

Replay recorded eval episodes deterministically (same env seed, the executed
absolute OSC actions) and record, at every 20 Hz step, the ACHIEVED end-effector
pose (position + rotation matrix in the world frame), gripper finger positions
and a 256x256 agentview frame. Verifies that the replayed object trajectory
matches the original recording.

Output per episode: data/outputs/hackathon_explore/e4_cosmos_fd/replay/<ckpt>_<seed>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cosmos_plv.rollout_recorder import _make_env, _render  # noqa: E402

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True, help="checkpoint_id:env_seed")
    ap.add_argument("--out", default="data/outputs/hackathon_explore/e4_cosmos_fd/replay")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=160)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for item in a.episodes:
        ck, seed = item.split(":")
        seed = int(seed)
        summ = json.loads((ROOT / f"data/outputs/hackathon_rollouts_eval/{ck}/summary.json").read_text())
        ep = [e for e in summ["episodes"] if e["env_seed"] == seed][0]
        rec = np.load(ROOT / ep["episode_npz"])
        task = ep["task"]
        ds = ROOT / f"data/robomimic/datasets/{task}/ph/low_dim_abs.hdf5"
        env = _make_env(ds, OBS_KEYS, True, True, (a.size, a.size), "agentview")
        env.seed(seed)
        env.reset()
        rs = env.env.env
        robot = rs.robots[0]
        site = robot.eef_site_id["right"]
        pos, mat, grip, frames, obj = [], [], [], [], []

        def sample():
            pos.append(np.array(rs.sim.data.site_xpos[site]))
            mat.append(np.array(rs.sim.data.site_xmat[site]).reshape(3, 3))
            grip.append(np.array(rs.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]))
            frames.append(_render(env))

        sample()
        obj.append(rec["object_pos"][0])
        n = min(a.max_steps, len(rec["action"]))
        for t in range(n):
            env.step(rec["action"][t])
            sample()
        # projection of the achieved EEF / object positions into the rendered image
        from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
        Tcam = get_camera_transform_matrix(rs.sim, "agentview", a.size, a.size)
        eef_px = project_points_from_world_to_camera(np.asarray(pos), Tcam, a.size, a.size)  # (row, col), raw render
        obj_px = project_points_from_world_to_camera(np.asarray(rec["object_pos"][: n + 1]), Tcam, a.size, a.size)
        # object trajectory check against the original recording (same probe as recorder)
        from cosmos_plv.sim_probe import make_probe
        probe = make_probe(task, rs)
        err_final = float(np.abs(probe.object_pos() - rec["object_pos"][n]).max())
        err_eef = float(np.abs(np.asarray(pos)[1:] - rec["eef_pos"][1:n + 1]).max())
        np.savez_compressed(out / f"{ck}_{seed}.npz", eef_pos=np.asarray(pos), eef_mat=np.asarray(mat),
                            gripper_qpos=np.asarray(grip), frames=np.asarray(frames, dtype=np.uint8),
                            eef_px_raw=np.asarray(eef_px), obj_px_raw=np.asarray(obj_px), cam_T=Tcam,
                            action=rec["action"][:n], object_pos=rec["object_pos"][:n + 1],
                            grasped=rec["grasped"][:n + 1], task=task, success=ep["success"],
                            failure_type=ep["failure_type"])
        report.append({"episode": item, "task": task, "steps": n, "max_eef_err_m": err_eef,
                       "final_object_err_m": err_final, "failure_type": ep["failure_type"]})
        print(json.dumps(report[-1]), flush=True)
    (out / "replay_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
