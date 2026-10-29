#!/usr/bin/env python3
"""Experiment 4 (exploratory), step 2 — Environment C (diffusers main, Cosmos3OmniPipeline).

Action-conditioned forward dynamics with nvidia/Cosmos3-Edge on REAL replayed
robosuite episodes. For every (episode, chunk start, condition) the model gets the
REAL frame at the chunk start plus 16 actions and generates 16 future frames.
The same frame and the same diffusion seed are used for every condition, so only
the actions differ (common random numbers).

Action adapter "libero" follows the Cosmos framework's LIBERO training pipeline
(cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py):
  robosuite OSC delta command [dpos/0.05, axisangle(R_goal R_cur^T)/0.5, gripper]
  -> [dpos(3), rot6d(matrix(axisangle(drot_cmd))) column-based (6), gripper01(1)]
  -> quantile normalisation with the bundled `global_raw` stats (q01/q99 -> [-1, 1]).
`libero` is a registered Cosmos3 domain (id 5) that diffusers marks as TODO
(configuration-dependent width); we register width 10 for rotation_space='6d'.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SNAP = Path.home() / ".cache/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/344d602b128d1bbdacb43b08d0a3626f46343e29"
STATS = json.loads((ROOT / "configs/hackathon/cosmos_framework_stats/libero_native_frame_wise_relative_rot6d.json").read_text())["global_raw"]
Q01, Q99 = np.asarray(STATS["q01"], np.float64), np.asarray(STATS["q99"], np.float64)
PROMPTS = {"can": "The robot arm picks up the red soda can from the left bin and places it into the right bin.",
           "square": "The robot arm picks up the square nut and slides it onto the square peg."}
CONDITIONS = ("main", "mirror", "gripflip", "unnorm", "zero", "shuffled", "reversed")


def axisangle_to_matrix(v):
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_to_axisangle(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def delta_commands(d):
    """(T,7) robosuite OSC delta-command equivalent of the executed absolute targets."""
    act, pos, mat = d["action"], d["eef_pos"], d["eef_mat"]
    out = []
    for t in range(len(act)):
        dp = (act[t, :3] - pos[t]) / 0.05
        R_goal = axisangle_to_matrix(act[t, 3:6])
        dr = matrix_to_axisangle(R_goal @ mat[t].T) / 0.5
        out.append(np.concatenate([np.clip(dp, -1, 1), np.clip(dr, -1, 1), act[t, 6:7]]))
    return np.asarray(out)


def libero10(cmd, gripper_flip=False, normalize=True):
    rows = []
    for c in cmd:
        R = axisangle_to_matrix(c[3:6])  # LIBERO treats the command's axis-angle values as radians
        rot6d = np.concatenate([R[:, 0], R[:, 1]])  # column-based (pose_utils rot6d)
        g = (1.0 - c[6]) / 2.0  # env -1 open / +1 close -> dataset 1 open / 0 closed (OpenVLA/LeRobot LIBERO)
        if gripper_flip:
            g = 1.0 - g
        rows.append(np.concatenate([c[:3], rot6d, [g]]))
    x = np.asarray(rows)
    if normalize:
        x = 2.0 * (x - Q01) / np.maximum(Q99 - Q01, 1e-8) - 1.0
    return x


def build(cond, cmd, cmd_other, s):
    seg = cmd[s:s + 16]
    if cond == "main" or cond == "mirror":
        return libero10(seg)
    if cond == "gripflip":
        return libero10(seg, gripper_flip=True)
    if cond == "unnorm":
        return libero10(seg, normalize=False)
    if cond == "zero":
        z = np.zeros_like(seg); z[:, 6] = seg[0, 6]
        return libero10(z)
    if cond == "shuffled":
        return libero10(cmd_other[s:s + 16])
    if cond == "reversed":
        r = seg.copy(); r[:, :6] = -r[:, :6]
        return libero10(r)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="data/outputs/hackathon_explore/e4_cosmos_fd/replay")
    ap.add_argument("--out", default="data/outputs/hackathon_explore/e4_cosmos_fd/gen")
    ap.add_argument("--starts", default="0,16,32,48,64")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    import diffusers.pipelines.cosmos.pipeline_cosmos3_omni as P
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from PIL import Image

    P._EMBODIMENT_TO_RAW_ACTION_DIM.setdefault("libero", 10)  # registered domain id 5; width for rot6d
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "jobs.jsonl"
    done = set()
    if meta_path.exists():
        done = {json.loads(l)["job"] for l in meta_path.read_text().splitlines()}
    eps = sorted((ROOT / a.replay).glob("*.npz"))
    data = {p.stem: dict(np.load(p, allow_pickle=True)) for p in eps}
    cmds = {k: delta_commands(v) for k, v in data.items()}
    names = list(data)
    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(str(SNAP), torch_dtype=torch.bfloat16, enable_safety_checker=False)
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    starts = [int(x) for x in a.starts.split(",")]
    for i, name in enumerate(names):
        d = data[name]
        task = str(d["task"])
        same_task = [n for n in names if str(data[n]["task"]) == task and n != name]
        other = same_task[i % len(same_task)] if same_task else name
        for s in starts:
            if s + 16 >= len(d["frames"]):
                continue
            frame0 = d["frames"][s]
            for cond in a.conditions.split(","):
                job = f"{name}|{s}|{cond}"
                if job in done:
                    continue
                acts = build(cond, cmds[name], cmds[other], s)
                img = np.ascontiguousarray(frame0[:, ::-1]) if cond == "mirror" else frame0
                t1 = time.time()
                r = pipe(prompt=PROMPTS[task], action=CosmosActionCondition(
                    mode="forward_dynamics", chunk_size=16, domain_name="libero", resolution_tier=256,
                    raw_actions=torch.tensor(acts, dtype=torch.float32), image=Image.fromarray(img),
                    view_point="third_person_view"),
                    fps=20, num_inference_steps=a.steps, guidance_scale=1.0,
                    generator=torch.Generator(device="cuda").manual_seed(1000 + s), use_system_prompt=False,
                    output_type="np")
                vid = (np.clip(np.asarray(r.video), 0, 1) * 255).astype(np.uint8) if np.asarray(r.video).dtype != np.uint8 else np.asarray(r.video)
                if cond == "mirror":
                    vid = vid[:, :, ::-1]  # back to the upright convention for comparison
                np.savez_compressed(out / f"{name}__{s:03d}__{cond}.npz", video=vid, actions=acts)
                rec = {"job": job, "episode": name, "start": s, "condition": cond, "task": task,
                       "other_episode": other if cond == "shuffled" else None, "seconds": round(time.time() - t1, 2),
                       "frames": int(vid.shape[0]), "hw": list(vid.shape[1:3]), "seed": 1000 + s,
                       "model": "nvidia/Cosmos3-Edge@344d602b", "domain": "libero(5)", "action_range": [float(acts.min()), float(acts.max())]}
                with open(meta_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({k: rec[k] for k in ("job", "seconds", "frames")}), flush=True)


if __name__ == "__main__":
    main()
