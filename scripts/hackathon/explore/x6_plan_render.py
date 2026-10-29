#!/usr/bin/env python3
"""x6_visual_plan_critic, stage 1 (Environment A, EXPLORATORY).

Rollout-free "visual plan" probe: for every checkpoint in the pool and a few
env seeds (210000.. by default, disjoint from the eval seeds 200000.. and the
training-time test seeds 100000..), reset the robomimic env, run ONE policy
call on the initial observation (obs history = 2 copies of the first obs) and
render the predicted 16-step absolute EEF plan into the agentview image.

Variant A ("plan1"): only the first policy call, drawn on the initial frame.
Variant B ("plan3"): a short REAL closed-loop rollout of 3 chunks (24 sim
steps); the 3 predicted chunks are drawn on the initial frame.

Saves per (checkpoint, seed): an .npz (plans, eef/object positions, projected
pixels, camera matrix, timings) and PNG overlays.  Re-runnable: existing
outputs are skipped, new checkpoints are picked up.

  bash -lc 'source scripts/hackathon/env.sh && python scripts/hackathon/explore/x6_plan_render.py'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

OUT_DEFAULT = ROOT / "data/outputs/hackathon_explore/x6_visual_plan_critic/plans"
H = W = 448


def grad_color(u: float):
    """dark purple -> blue -> light yellow (no red, the can is red)."""
    stops = [(0.0, (40, 0, 90)), (0.5, (0, 150, 255)), (1.0, (255, 255, 120))]
    for (u0, c0), (u1, c1) in zip(stops[:-1], stops[1:]):
        if u <= u1:
            a = (u - u0) / max(u1 - u0, 1e-9)
            return tuple(int(round(c0[i] + a * (c1[i] - c0[i]))) for i in range(3))
    return stops[-1][1]


def draw_paths(img: np.ndarray, paths, t_index, t_max: float) -> Image.Image:
    """paths: list of (N,2) pixel (row,col) arrays; t_index: list of (N,) times."""
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    for px, ts in zip(paths, t_index):
        pts = [(int(c), int(r)) for r, c in px]
        for k in range(len(pts) - 1):
            d.line([pts[k], pts[k + 1]], fill=grad_color(ts[k] / t_max), width=2)
        for k, (x, y) in enumerate(pts):
            col = grad_color(ts[k] / t_max)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col, outline=(255, 255, 255))
    return im


def project(P: np.ndarray, pts: np.ndarray, flip_rows: bool):
    """World points (N,3) -> float pixel (row, col) + in-frame mask."""
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    pix = (P @ hom.T).T
    u = pix[:, 0] / pix[:, 2]
    v = pix[:, 1] / pix[:, 2]
    row, col = v, u
    if flip_rows:
        row = (H - 1) - row
    inside = (row >= 0) & (row <= H - 1) & (col >= 0) & (col <= W - 1) & (pix[:, 2] > 0)
    return np.stack([row, col], axis=1), inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "data/outputs/hackathon_pool"))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seeds", default="210000:210005")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--sampling-seed-offset", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--filter", default="", help="substring filter on checkpoint_id")
    ap.add_argument("--max-ckpts", type=int, default=0)
    ap.add_argument("--chunks-fine", type=int, default=1,
                    help="chunks for non-eval-epoch checkpoints (saves GPU time)")
    ap.add_argument("--flip-rows", type=int, default=0,
                    help="0 verified on debug images: robomimic's [::-1] render matches robosuite pixel rows")
    args = ap.parse_args()

    import torch
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    from diffusion_policy.scripts.plv_chunk_aware_checkpoint_metrics import set_sampling_seed
    from diffusion_policy.scripts.plv_policy_induced_sim_calibration import undo_abs_action
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    from cosmos_plv.checkpoint_adapter import discover_checkpoints, load_policy
    from cosmos_plv.rollout_recorder import _make_env, _render
    from cosmos_plv.sim_probe import make_probe

    a, b = (int(x) for x in args.seeds.split(":"))
    seeds = list(range(a, b))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = discover_checkpoints(Path(args.pool))
    recs = [r for r in recs if args.filter in r.checkpoint_id]
    # interleave runs, coarse epochs first, so partial runs cover every run
    def is_coarse(r):
        return (r.task == "can" and r.epoch % 20 == 0) or (r.task == "square" and r.epoch % 30 == 0)

    def prio(r):
        return (0 if is_coarse(r) else 1, r.epoch, r.checkpoint_id)
    recs.sort(key=prio)
    todo = [r for r in recs if not all((out / r.checkpoint_id / f"seed{s}.npz").is_file() for s in seeds)]
    if args.max_ckpts:
        todo = todo[: args.max_ckpts]
    print(f"[x6] {len(recs)} checkpoints in pool, {len(todo)} to do, seeds {seeds}", flush=True)

    envs, cache, debug_done = {}, {}, set()
    log_path = out / "render_log.jsonl"
    for r in todo:
        t_load = time.time()
        try:
            task_cfg, policy, _ds, ex = load_policy(ROOT / r.checkpoint_path, args.device, cache)
        except Exception as e:  # checkpoint still being written etc.
            print(f"[x6] load failed {r.checkpoint_id}: {type(e).__name__}: {e}", flush=True)
            continue
        load_s = time.time() - t_load
        task = r.task
        transformer = RotationTransformer("axis_angle", "rotation_6d") if ex["abs_action"] else None
        n_obs, n_act, n_lat = ex["n_obs_steps"], ex["n_action_steps"], ex["n_latency_steps"]
        if task not in envs:
            env = _make_env(ROOT / ex["dataset_path"], ex["obs_keys"], ex["abs_action"], True, (H, W), "agentview")
            envs[task] = env
        env = envs[task]
        rs_env = env.env.env
        cdir = out / r.checkpoint_id
        cdir.mkdir(parents=True, exist_ok=True)
        n_chunks = args.chunks if is_coarse(r) else args.chunks_fine
        for seed in seeds:
            npz_path = cdir / f"seed{seed}.npz"
            if npz_path.is_file():
                continue
            t0 = time.time()
            env.seed(seed)
            obs = env.reset()
            probe = make_probe(task, rs_env)
            s0 = probe.sample()
            img0 = _render(env).copy()
            P = get_camera_transform_matrix(rs_env.sim, "agentview", H, W)
            reset_s = time.time() - t0
            hist = [obs, obs]
            plans, plan_t, eef_tr, obj_tr, call_s = [], [], [s0["eef_pos"]], [s0["object_pos"]], []
            if hasattr(policy, "reset"):
                policy.reset()
            set_sampling_seed(seed + args.sampling_seed_offset)
            t = 0
            for c in range(n_chunks):
                obs_hist = np.stack(hist[-n_obs:])[None].astype(np.float32)
                tc = time.time()
                with torch.no_grad():
                    o = policy.predict_action({"obs": torch.from_numpy(obs_hist).to(args.device)})
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                call_s.append(time.time() - tc)
                ap_ = o["action_pred"].detach().cpu().numpy()[0]
                plans.append(ap_)
                plan_t.append(t)
                if c == n_chunks - 1:
                    break
                act = undo_abs_action(o["action"].detach().cpu().numpy()[:, n_lat:], transformer)[0]
                for k in range(act.shape[0]):
                    obs, _r, _d, _i = env.step(act[k])
                    hist.append(obs)
                    s = probe.sample()
                    eef_tr.append(s["eef_pos"])
                    obj_tr.append(s["object_pos"])
                    t += 1
            plans = np.asarray(plans, dtype=np.float32)  # (chunks,16,10)
            # projections
            pos = plans[:, :, :3]
            pix1, in1 = project(P, pos[0].astype(np.float64), bool(args.flip_rows))
            pixall, inall = project(P, pos.reshape(-1, 3).astype(np.float64), bool(args.flip_rows))
            pix_eef, _ = project(P, np.asarray([s0["eef_pos"]], dtype=np.float64), bool(args.flip_rows))
            pix_obj, _ = project(P, np.asarray([s0["object_pos"]], dtype=np.float64), bool(args.flip_rows))
            # images
            Image.fromarray(img0).save(cdir / f"seed{seed}_raw.png")
            tA = np.arange(pos.shape[1], dtype=float)
            draw_paths(img0, [pix1], [tA], float(pos.shape[1] - 1)).save(cdir / f"seed{seed}_plan1.png")
            t_idx = [np.asarray(plan_t[c] + np.arange(pos.shape[1]) - (n_obs - 1), dtype=float) for c in range(len(plans))]
            tmin = min(x.min() for x in t_idx)
            t_idx = [x - tmin for x in t_idx]
            tmax = max(x.max() for x in t_idx)
            if len(plans) > 1:
                draw_paths(img0, [pixall[c * pos.shape[1]:(c + 1) * pos.shape[1]] for c in range(len(plans))],
                           t_idx, float(tmax)).save(cdir / f"seed{seed}_plan{len(plans)}.png")
            if task not in debug_done:
                im = Image.fromarray(img0).convert("RGB")
                d = ImageDraw.Draw(im)
                for (rr, cc), col in ((pix_eef[0], (0, 255, 0)), (pix_obj[0], (255, 0, 255))):
                    d.line([(cc - 10, rr), (cc + 10, rr)], fill=col, width=2)
                    d.line([(cc, rr - 10), (cc, rr + 10)], fill=col, width=2)
                im.save(out / f"debug_projection_{task}_{r.checkpoint_id}_seed{seed}.png")
                debug_done.add(task)
            np.savez_compressed(
                npz_path,
                action_pred=plans, plan_t=np.asarray(plan_t), eef0=np.asarray(s0["eef_pos"]),
                obj0=np.asarray(s0["object_pos"]), eef_traj=np.asarray(eef_tr), obj_traj=np.asarray(obj_tr),
                pix_plan1=pix1, inframe_plan1=in1, pix_all=pixall, inframe_all=inall,
                pix_eef0=pix_eef[0], pix_obj0=pix_obj[0], cam_P=P,
                call_s=np.asarray(call_s), reset_s=reset_s, load_s=load_s,
            )
            rec = {"checkpoint_id": r.checkpoint_id, "task": task, "epoch": r.epoch, "env_seed": seed,
                   "sampling_seed": seed + args.sampling_seed_offset, "load_s": round(load_s, 3),
                   "reset_render_s": round(reset_s, 3), "call_s": [round(x, 3) for x in call_s],
                   "n_chunks": len(plans), "horizon": int(ex["horizon"]), "n_obs_steps": n_obs,
                   "n_action_steps": n_act, "num_inference_steps": int(ex["num_inference_steps"])}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"[x6] {r.checkpoint_id} done (load {load_s:.1f}s)", flush=True)
        del policy
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("[x6] finished", flush=True)


if __name__ == "__main__":
    main()
