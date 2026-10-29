#!/usr/bin/env python3
"""y3_visual_robustness Step 1+2 (Environment A): replay recorded EXECUTED actions and
re-render the episode from other cameras.  EXPLORATORY.

Per episode: env built exactly like cosmos_plv.rollout_recorder (_make_env with the
checkpoint's dataset / obs keys / absolute OSC controller, render_offscreen=True,
448x448), env.seed(env_seed); env.reset(); then one env.step per row of the npz
``action`` array.  The privileged probe (same as the recorder) is sampled after every
step and compared with the recorded eef_pos / object_pos / grasped / success.
Frames are rendered at t % 2 == 0 (same steps as the original 10-fps video) for each
requested camera, after making the EGL context current (rollout_recorder._render).

Outputs (scratch): <out>/<camera>/<ckpt>/seed<seed>.mp4 (write_mp4, 10 fps, same
encoder settings as the recorder), <camera>/<ckpt>/seed<seed>_last.png and 8 keyframe
PNGs decoded from that mp4, and <out>/replay_log.jsonl.

  bash -lc 'source scripts/hackathon/env.sh && python scripts/hackathon/explore/y3_replay_render.py \
      --limit 4 --cameras agentview --tag check'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json"))
    ap.add_argument("--out", default=str(ROOT / "data/outputs/hackathon_explore/y3_visual_robustness/render"))
    ap.add_argument("--cameras", default="agentview,frontview,birdview,robot0_eye_in_hand")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--no-save", action="store_true", help="determinism check only, no files except the log")
    args = ap.parse_args()

    import imageio.v2 as imageio

    from cosmos_plv.rollout_recorder import _make_env
    from cosmos_plv.sim_probe import make_probe
    from cosmos_plv.video_recorder import read_mp4, write_mp4

    cams = [c for c in args.cameras.split(",") if c]
    eps = json.loads(Path(args.episodes).read_text())["episodes"]
    if args.limit:
        eps = eps[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"replay_log_{args.tag}.jsonl"
    for ep in eps:
        t0 = time.time()
        task = ep["task"]
        rec = np.load(ROOT / ep["episode_npz"])
        acts = rec["action"]
        dataset_path = ROOT / f"data/robomimic/datasets/{task}/ph/low_dim_abs.hdf5"
        env = _make_env(dataset_path, OBS_KEYS, True, True, (448, 448), "agentview")
        env.seed(int(ep["env_seed"]))
        env.reset()
        rs = env.env.env
        probe = make_probe(task, rs)
        ctx = getattr(rs.sim, "_render_context_offscreen", None)

        def render_all():
            if ctx is not None:
                ctx.gl_ctx.make_current()
            return {c: np.asarray(env.env.render(mode="rgb_array", height=448, width=448, camera_name=c)).copy()
                    for c in cams}

        tr = {k: [v] for k, v in probe.sample().items()}
        frames = {c: [] for c in cams}
        f0 = render_all()
        for c in cams:
            frames[c].append(f0[c])
        for k in range(acts.shape[0]):
            env.step(acts[k])
            s = probe.sample()
            for key, val in s.items():
                tr[key].append(val)
            if (k + 1) % 2 == 0:
                fr = render_all()
                for c in cams:
                    frames[c].append(fr[c])
        tr = {k: np.asarray(v) for k, v in tr.items()}
        eef_err = float(np.max(np.abs(tr["eef_pos"].astype(np.float32) - rec["eef_pos"])))
        obj_err = float(np.max(np.abs(tr["object_pos"].astype(np.float32) - rec["object_pos"])))
        grasp_agree = float(np.mean(tr["grasped"] == rec["grasped"]))
        succ_step_agree = float(np.mean(tr["success"] == rec["success"]))
        succ_replay = bool(tr["success"].any())
        # pixel check vs the original (H.264-decoded) agentview video
        pix = None
        if "agentview" in cams:
            orig = read_mp4(ROOT / ep["video"])
            n = min(len(orig), len(frames["agentview"]))
            idx = sorted({0, n // 2, n - 1})
            pix = {"orig_frames": len(orig), "replay_frames": len(frames["agentview"]),
                   "psnr_raw_vs_orig_mp4": [round(psnr(frames["agentview"][i], orig[i]), 2) for i in idx],
                   "max_abs_raw_vs_orig_mp4": [int(np.max(np.abs(frames["agentview"][i].astype(int) - orig[i].astype(int))))
                                               for i in idx]}
        saved = {}
        if not args.no_save:
            for c in cams:
                d = out / c / ep["checkpoint_id"]
                d.mkdir(parents=True, exist_ok=True)
                mp4 = d / f"seed{ep['env_seed']}.mp4"
                write_mp4(mp4, frames[c], fps=10)
                dec = read_mp4(mp4)
                kidx = [int(round(x)) for x in np.linspace(0, len(dec) - 1, 8)]
                for j in kidx:
                    imageio.imwrite(d / f"seed{ep['env_seed']}_k{j:03d}.png", dec[j])
                imageio.imwrite(d / f"seed{ep['env_seed']}_last.png", dec[-1])
                saved[c] = {"mp4": str(mp4.relative_to(ROOT)), "last_png": str((d / f"seed{ep['env_seed']}_last.png").relative_to(ROOT)),
                            "n_frames": len(dec)}
                if c == "agentview" and pix is not None:
                    n = min(len(dec), len(orig))
                    pix["psnr_reencoded_vs_orig_mp4_last"] = round(psnr(dec[n - 1], orig[n - 1]), 2)
        row = {"checkpoint_id": ep["checkpoint_id"], "env_seed": ep["env_seed"], "task": task,
               "n_actions": int(acts.shape[0]), "max_abs_eef_err_m": eef_err, "max_abs_obj_err_m": obj_err,
               "grasped_step_agreement": grasp_agree, "success_step_agreement": succ_step_agree,
               "success_recorded": bool(ep["success"]), "success_replay": succ_replay,
               "success_match": succ_replay == bool(ep["success"]), "pixel_check": pix, "saved": saved,
               "wall_s": round(time.time() - t0, 2)}
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("checkpoint_id", "env_seed", "max_abs_eef_err_m", "max_abs_obj_err_m",
                                              "success_match", "wall_s")} | {"pix": pix}), flush=True)
        # NOTE: rs.close() here breaks the NEXT env's offscreen render (frames then differ
        # from the recorded video, PSNR ~14 dB, found in the y3 check run); envs are only
        # dereferenced, as in a recorder batch where all envs stay open.
        del env, rs, probe


if __name__ == "__main__":
    main()
