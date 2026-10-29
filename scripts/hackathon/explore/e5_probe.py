#!/usr/bin/env python3
"""Experiment 4b (exploratory) — Environment C. Probes after the E4 null result.

E4 sent actions through domain_name="libero" (id 5). A CPU read of the checkpoint
(scripts/hackathon/explore/e5_domain_weights.py) shows that slot is untrained in
nvidia/Cosmos3-Edge@344d602b: zero biases and initialization-level weights. Only
slots 1,2,3,6,7,8,12,13,15,20 are trained (the model card's embodiment list).

Stages (same diffusion seed inside a unit, so only the actions differ):
  S0  harness positive control: the model card's UMI forward-dynamics example
      (main / negated translation / hold / other chunk; seeds 0-2). Must reproduce
      the card's own output and respond to the actions more than to the seed.
  S1  trained Franka slot 8: synthetic constant-velocity sweeps (+-x, +-y, +-z of
      the action frame, 1 cm per step, and hold) on 4 real robomimic frames, in two
      encodings that share slot 8: D_raw (droid_lerobot, unnormalized, 15 fps) and
      R_q (robomind-franka, quantile-normalized with its stats, 10 fps).
  S2  untrained-slot control: the same kind of sweep in the LIBERO encoding sent to
      slot 5 (libero) and slot 9 (another untrained slot). If both are random
      projections, their responses should not agree with each other.
  S3  slot 8 on the real replays (main / zero / reversed / gripflip / seed copy),
      with the action frame fitted in S1. Run only if S1 passes.
Output: data/outputs/hackathon_explore/e5_probe/<stage>/<job>.npz + jobs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_generate import PROMPTS, SNAP, libero10  # noqa: E402

REPLAY = ROOT / "data/outputs/hackathon_explore/e4_cosmos_fd/replay"
OUT = ROOT / "data/outputs/hackathon_explore/e5_probe"
STATS_DIR = ROOT / "configs/hackathon/cosmos_framework_stats"
RZ90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
ID6 = np.array([1, 0, 0, 0, 1, 0.0])
S1_FRAMES = ["can_s1_e0020_200000", "can_s1_e0020_200001", "square_s1_e0060_200000", "square_s1_e0120_200004"]
S2_FRAMES = ["can_s1_e0020_200000", "square_s1_e0060_200000"]
S_START = 32
ARMS = [("hold", None, 0), ("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1), ("-y", 1, -1), ("+z", 2, 1), ("-z", 2, -1)]


def qn(x, q01, q99):
    return 2.0 * (x - q01) / np.maximum(q99 - q01, 1e-8) - 1.0


def robomind_stats():
    s = json.loads((STATS_DIR / "robomind_franka_stats.json").read_text())
    return np.asarray(s["q01"], np.float64), np.asarray(s["q99"], np.float64)


def sweep8(axis, sign, enc, step=0.010):
    x = np.tile(np.r_[0, 0, 0, ID6, 1.0], (16, 1))
    if axis is not None:
        x[:, axis] = sign * step
    return x if enc == "D_raw" else qn(x, *robomind_stats())


def sweep5(axis, sign, g):
    c = np.zeros((16, 7))
    c[:, 6] = g
    if axis is not None:
        c[:, axis] = sign * 0.4  # OSC command units: 0.4 * 0.05 m = 2 cm per step
    return libero10(c)


ENC = {"D_raw": ("droid_lerobot", 15), "R_q": ("robomind-franka", 10)}


def jobs_s0():
    spec = json.loads((SNAP / "assets/example_action_fd_umi_action_chunks.json").read_text())
    a0, a1 = np.asarray(spec["action_chunks"][0]), np.asarray(spec["action_chunks"][1])
    neg = a0 * np.r_[-1, -1, -1, np.ones(7)]
    hold = np.tile(np.r_[0, 0, 0, ID6, a0[0, 9]], (16, 1))
    from PIL import Image
    img = np.asarray(Image.open(SNAP / "assets/example_action_fd_umi_first_frame.png").convert("RGB"))
    for arm, acts, seeds in (("main", a0, (0, 1, 2)), ("negpos", neg, (0, 1, 2)), ("hold", hold, (0, 1, 2)), ("other", a1, (0,))):
        for sd in seeds:
            yield dict(job=f"S0|umi|{arm}|{sd}", stage="S0", unit="umi", arm=arm, seed=sd, domain="umi", fps=int(spec["fps"]),
                       view="ego_view", prompt=spec["prompt"], image=img, actions=acts)


def jobs_s1():
    for ep in S1_FRAMES:
        d = np.load(REPLAY / f"{ep}.npz", allow_pickle=True)
        for enc, (dom, fps) in ENC.items():
            for arm, axis, sign in ARMS:
                yield dict(job=f"S1|{ep}|{enc}|{arm}", stage="S1", unit=ep, enc=enc, arm=arm, seed=1000 + S_START, domain=dom,
                           fps=fps, view="third_person_view", prompt=PROMPTS[str(d["task"])], image=d["frames"][S_START],
                           actions=sweep8(axis, sign, enc))


def jobs_s2():
    from e4_generate import delta_commands
    for ep in S2_FRAMES:
        d = dict(np.load(REPLAY / f"{ep}.npz", allow_pickle=True))
        g = float(delta_commands(d)[S_START, 6])
        for dom in ("libero", "libero_slot9"):
            for arm, axis, sign in ARMS:
                yield dict(job=f"S2|{ep}|{dom}|{arm}", stage="S2", unit=ep, enc=dom, arm=arm, seed=1000 + S_START, domain=dom,
                           fps=20, view="third_person_view", prompt=PROMPTS[str(d["task"])], image=d["frames"][S_START],
                           actions=sweep5(axis, sign, g))


def franka8(pos, mat, g01, M, enc):
    """inv(T_k) T_{k+1} of achieved poses in the frame R = R_site @ M; rot6d column-based; gripper 1 = open."""
    R = mat @ M
    out = []
    for k in range(16):
        dR = R[k].T @ R[k + 1]
        out.append(np.r_[R[k].T @ (pos[k + 1] - pos[k]), dR[:, 0], dR[:, 1], g01[k]])
    x = np.asarray(out)
    return x if enc == "D_raw" else qn(x, *robomind_stats())


def jobs_s3(enc, M):
    """Real replays on slot 8. Poses are subsampled to the encoding's fps from the 20 Hz replay."""
    from e4_generate import delta_commands
    dom, fps = ENC[enc]
    stride = 20 / fps
    for p in sorted(REPLAY.glob("*.npz")):
        ep = p.stem
        d = dict(np.load(p, allow_pickle=True))
        cmd = delta_commands(d)
        for s in (16, 48):
            idx = np.round(s + stride * np.arange(17)).astype(int)
            if idx[-1] >= len(d["frames"]):
                continue
            pos, mat = d["eef_pos"][idx], d["eef_mat"][idx]
            g01 = (1.0 - cmd[np.minimum(idx[:-1], len(cmd) - 1), 6]) / 2.0
            main = franka8(pos, mat, g01, M, enc)
            zero = franka8(np.repeat(pos[:1], 17, 0), np.repeat(mat[:1], 17, 0), g01, M, enc)
            rev = franka8(pos[::-1], mat[::-1], g01, M, enc)  # the same path traversed backwards
            arms = [("main", main, 5000 + s), ("zero", zero, 5000 + s), ("reversed", rev, 5000 + s)]
            if s == 16:
                arms += [("gripflip", franka8(pos, mat, 1.0 - g01, M, enc), 5000 + s), ("main_seed2", main, 9000 + s)]
            for arm, acts, sd in arms:
                yield dict(job=f"S3|{ep}|{s}|{enc}|{arm}", stage="S3", unit=f"{ep}|{s}", enc=enc, arm=arm, seed=sd, domain=dom,
                           fps=fps, view="third_person_view", prompt=PROMPTS[str(d["task"])], image=d["frames"][s],
                           actions=acts, real_idx=idx.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="S0")
    ap.add_argument("--s3-enc", default="R_q")
    ap.add_argument("--s3-M", default=None, help="JSON 3x3 action-frame rotation fitted in S1 (R = R_site @ M)")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    import diffusers.pipelines.cosmos.pipeline_cosmos3_omni as P
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from PIL import Image

    P._EMBODIMENT_TO_RAW_ACTION_DIM.setdefault("libero", 10)
    P._EMBODIMENT_TO_DOMAIN_ID["libero_slot9"] = 9  # untrained slot used only as a control
    P._EMBODIMENT_TO_RAW_ACTION_DIM["libero_slot9"] = 10
    OUT.mkdir(parents=True, exist_ok=True)
    meta = OUT / "jobs.jsonl"
    done = {json.loads(l)["job"] for l in meta.read_text().splitlines()} if meta.exists() else set()
    gens = []
    for st in a.stages.split(","):
        if st == "S0":
            gens.append(jobs_s0())
        elif st == "S1":
            gens.append(jobs_s1())
        elif st == "S2":
            gens.append(jobs_s2())
        elif st == "S3":
            M = np.asarray(json.loads(a.s3_M), np.float64) if a.s3_M else RZ90
            gens.append(jobs_s3(a.s3_enc, M))
    pipe = Cosmos3OmniPipeline.from_pretrained(str(SNAP), torch_dtype=torch.bfloat16, enable_safety_checker=False)
    pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False)
    for g in gens:
        for j in g:
            if j["job"] in done:
                continue
            t1 = time.time()
            r = pipe(prompt=j["prompt"], action=CosmosActionCondition(
                mode="forward_dynamics", chunk_size=16, domain_name=j["domain"], resolution_tier=256,
                raw_actions=torch.tensor(j["actions"], dtype=torch.float32), image=Image.fromarray(np.ascontiguousarray(j["image"])),
                view_point=j["view"]),
                fps=j["fps"], num_inference_steps=a.steps, guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(j["seed"]), use_system_prompt=False, output_type="np")
            v = np.asarray(r.video)
            vid = v if v.dtype == np.uint8 else (np.clip(v, 0, 1) * 255).round().astype(np.uint8)
            d = OUT / j["stage"]
            d.mkdir(exist_ok=True)
            name = j["job"].replace("|", "__")
            np.savez_compressed(d / f"{name}.npz", video=vid, actions=np.asarray(j["actions"], np.float32))
            rec = {k: j[k] for k in j if k not in ("image", "actions", "prompt")}
            rec.update(seconds=round(time.time() - t1, 2), frames=int(vid.shape[0]), hw=list(vid.shape[1:3]),
                       model="nvidia/Cosmos3-Edge@344d602b", file=str((d / f"{name}.npz").relative_to(ROOT)))
            with open(meta, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec[k] for k in ("job", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
