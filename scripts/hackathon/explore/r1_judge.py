#!/usr/bin/env python3
"""r1 counterfactual replanning, step 3 — judge inputs and judge calls (Environment B).

  inputs   build per-event clips from the RECORDED agentview video (no simulator needed):
           J1  frames from step T-40 to T (10 fps source, written at 4 fps = 2.5x slow motion)
           J2  (and J2s with the labels swapped) the same clip + 4 held copies of the frame at T with both 7-step end-effector
               paths drawn (old plan action_pred[c][9:16], new plan action_pred[c+1][1:8]),
               labelled 1 / 2 in the per-event random order fixed by r1_select.py
           CF  frames at T-16 and T as PNG for the local Cosmos perception rule
  nemotron Nemotron-3.5-Super-VL on Nebius Token Factory, prompts J1 and J2 (frozen below);
           amendment 1 (after the dev pilot): J2s (label-swapped J2) and J1b (v6 perception on the CF frames)
  cosmos   Cosmos3-Edge local, frozen v6 perception prompt on the two CF frames
Protocol: docs/hackathon/explore/r1_counterfactual_replan_protocol.md
Output: data/outputs/hackathon_explore/r1_judge/<split>/{inputs/,J1.jsonl,J2.jsonl,CF.jsonl}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/outputs/hackathon_explore/r1_judge"
OBJ = {"can": "red soda can", "square": "square nut"}
GOAL = {"can": "pick up the red soda can from the left bin and place it into the right bin",
        "square": "pick up the square nut and slide it onto the square peg"}
COLORS = {"1": (255, 140, 0), "2": (0, 220, 255)}
FONT = None

PROMPT_J1 = (
    "These frames show a robot arm manipulating a {obj} in a simulator, played back 2.5x slower than real time "
    "(about 2 seconds of real time). Describe only what is visible. Did anything unexpected happen to the {obj} "
    "during this clip, for example it slipped in or fell out of the gripper, was dropped, was pushed or knocked "
    "over, or the gripper closed but missed it?\n"
    "Return exactly one JSON object and no other text:\n"
    '{{\n  "unexpected_event": true or false,\n'
    '  "event": "none" | "slipped" | "dropped" | "pushed_or_knocked" | "missed_grasp" | "other",\n'
    '  "replan_necessity": 1 | 2 | 3 | 4 | 5,\n'
    '  "explanation": "one sentence about what you see"\n}}\n'
    "replan_necessity: 1 = the robot can simply continue its current motion, 5 = the robot must change its plan now."
)
PROMPT_J2 = (
    "The video shows a robot arm manipulating a {obj} in a simulator (2.5x slow motion). The final frame is held "
    "and shows two candidate continuations of the gripper for the next 0.35 seconds: an orange path labelled 1 and "
    "a cyan path labelled 2. Filled squares mark points where the gripper is commanded to close. The robot's task "
    "is to {goal}. Judging only from the situation visible in the final frame, which continuation better "
    "accomplishes the task?\n"
    "Return exactly one JSON object and no other text:\n"
    '{{\n  "better": "1" | "2" | "no_meaningful_difference",\n'
    '  "explanation": "one sentence"\n}}'
)


def events(split):
    return json.loads((ROOT / f"results/hackathon/explore/r1_events_{split}.json").read_text())["events"]


def key(ev):
    return f"{ev['checkpoint_id']}_{ev['env_seed']}_{ev['c']}"


def proj(T, P):
    h = np.r_[P, 1.0] @ np.asarray(T).T
    return h[1] / h[2], h[0] / h[2]  # (row, col)


def build_inputs(split):
    import imageio.v2 as imageio
    from PIL import ImageFont

    from cosmos_plv.video_recorder import read_mp4, write_mp4

    global FONT
    FONT = ImageFont.load_default(size=22)
    cam = json.loads((ROOT / "results/hackathon/explore/r1_cam_T.json").read_text())
    d = OUT / split / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    cache = {}
    for ev in events(split):
        k = key(ev)
        if (d / f"{k}_J2s.mp4").exists():
            continue
        if ev["video"] not in cache:
            cache = {ev["video"]: read_mp4(ROOT / ev["video"])}
        frames = cache[ev["video"]]
        T = ev["T"]
        f0, f1 = max(0, (T - 40) // 2), min(T // 2, len(frames) - 1)
        clip = frames[f0:f1 + 1]
        write_mp4(d / f"{k}_J1.mp4", clip, fps=4)
        rec = np.load(ROOT / ev["episode_npz"])
        ap = rec["action_pred"]
        paths = {"old": ap[ev["c"]][9:16], "new": ap[ev["c"] + 1][1:8]}
        for variant, new_is_1 in (("J2", ev["new_is_1"]), ("J2s", not ev["new_is_1"])):
            draw_overlay(d / f"{k}_{variant}.mp4", clip, frames[f1], cam[ev["task"]], rec["eef_pos"][T], paths, new_is_1)
        imageio.imwrite(d / f"{k}_CF_a.png", frames[max(0, (T - 16) // 2)])
        imageio.imwrite(d / f"{k}_CF_b.png", frames[f1])
    print("inputs ready:", len(list(d.glob("*_J2.mp4"))))


def draw_overlay(path, clip, last, cam_T, eef_T, paths, new_is_1):
    from PIL import Image, ImageDraw

    from cosmos_plv.video_recorder import write_mp4

    lab = {"new": "1", "old": "2"} if new_is_1 else {"new": "2", "old": "1"}
    img = Image.fromarray(last).convert("RGB")
    dr = ImageDraw.Draw(img)
    start = proj(cam_T, eef_T)
    for name in ("old", "new"):
        col = COLORS[lab[name]]
        pts = [start] + [proj(cam_T, p[:3]) for p in paths[name]]
        dr.line([(c, r) for r, c in pts], fill=col, width=3)
        for (r, c), a in zip(pts[1:], paths[name]):
            if a[9] > 0:  # gripper commanded closed
                dr.rectangle([c - 4, r - 4, c + 4, r + 4], fill=col)
            else:
                dr.ellipse([c - 3, r - 3, c + 3, r + 3], outline=col, width=2)
        r, c = pts[-1]
        dr.text((c + 6, r - 14), lab[name], fill=col, font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    over = np.asarray(img)
    write_mp4(path, list(clip) + [over] * 4, fps=4)


def run_nemotron(split, variant, workers=4):
    from cosmos_plv.nebius_backend import NebiusVLMClient

    out = OUT / split / f"{variant}.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    client = NebiusVLMClient(evidence_path=str(ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"))
    todo = [ev for ev in events(split) if key(ev) not in done]

    def one(ev):
        k = key(ev)
        if variant == "J1b":  # amendment 1: per-frame perception (frozen v6 prompt) at T-16 and T
            from cosmos_plv.visual_diagnostics import build_perception_prompt
            row = {"key": k, "variant": variant}
            for side in ("a", "b"):
                res = client.generate(build_perception_prompt(ev["task"]), images=[str(OUT / split / "inputs" / f"{k}_CF_{side}.png")],
                                      max_tokens=300, tag=f"r1_{variant}_{split}")
                row[f"raw_{side}"], row[f"error_{side}"], row[f"request_id_{side}"] = res.raw_text, res.error, res.request_id
            return row
        prompt = (PROMPT_J1 if variant == "J1" else PROMPT_J2).format(obj=OBJ[ev["task"]], goal=GOAL[ev["task"]])
        res = client.generate(prompt, video=str(OUT / split / "inputs" / f"{k}_{variant}.mp4"), max_tokens=400,
                              tag=f"r1_{variant}_{split}")
        return {"key": k, "variant": variant, "raw_text": res.raw_text, "error": res.error, "request_id": res.request_id,
                "latency_s": res.latency_s, "served_model": res.served_model}

    with ThreadPoolExecutor(workers) as ex:
        for row in ex.map(one, todo):
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
    print(variant, split, "rows:", len(out.read_text().splitlines()))


def run_cosmos(split):
    from PIL import Image

    from cosmos_plv.cosmos_backend import CosmosReasonerLocal
    from cosmos_plv.visual_diagnostics import build_perception_prompt

    out = OUT / split / "CF.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    model = CosmosReasonerLocal()
    for ev in events(split):
        k = key(ev)
        if k in done:
            continue
        row = {"key": k}
        for side in ("a", "b"):
            res = model.generate(build_perception_prompt(ev["task"]), images=[str(OUT / split / "inputs" / f"{k}_CF_{side}.png")],
                                 max_new_tokens=300, reasoning=False, seed=0)  # same settings as the v6 eval run
            row[f"raw_{side}"], row[f"error_{side}"] = res.raw_text, res.error
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
    print("CF", split, "rows:", len(out.read_text().splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["inputs", "nemotron", "cosmos"])
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--variants", default="J1,J2,J2s,J1b")
    a = ap.parse_args()
    if a.step == "inputs":
        build_inputs(a.split)
    elif a.step == "nemotron":
        for v in a.variants.split(","):
            run_nemotron(a.split, v)
    else:
        run_cosmos(a.split)


if __name__ == "__main__":
    main()
