#!/usr/bin/env python3
"""x2_early_stop (EXPLORATORY, post-hoc): Nemotron-3.5-Super-VL on video PREFIXES.

Question asked of the model: "Is this attempt going to succeed?" given only the
first ``cut_s`` seconds of a recorded rollout.  Prefixes are cut at ABSOLUTE
times (not at a fraction of the episode length): every episode that is still
running at the cut gets a clip of identical length, so clip length cannot leak
the outcome (successful episodes end 16 steps after success and are shorter).
Episodes that already succeeded before the cut need no decision and are not sent.

Protocol (fixed before looking at eval):
  1. dev split (data/outputs/hackathon_rollouts_dev): all prompt variants at
     cut 4 s; the variant with the higher dev balanced accuracy of
     ``will_succeed`` (tie: higher AUROC of the ordinal score) is selected.
  2. eval split: the selected variant, unchanged, at 4 s (all running episodes)
     and 8 s (running episodes of checkpoints with epoch > 0, within budget).

Nebius calls go through NebiusVLMClient with an evidence log; a hard budget is
enforced by counting this exploration's tags in that log.  Resumable: rows
already in --out are skipped.

Env B:
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/x2_early_stop_nebius.py --split dev --variants A_plain,B_evidence --cuts 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cosmos_plv.visual_diagnostics import OBJECT_NAME, TASK_DESCRIPTIONS, _first_json_object, split_reasoning  # noqa: E402

NAME = "x2_early_stop"
EVIDENCE = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
SCRATCH = ROOT / "data/outputs/hackathon_explore" / NAME
SPLIT_ROOT = {"dev": ROOT / "data/outputs/hackathon_rollouts_dev", "eval": ROOT / "data/outputs/hackathon_rollouts_eval"}
SIM_HZ = 20.0
VIDEO_STEP = 2  # video frame k = sim step 2k
HORIZON_S = 20.0
CONF = ("low", "medium", "high")
ORDINAL = {(True, "high"): 1.0, (True, "medium"): 0.8, (True, "low"): 0.6,
           (False, "low"): 0.4, (False, "medium"): 0.2, (False, "high"): 0.0}

JSON_A = '{"will_succeed": true or false, "confidence": "low" | "medium" | "high", "reason": "one sentence"}'
JSON_B = ('{"gripper_closed_on_object": true or false, "object_lifted": true or false, '
          '"object_fell_or_pushed_away": true or false, "will_succeed": true or false, '
          '"confidence": "low" | "medium" | "high", "reason": "one sentence"}')


def build_prompt(variant: str, task: str, cut_s: float) -> str:
    head = (
        f"The video shows only the FIRST {cut_s:.0f} seconds of one attempt by a learned robot policy in a "
        "simulated manipulation task, recorded by a fixed third-person camera. The attempt is NOT finished: "
        f"the robot is allowed up to {HORIZON_S:.0f} seconds in total and keeps acting after the clip ends.\n"
        f"Goal of the task: {TASK_DESCRIPTIONS[task]}\n"
    )
    if variant == "A_plain":
        return head + (
            f"Question: based only on what you see so far, is this attempt going to succeed within the "
            f"{HORIZON_S:.0f} seconds?\n"
            "Return exactly one JSON object and no other text:\n" + JSON_A
        )
    if variant == "B_evidence":
        obj = OBJECT_NAME[task]
        return head + (
            "Many attempts fail. Judge only from what is visible; do not assume that the robot will do what it "
            "is supposed to do.\n"
            f"Before answering, look at the LAST frames of the clip and check: (1) are the gripper fingers closed "
            f"around the {obj}? (2) is the {obj} lifted clearly above where it started? (3) has the {obj} fallen, "
            "slipped out of the gripper, or been pushed away?\n"
            f"Then answer: is this attempt going to succeed within the {HORIZON_S:.0f} seconds?\n"
            "Return exactly one JSON object and no other text:\n" + JSON_B
        )
    raise KeyError(variant)


def parse(raw: str):
    _, answer = split_reasoning(raw or "")
    blob = _first_json_object(answer)
    if blob is None:
        return "no_json", {}
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return "invalid_json", {}
    if not isinstance(obj, dict):
        return "invalid_json", {}
    ws, conf = obj.get("will_succeed"), obj.get("confidence")
    if not isinstance(ws, bool):
        return "bad_types", obj
    if conf not in CONF:
        return "bad_enum", obj
    return "ok", obj


def episodes(split: str):
    for summ in sorted(SPLIT_ROOT[split].glob("*/summary.json")):
        for ep in json.loads(summ.read_text())["episodes"]:
            if ep.get("video"):
                yield ep


def running_at(ep, cut_step: int) -> bool:
    end = int(ep["n_steps"]) - 1
    fss = int(ep["first_success_step"])
    succeeded_by_cut = ep["success"] and 0 <= fss <= cut_step
    return end > cut_step and not succeeded_by_cut


def epoch_of(ckpt_id: str) -> int:
    return int(ckpt_id.rsplit("_e", 1)[1])


def make_clip(ep, cut_step: int) -> Path:
    import imageio.v2 as imageio

    out = SCRATCH / "clips" / ep["checkpoint_id"] / f"seed{ep['env_seed']}_t{cut_step:03d}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    last = cut_step // VIDEO_STEP
    reader = imageio.get_reader(str(ROOT / ep["video"]))
    frames = []
    for i, f in enumerate(reader):
        if i > last:
            break
        frames.append(f)
    reader.close()
    tmp = out.with_suffix(".tmp.mp4")
    with imageio.get_writer(str(tmp), fps=float(ep["video_fps"]), codec="libx264", quality=8,
                            macro_block_size=16, pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
    tmp.rename(out)
    return out


def calls_used() -> int:
    if not EVIDENCE.exists():
        return 0
    n = 0
    for line in EVIDENCE.read_text().splitlines():
        try:
            if json.loads(line).get("tag", "").startswith(f"explore/{NAME}/"):
                n += 1
        except json.JSONDecodeError:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "eval"), required=True)
    ap.add_argument("--variants", default="A_plain,B_evidence")
    ap.add_argument("--cuts", default="4", help="comma list of absolute cut times in seconds")
    ap.add_argument("--skip-epoch0-cuts", default="", help="cut times (s) at which epoch-0 checkpoints are skipped")
    ap.add_argument("--budget", type=int, default=300, help="max total Nebius calls for this exploration")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(SCRATCH / "nebius_prefix_calls.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("parse_status") is not None and not r.get("backend_error"):
                done.add((r["split"], r["variant"], r["checkpoint_id"], r["env_seed"], r["cut_step"]))
    skip0 = {float(x) for x in a.skip_epoch0_cuts.split(",") if x}
    jobs = []
    for cut_s in [float(x) for x in a.cuts.split(",")]:
        cut_step = int(round(cut_s * SIM_HZ))
        for variant in a.variants.split(","):
            for ep in episodes(a.split):
                if not running_at(ep, cut_step):
                    continue
                if cut_s in skip0 and epoch_of(ep["checkpoint_id"]) == 0:
                    continue
                key = (a.split, variant, ep["checkpoint_id"], ep["env_seed"], cut_step)
                if key not in done:
                    jobs.append((variant, cut_s, cut_step, ep))
    used = calls_used()
    room = max(0, a.budget - used)
    print(json.dumps({"pending_jobs": len(jobs), "calls_used": used, "budget": a.budget, "room": room}), flush=True)
    if len(jobs) > room:
        rng = random.Random(0)
        rng.shuffle(jobs)
        jobs = sorted(jobs[:room], key=lambda j: (j[1], j[0], j[3]["checkpoint_id"], j[3]["env_seed"]))
        print(json.dumps({"truncated_to": len(jobs)}), flush=True)
    if a.dry_run or not jobs:
        return

    from cosmos_plv.nebius_backend import NebiusVLMClient

    client = NebiusVLMClient(evidence_path=str(EVIDENCE))
    lock = threading.Lock()

    def run(job):
        variant, cut_s, cut_step, ep = job
        clip = make_clip(ep, cut_step)
        prompt = build_prompt(variant, ep["task"], cut_s)
        tag = f"explore/{NAME}/{a.split}/{variant}/t{cut_step:03d}/{ep['checkpoint_id']}/seed{ep['env_seed']}"
        res = client.generate(prompt, video=str(clip), max_tokens=a.max_tokens, reasoning=False, tag=tag)
        status, obj = parse(res.raw_text) if not res.error else ("backend_error", {})
        ws = obj.get("will_succeed") if status == "ok" else None
        conf = obj.get("confidence") if status == "ok" else None
        row = {
            "exploration": NAME, "split": a.split, "variant": variant, "cut_s": cut_s, "cut_step": cut_step,
            "checkpoint_id": ep["checkpoint_id"], "task": ep["task"], "env_seed": ep["env_seed"],
            "clip": str(clip.relative_to(ROOT)), "clip_frames": cut_step // VIDEO_STEP + 1,
            "truth_success": ep["success"], "truth_failure_type": ep["failure_type"],
            "first_success_step": ep["first_success_step"], "n_steps": ep["n_steps"],
            "parse_status": status, "will_succeed": ws, "confidence": conf,
            "score_p_success": ORDINAL.get((ws, conf)) if status == "ok" else None,
            "parsed": obj, "raw_text": res.raw_text, "latency_s": res.latency_s,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "model_id": res.model_id, "served_model": res.served_model, "request_id": res.request_id,
            "backend_error": res.error, "prompt": prompt, "tag": tag,
        }
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps({k: row[k] for k in ("variant", "cut_s", "checkpoint_id", "env_seed", "truth_success",
                                               "will_succeed", "confidence", "parse_status", "latency_s")}),
              flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(run, jobs))
    print(json.dumps({"calls_used_after": calls_used()}), flush=True)


if __name__ == "__main__":
    main()
