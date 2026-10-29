#!/usr/bin/env python3
"""EXPLORATORY (x3_plv_failure_link, part 4): Nemotron as a time-stamped DROP detector.

Motivation (from parts 1-3): replanning disagreement flags no_grasp-type failures but is blind to drops,
and the existing whole-video Nemotron v2 diagnostic files most drops under "not_placed" with unusable
failure times (median |error| 3.7 s).  This probe asks Nemotron ONE focused question -- did the object
leave the gripper before the goal, and at which burnt-in time stamp -- so that a PLV user without
simulator state could (a) count drops per checkpoint and (b) align replanning statistics to drop times.

Videos: each recorded agentview mp4 (10 fps) is resampled to exactly 4 fps (frame k = t 0.25k s) with the
time stamp drawn in the top-left corner, written to data/outputs/hackathon_explore/x3_plv_failure_link/
videos_ts/.  NebiusVLMClient re-encodes at 4 fps (step 1), so time stamps and real time agree.

Splits: --split dev (prompt check only) or --split eval (report; all episodes of checkpoints with
epoch > 0, i.e. selection does not use outcomes).  Calls are logged by NebiusVLMClient to
results/hackathon/explore/nebius_calls_explore.jsonl with tag explore/x3_plv_failure_link/droptimer/...
Hard cap: 800 calls with this tag prefix (counted from the evidence file before starting).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cosmos_plv.visual_diagnostics import END_STATE, OBJECT_NAME, TASK_DESCRIPTIONS  # noqa: E402

NAME = "x3_plv_failure_link"
EVIDENCE = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
TAG = f"explore/{NAME}/droptimer"
SCRATCH = ROOT / f"data/outputs/hackathon_explore/{NAME}"
CAP = 800
VARIANT = "droptimer_v1"


def build_prompt_v2(task: str, duration_s: float) -> str:
    """v2 (written after looking at v1 dev errors: false 'left early' on no_grasp pushes, missed drops):
    the model must first commit to the time stamps at which the object is held IN THE AIR."""
    obj = OBJECT_NAME[task]
    return (
        "The video shows one ATTEMPT by a learned robot policy in a simulated manipulation task, recorded by a "
        f"fixed third-person camera at 4 frames per second (about {duration_s:.1f} seconds). Every frame shows its "
        "time stamp in seconds since the start in the top-left corner (for example 't=3.25s').\n"
        "Many attempts fail. Judge only from what is visible in the frames.\n"
        f"Intended goal of the attempt: {TASK_DESCRIPTIONS[task]}\n{END_STATE[task]}\n\n"
        f"Step 1. Find the first time stamp at which the {obj} is clearly IN THE AIR between the closed gripper "
        f"fingers (raised above the surface it started on, not just touched, pushed, tipped or dragged). If this "
        "never happens, the object was never lifted and nothing can be dropped.\n"
        f"Step 2. Only if it was lifted: after that time, does the {obj} separate from the gripper (falls, slips "
        "out, or is let go) while it is NOT yet at the goal described above? Being released at the goal is NOT an "
        "early release, even if the release looks clumsy.\n"
        f"Step 3. If it separated early, give the time stamp of the first frame in which the {obj} is clearly no "
        "longer between the fingers.\n\n"
        "Return exactly one JSON object (all strings in double quotes) and no other text after it:\n"
        "{\n"
        '  "time_first_in_air_s": number or null,\n'
        '  "object_lifted_in_gripper": true or false,\n'
        '  "left_gripper_before_goal": true or false,\n'
        '  "time_left_gripper_s": number or null,\n'
        '  "task_success": true or false,\n'
        '  "explanation": "one short sentence"\n'
        "}"
    )


PROMPTS = {"droptimer_v1": None, "droptimer_v2": build_prompt_v2}


def build_prompt(task: str, duration_s: float) -> str:
    obj = OBJECT_NAME[task]
    return (
        "The video shows one ATTEMPT by a learned robot policy in a simulated manipulation task, recorded by a "
        f"fixed third-person camera at 4 frames per second (about {duration_s:.1f} seconds). Every frame shows its "
        "time stamp in seconds since the start in the top-left corner (for example 't=3.25s').\n"
        "Many attempts fail. Judge only from what is visible in the frames.\n"
        f"Intended goal of the attempt: {TASK_DESCRIPTIONS[task]}\n{END_STATE[task]}\n\n"
        f"Focus on the gripper fingers and the {obj}.\n"
        f"(1) Is the {obj} ever lifted off the surface it rests on while it is held between the gripper fingers?\n"
        f"(2) If it was lifted: does the {obj} later leave the gripper -- falls, slips out, or is let go -- "
        "anywhere that is NOT the goal, before the goal is reached? Letting go of it AT the goal is not leaving early.\n"
        f"(3) If it left the gripper early, read the time stamp of the first frame in which the {obj} is clearly no "
        "longer held.\n\n"
        "Return exactly one JSON object and no other text after it:\n"
        "{\n"
        '  "object_lifted_in_gripper": true or false,\n'
        '  "left_gripper_before_goal": true or false,\n'
        '  "time_left_gripper_s": number or null,\n'
        '  "task_success": true or false,\n'
        '  "explanation": "one short sentence"\n'
        "}"
    )


def parse(raw: str):
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    ok = (isinstance(obj.get("object_lifted_in_gripper"), bool)
                          and isinstance(obj.get("left_gripper_before_goal"), bool)
                          and isinstance(obj.get("task_success"), bool)
                          and (obj.get("time_left_gripper_s") is None or isinstance(obj.get("time_left_gripper_s"), (int, float))))
                    return obj, ("ok" if ok else "schema_error")
        start = text.find("{", start + 1)
    # fallback: malformed JSON (e.g. unquoted explanation string) -> read the typed fields individually
    obj = {}
    for key in ("object_lifted_in_gripper", "left_gripper_before_goal", "task_success", "time_left_gripper_s"):
        m = re.search(rf'"{key}"\s*:\s*(true|false|null|-?\d+(?:\.\d+)?)', text)
        if m:
            v = m.group(1)
            obj[key] = True if v == "true" else False if v == "false" else None if v == "null" else float(v)
    if all(isinstance(obj.get(k), bool) for k in ("object_lifted_in_gripper", "left_gripper_before_goal", "task_success")):
        obj.setdefault("time_left_gripper_s", None)
        return obj, "ok_field_regex"
    return None, "no_json"


def timestamped_video(src: Path, dst: Path, src_fps: float = 10.0) -> float:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    if dst.is_file():
        r = imageio.get_reader(str(dst))
        n = r.count_frames()
        r.close()
        return n / 4.0
    frames = [f for f in imageio.get_reader(str(src))]
    dur = len(frames) / src_fps
    font = ImageFont.load_default(size=30)
    out = []
    k = 0
    while k * 0.25 < dur - 1e-9:
        idx = min(len(frames) - 1, int(round(k * 0.25 * src_fps)))
        im = Image.fromarray(frames[idx]).convert("RGB")
        d = ImageDraw.Draw(im)
        txt = f"t={k * 0.25:.2f}s"
        d.rectangle([0, 0, 180, 40], fill=(0, 0, 0))
        d.text((6, 3), txt, fill=(255, 255, 0), font=font)
        out.append(np.asarray(im))
        k += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(dst), fps=4, codec="libx264", quality=7, macro_block_size=16,
                            pixelformat="yuv420p") as w:
        for f in out:
            w.append_data(f)
    return len(out) / 4.0


def used_calls() -> int:
    if not EVIDENCE.is_file():
        return 0
    n = 0
    for line in EVIDENCE.read_text().splitlines():
        try:
            if json.loads(line).get("tag", "").startswith(TAG):
                n += 1
        except json.JSONDecodeError:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", choices=sorted(PROMPTS), default="droptimer_v1")
    a = ap.parse_args()
    PROMPTS["droptimer_v1"] = build_prompt
    from cosmos_plv.nebius_backend import NebiusVLMClient

    rd = ROOT / ("data/outputs/hackathon_rollouts_dev" if a.split == "dev" else "data/outputs/hackathon_rollouts_eval")
    out_path = SCRATCH / (f"nemotron_droptimer_{a.split}.jsonl" if a.variant == "droptimer_v1"
                          else f"nemotron_{a.variant}_{a.split}.jsonl")
    done = set()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("parse_status") in ("ok", "ok_field_regex", "schema_error", "no_json"):
                done.add((r["checkpoint_id"], r["env_seed"]))
    jobs = []
    for summ in sorted(rd.glob("*/summary.json")):
        s = json.loads(summ.read_text())
        for ep in s["episodes"]:
            ep_epoch = int(ep["checkpoint_id"].rsplit("_e", 1)[1])
            if a.split == "eval" and ep_epoch == 0:
                continue  # epoch-0 checkpoints are trivially failing (no_grasp/other); not where drops matter
            if (ep["checkpoint_id"], int(ep["env_seed"])) in done:
                continue
            jobs.append(ep)
    if a.limit:
        jobs = jobs[: a.limit]
    used = used_calls()
    if used + len(jobs) > CAP:
        print(f"refusing: {used} calls used, {len(jobs)} planned, cap {CAP}")
        sys.exit(2)
    print(f"{len(jobs)} episodes to judge ({used} calls already used under {TAG})", flush=True)
    client = NebiusVLMClient(evidence_path=str(EVIDENCE))
    lock = threading.Lock()

    def run(ep):
        src = ROOT / ep["video"]
        dst = SCRATCH / "videos_ts" / f"{ep['checkpoint_id']}_seed{ep['env_seed']}.mp4"
        dur = timestamped_video(src, dst)
        prompt = PROMPTS[a.variant](ep["task"], dur)
        res, obj, status = None, None, "backend_error"
        for attempt in range(2):  # at most one retry, only for backend errors
            res = client.generate(prompt, video=str(dst), max_tokens=400, reasoning=False, temperature=0.2,
                                  tag=f"{TAG}/{a.variant}/{a.split}/{ep['checkpoint_id']}/{ep['env_seed']}")
            if res.error:
                continue
            obj, status = parse(res.raw_text)
            break
        row = {"checkpoint_id": ep["checkpoint_id"], "env_seed": int(ep["env_seed"]), "task": ep["task"],
               "variant": a.variant, "model_id": client.model_id, "split": a.split,
               "truth_success": bool(ep["success"]), "truth_failure_type": ep["failure_type"],
               "truth_lifted": bool(ep.get("lifted")), "truth_dropped": bool(ep.get("dropped")),
               "video_ts": str(dst.relative_to(ROOT)), "video_duration_s": dur,
               "parse_status": status, "pred": obj, "raw_text": res.raw_text if res else "",
               "backend_error": res.error if res else None, "latency_s": res.latency_s if res else None,
               "request_id": res.request_id if res else None,
               "prompt_tokens": res.prompt_tokens if res else None, "completion_tokens": res.completion_tokens if res else None}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return status

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        stats = list(ex.map(run, jobs))
    print({s: stats.count(s) for s in set(stats)}, "calls used now:", used_calls())


if __name__ == "__main__":
    main()
