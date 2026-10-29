#!/usr/bin/env python3
"""EXPLORATORY / POST-HOC: x4_nemotron_modes.

Alternative ways to apply nvidia/Nemotron-3.5-Super-VL (Nebius Token Factory) to
physical failure diagnosis of recorded robosuite rollouts.  The frozen v2 video
prompt (eval success acc 0.959) gets the failure TYPE right on only ~40% of the
failed episodes and misses "wrong compartment" (can) / "nut resting on top of
the peg" (square) cases.

Variants (all one Nebius call per episode):
  A_track8   8 evenly spaced keyframes as 8 separate images in ONE request,
             task-free per-frame perception (fine-grained location + held),
             failure type derived by a fixed rule that mirrors failure_annotator.
  B_v2reason the frozen v2 video prompt text, unchanged, with reasoning enabled
             (enable_thinking) and max_tokens 4096.
  C_vidcomp  video + full-resolution last frame, task-free compartment/peg-aware
             fact schema (lifted? held at end? final fine-grained location),
             same fixed rule.
  D_grid8    the same 8 keyframes tiled into ONE 2x4 grid image with timestamps,
             per-panel perception, same fixed rule.
  V2_frozen  the frozen v2 settings (reasoning off, max_tokens 700); only used to
             extend the frozen baseline to eval episodes recorded after it ran.

Variant choice happens on the DEV split only (``select``); the chosen variants
are then run unchanged on the EVAL split.

  python scripts/hackathon/explore/x4_nemotron_modes.py run --split dev --variant A_track8
  python scripts/hackathon/explore/x4_nemotron_modes.py select
  python scripts/hackathon/explore/x4_nemotron_modes.py run --split eval --variant <chosen>
  python scripts/hackathon/explore/x4_nemotron_modes.py score
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

from cosmos_plv.failure_annotator import annotate_episode  # noqa: E402
from cosmos_plv.provenance import repo_relative, utc_now  # noqa: E402
from cosmos_plv.visual_diagnostics import (  # noqa: E402
    ParsedDiagnostic,
    _first_json_object,
    _repair_closing_bracket,
    agreement_row,
    build_prompt,
    parse_diagnostic,
    split_reasoning,
)

NAME = "x4_nemotron_modes"
SCRATCH = ROOT / "data/outputs/hackathon_explore" / NAME
EVIDENCE = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
RESULTS = ROOT / "results/hackathon/explore" / f"{NAME}.json"
SELECTION = SCRATCH / "dev_selection.json"
CALL_BUDGET = 800
SPLITS = {
    "dev": ROOT / "data/outputs/hackathon_rollouts_dev",
    "eval": ROOT / "data/outputs/hackathon_rollouts_eval",
}
FROZEN_V2 = {
    "dev": ROOT / "data/outputs/hackathon_cosmos/dev_prompt_selection.jsonl",
    "eval": ROOT / "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl",
}
VARIANTS = ("A_track8", "B_v2reason", "C_vidcomp", "D_grid8", "V2_frozen")
CANDIDATES = ("A_track8", "B_v2reason", "C_vidcomp", "D_grid8")
SCORED_CANDIDATES = ("A_track8", "A_track8_r2", "B_v2reason", "C_vidcomp", "D_grid8", "D_grid8_r2")
N_KEYFRAMES = 8
SETTLE_STEPS = 10  # the square nut spawns ~6 cm above the table and settles within ~5 steps

# ----------------------------------------------------------------------------- prompts

LOCS = {
    "can": ("left_bin", "right_bin_front_right", "right_bin_front_left", "right_bin_back_left",
            "right_bin_back_right", "in_gripper", "on_rim_or_table", "fell_to_floor", "not_visible"),
    "square": ("on_square_peg_low", "on_square_peg_high", "on_round_peg", "on_table", "in_gripper",
               "off_table", "not_visible"),
}
GOAL_LOC = {"can": "right_bin_front_right", "square": "on_square_peg_low"}
FELL_LOCS = {"fell_to_floor", "off_table"}
OBJ = {"can": "red can", "square": "square nut"}
SCENE = {
    "can": (
        "The scene shows a robot arm above two bins seen from a fixed camera: a light wooden bin on the LEFT "
        "and a dark wooden bin on the RIGHT. The dark bin is divided into four compartments: two FRONT "
        "compartments (nearest the camera, at the bottom of the image) and two BACK compartments (farther "
        "away). The FRONT-RIGHT compartment contains a small grey can-shaped marker; the FRONT-LEFT "
        "compartment contains a small grey cube marker; the back compartments contain tall grey blocks. "
        "There is exactly one red soda can (ignore the grey markers)."
    ),
    "square": (
        "The scene shows a robot arm above a white table seen from a fixed camera. On the table stand two "
        "vertical pegs: a round grey metal peg on the left and a square brown wooden peg on the right. There "
        "is one flat square brown nut with a handle (a square ring with a square hole)."
    ),
}
LOC_HELP = {
    "can": (
        '"left_bin" = inside the light wooden bin on the left; '
        '"right_bin_front_right" = inside the front-right compartment of the dark bin (the one with the grey '
        'can-shaped marker); '
        '"right_bin_front_left" = inside the front-left compartment of the dark bin (the one with the small '
        'grey cube); '
        '"right_bin_back_left" / "right_bin_back_right" = inside one of the two back compartments of the dark '
        'bin; '
        '"in_gripper" = carried in the air between the gripper fingers, not resting on anything; '
        '"on_rim_or_table" = resting on a bin wall, a rim, or a surface outside both bins; '
        '"fell_to_floor" = fell off to the floor below the bins; '
        '"not_visible" = cannot be seen'
    ),
    "square": (
        '"on_square_peg_low" = the square wooden peg passes through the nut\'s hole and the nut sits low on '
        'the peg, at or close to the table; '
        '"on_square_peg_high" = the nut is at the square peg but high up: resting on top of the peg, tilted on '
        'it, or not yet slid down; '
        '"on_round_peg" = the nut is around the round grey peg; '
        '"on_table" = the nut lies on the table, not around any peg; '
        '"in_gripper" = carried in the air between the gripper fingers, away from the pegs; '
        '"off_table" = fell off the table; '
        '"not_visible" = cannot be seen'
    ),
}


def _loc_enum(task: str) -> str:
    return " | ".join(f'"{x}"' for x in LOCS[task])


def build_track_prompt(task: str, times_s: List[float], grid: bool) -> str:
    k = len(times_s)
    if grid:
        intro = (
            f"The image is a grid of {k} panels (2 rows x {k // 2} columns, read left to right, top row first). "
            f"Each panel is labelled with its number and time; they are keyframes in time order from one "
            f"recorded robot episode, all from the same fixed camera.\n"
        )
        unit = "panel"
    else:
        stamps = ", ".join(f"image {i + 1} at t={t:.1f}s" for i, t in enumerate(times_s))
        intro = (
            f"The {k} images are keyframes in time order from one recorded robot episode, all from the same "
            f"fixed camera ({stamps}).\n"
        )
        unit = "image"
    obj = OBJ[task]
    entries = ",\n".join(
        f'    {{"{unit}": {i + 1}, "location": <one of the values>, "held": true or false}}' for i in range(k)
    )
    return (
        intro
        + f"{SCENE[task]}\n"
        f"Describe only what is visible. For EVERY {unit}, say where the {obj} is and whether it is held "
        f"between the gripper fingers. Judge each {unit} on its own; do not assume what the robot intended.\n"
        f"Location values: {LOC_HELP[task]}.\n"
        "The location describes where the object is, even if the gripper still touches it; "
        '"held" is true only if the object is clamped between the closed gripper fingers.\n'
        "Return exactly one valid JSON object (every string in double quotes) and no other text:\n"
        "{\n"
        f'  "{unit}s": [\n{entries}\n  ],\n'
        f'  "note": "one short sentence on what happens to the {obj}"\n'
        "}\n"
        f"Allowed location values: {_loc_enum(task)}."
    )


def build_vidcomp_prompt(task: str, duration_s: float) -> str:
    obj = OBJ[task]
    return (
        "The video shows one recorded robot episode from a fixed camera, sampled at about 5 frames per second "
        f"(about {duration_s:.1f} s). After the video, one extra still image shows the FINAL frame at full "
        "resolution.\n"
        f"{SCENE[task]}\n"
        f"Describe only what is visible; do not assume what the robot intended. Answer these questions about "
        f"the {obj}:\n"
        f"(1) did the gripper fingers ever close around the {obj}?\n"
        f"(2) was the {obj} ever lifted and carried in the air by the gripper?\n"
        f"(3) after being lifted, did the {obj} ever leave the gripper (released, slipped or fell)?\n"
        f"(4) where is the {obj} in the FINAL frame? Use exactly one of: {LOC_HELP[task]}.\n"
        f"(5) is the {obj} still clamped between the gripper fingers in the final frame?\n"
        "Return exactly one valid JSON object (every string in double quotes) and no other text:\n"
        "{\n"
        '  "gripper_closed_on_object": true or false,\n'
        '  "object_lifted_in_air": true or false,\n'
        '  "object_left_gripper_after_lift": true or false,\n'
        f'  "final_location": one of {_loc_enum(task)},\n'
        '  "held_at_end": true or false,\n'
        f'  "note": "one short sentence on what happens to the {obj}"\n'
        "}"
    )


# ----------------------------------------------------------------------------- parsing + fixed rule


_UNQUOTED = re.compile(r'^(\s*"[A-Za-z_]+"\s*:\s*)(?!["\d\-\[\{]|true\b|false\b|null\b)(\S.*?)"?(,?)\s*$')


def repair_missing_quotes(raw: str) -> str:
    """Single recorded repair (chosen on the dev smoke test): a string value written without its opening
    quote, e.g. '"explanation": The robot ...",'.  Only quote characters are added; words are untouched."""
    out = []
    for line in (raw or "").splitlines():
        m = _UNQUOTED.match(line)
        if m:
            val = m.group(2).rstrip()
            if val.endswith('"'):
                val = val[:-1]
            line = m.group(1) + '"' + val.replace('"', '\\"') + '"' + m.group(3)
        out.append(line)
    return "\n".join(out)


def parse_diagnostic_repaired(raw: str) -> Tuple[ParsedDiagnostic, bool]:
    p = parse_diagnostic(raw)
    if p.parse_status in ("ok", "ok_repaired"):
        return p, False
    fixed = repair_missing_quotes(raw)
    if fixed != raw:
        q = parse_diagnostic(fixed)
        if q.parse_status in ("ok", "ok_repaired"):
            q.parse_status = "ok_repaired"
            q.errors.append("repaired: missing opening quote")
            return q, True
    return p, False


def _load_json(raw: str, _retry: bool = True) -> Tuple[Optional[Any], str, List[str], Optional[str]]:
    if _retry:
        obj, st, err, rs = _load_json(raw, _retry=False)
        if obj is None:
            fixed = repair_missing_quotes(raw)
            if fixed != raw:
                obj2, st2, err2, rs2 = _load_json(fixed, _retry=False)
                if obj2 is not None:
                    return obj2, "ok_repaired", err2 + ["repaired: missing opening quote"], rs2
        return obj, st, err, rs
    reasoning, answer = split_reasoning(raw or "")
    blob = _first_json_object(answer)
    errors: List[str] = []
    repaired = False
    if blob is None:
        blob = _repair_closing_bracket(answer)
        repaired = blob is not None
    if blob is None:
        return None, "no_json", ["no JSON object in answer"], reasoning
    try:
        obj = json.loads(re.sub(r"^```(?:json)?|```$", "", blob.strip()))
    except json.JSONDecodeError as e:
        return None, "invalid_json", [f"json: {e}"], reasoning
    if not isinstance(obj, dict):
        return None, "invalid_json", ["top level is not an object"], reasoning
    if repaired:
        errors.append("repaired: closing ']' -> '}'")
    return obj, ("ok_repaired" if repaired else "ok"), errors, reasoning


def fixed_rule(task: str, final_loc: str, ever_held: bool, held_last: bool) -> Tuple[bool, str]:
    """Mirror of failure_annotator.py, applied to visual facts (fixed before any data was seen)."""
    success = final_loc == GOAL_LOC[task]
    if success:
        return True, "none"
    if not ever_held:
        return False, ("other" if final_loc in FELL_LOCS else "no_grasp")
    if held_last:
        return False, "not_placed"
    return False, "dropped"


def _main_fields(task, success, ftype, contact, lifted, note, extra=None):
    f = {
        "contact_observed": bool(contact),
        "object_lifted": bool(lifted),
        "object_dropped": ftype == "dropped",
        "task_success": bool(success),
        "failure_type": ftype,
        "failure_time_s": None,
        "explanation": note,
        "uncertainty_note": "",
    }
    if extra:
        f.update(extra)
    return f


def parse_track(raw: str, task: str, k: int, unit: str) -> Tuple[ParsedDiagnostic, List[Dict[str, Any]]]:
    obj, status, errors, reasoning = _load_json(raw)
    if obj is None:
        return ParsedDiagnostic(status, {}, errors, reasoning, raw or ""), []
    frames = obj.get(unit + "s")
    if not isinstance(frames, list) or len(frames) != k:
        n = len(frames) if isinstance(frames, list) else None
        return ParsedDiagnostic("missing_fields", {}, errors + [f"{unit}s: expected {k}, got {n}"], reasoning,
                                raw), []
    track = []
    for i, fr in enumerate(frames):
        ok = (isinstance(fr, dict) and fr.get("location") in LOCS[task] and isinstance(fr.get("held"), bool))
        track.append({"i": i, "location": fr.get("location") if isinstance(fr, dict) else None,
                      "held": fr.get("held") if isinstance(fr, dict) else None, "valid": bool(ok)})
        if not ok:
            errors.append(f"{unit} {i + 1}: invalid entry")
    if not track[-1]["valid"]:
        return ParsedDiagnostic("last_frame_unparsed", {}, errors, reasoning, raw), track
    valid = [t for t in track if t["valid"]]
    held = [t["held"] or t["location"] == "in_gripper" for t in valid]
    final = valid[-1]["location"]
    success, ftype = fixed_rule(task, final, any(held), held[-1])
    seq = ", ".join(f"{t['i'] + 1}:{t['location']}{'(held)' if h else ''}" for t, h in zip(valid, held))
    note = obj.get("note") if isinstance(obj.get("note"), str) else ""
    fields = _main_fields(task, success, ftype, any(held), any(held), f"{seq}. {note}",
                          {"final_location": final})
    return ParsedDiagnostic(status, fields, errors, reasoning, raw), track


VIDCOMP_SCHEMA = {
    "gripper_closed_on_object": bool,
    "object_lifted_in_air": bool,
    "object_left_gripper_after_lift": bool,
    "final_location": str,
    "held_at_end": bool,
}


def parse_vidcomp(raw: str, task: str) -> ParsedDiagnostic:
    obj, status, errors, reasoning = _load_json(raw)
    if obj is None:
        return ParsedDiagnostic(status, {}, errors, reasoning, raw or "")
    missing = [k for k in VIDCOMP_SCHEMA if k not in obj]
    bad = [k for k, t in VIDCOMP_SCHEMA.items() if k in obj and not isinstance(obj[k], t)]
    enum = isinstance(obj.get("final_location"), str) and obj["final_location"] not in LOCS[task]
    if missing or bad or enum:
        st = "missing_fields" if missing else "bad_types" if bad else "bad_enum"
        return ParsedDiagnostic(st, {}, errors + [f"missing {missing} bad {bad} enum {enum}"], reasoning, raw)
    final = obj["final_location"]
    held_last = bool(obj["held_at_end"]) or final == "in_gripper"
    ever_held = bool(obj["object_lifted_in_air"]) or held_last
    success, ftype = fixed_rule(task, final, ever_held, held_last)
    note = obj.get("note") if isinstance(obj.get("note"), str) else ""
    fields = _main_fields(task, success, ftype, obj["gripper_closed_on_object"] or ever_held, ever_held, note,
                          {"final_location": final, "facts": {k: obj[k] for k in VIDCOMP_SCHEMA}})
    return ParsedDiagnostic(status, fields, errors, reasoning, raw)


# ----------------------------------------------------------------------------- inputs


def iter_episodes(split: str):
    for summ in sorted(SPLITS[split].glob("*/summary.json")):
        data = json.loads(summ.read_text())
        for ep in data["episodes"]:
            if ep.get("video") and (ROOT / ep["video"]).exists():
                yield ep


def keyframes(ep: Dict[str, Any], k: int = N_KEYFRAMES) -> Tuple[List[Path], List[float], Path]:
    """Save k evenly spaced keyframes (jpg, 448 px) and the full-res last frame (png) to scratch."""
    import imageio.v2 as imageio
    from PIL import Image

    out = SCRATCH / "frames" / ep["checkpoint_id"] / f"seed{ep['env_seed']}"
    frames_list = None
    n_f = int(ep["video_frames"])
    idx = [int(round(x)) for x in np.linspace(0, n_f - 1, k)]
    paths = [out / f"kf{j:03d}.jpg" for j in idx]
    last = out / "last.png"
    if not (all(p.exists() for p in paths) and last.exists()):
        frames_list = [f for f in imageio.get_reader(str(ROOT / ep["video"]))]
        n_f = len(frames_list)
        idx = [int(round(x)) for x in np.linspace(0, n_f - 1, k)]
        paths = [out / f"kf{j:03d}.jpg" for j in idx]
        out.mkdir(parents=True, exist_ok=True)
        for j, p in zip(idx, paths):
            Image.fromarray(frames_list[j]).save(p, quality=92)
        Image.fromarray(frames_list[-1]).save(last)
    times = [j / float(ep["video_fps"]) for j in idx]
    return paths, times, last


def grid_image(ep, paths: List[Path], times: List[float], panel: int = 384) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    out = paths[0].parent / f"grid2x4_{panel}.jpg"
    if out.exists():
        return out
    cols, rows, band = len(paths) // 2, 2, 28
    canvas = Image.new("RGB", (cols * panel, rows * (panel + band)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for i, (p, t) in enumerate(zip(paths, times)):
        r, c = divmod(i, cols)
        x, y = c * panel, r * (panel + band)
        canvas.paste(Image.open(p).convert("RGB").resize((panel, panel), Image.BILINEAR), (x, y + band))
        draw.text((x + 6, y + 3), f"{i + 1}  t={t:.1f}s", fill="black", font=font)
        draw.rectangle([x, y, x + panel - 1, y + panel + band - 1], outline="black", width=2)
    canvas.save(out, quality=92)
    return out


def truth_extras(ep: Dict[str, Any]) -> Dict[str, Any]:
    """Sim-state facts used ONLY for reporting subsets (never as model input)."""
    z = np.load(ROOT / ep["episode_npz"])
    obj = z["object_pos"]
    end = obj[-1]
    ex: Dict[str, Any] = {"object_end_xyz": [round(float(v), 4) for v in end]}
    if ep["task"] == "can":
        # robosuite PickPlaceCan: bin2 centre (0.1, 0.28), size 0.39 x 0.49; can compartment x>=0.1, y>=0.28
        in_right_bin = (-0.095 <= end[0] <= 0.295) and (0.035 <= end[1] <= 0.525)
        ex["hard_case"] = "wrong_compartment" if (in_right_bin and not ep["success"] and not (
            end[0] >= 0.1 and end[1] >= 0.28)) else None
    else:
        near_sq_peg = float(np.hypot(end[0] - 0.23, end[1] - 0.10)) < 0.04
        ex["hard_case"] = "nut_high_on_peg" if (near_sq_peg and end[2] > 0.86 and not ep["success"]) else None
    tr = {k: z[k][SETTLE_STEPS:] for k in ("eef_pos", "object_pos", "grasped", "success")}
    ex["harmonized_failure_type"] = annotate_episode(tr)["failure_type"]
    return ex


# ----------------------------------------------------------------------------- run

_lock = threading.Lock()


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


def out_path(split: str, variant: str) -> Path:
    return SCRATCH / f"{split}_{variant}.jsonl"


def run_one(client, ep: Dict[str, Any], variant: str, split: str) -> Dict[str, Any]:
    task = ep["task"]
    duration = ep["video_frames"] / float(ep["video_fps"])
    tag = f"explore/{NAME}/{split}/{variant}/{ep['checkpoint_id']}/seed{ep['env_seed']}"
    video, images, track = None, None, None
    if variant in ("A_track8", "D_grid8"):
        paths, times, _ = keyframes(ep)
        prompt = build_track_prompt(task, times, grid=(variant == "D_grid8"))
        images = [str(p) for p in paths] if variant == "A_track8" else [str(grid_image(ep, paths, times))]
        max_tokens, reasoning = 900, False
    elif variant == "C_vidcomp":
        _, _, last = keyframes(ep)
        prompt = build_vidcomp_prompt(task, duration)
        video, images = str(ROOT / ep["video"]), [str(last)]
        max_tokens, reasoning = 500, False
    elif variant == "B_v2reason":
        prompt = build_prompt(task, 4.0, duration, False, "v2_evidence_first")  # frozen v2 text, unchanged
        video = str(ROOT / ep["video"])
        max_tokens, reasoning = 4096, True
    elif variant == "V2_frozen":
        prompt = build_prompt(task, 4.0, duration, False, "v2_evidence_first")
        video = str(ROOT / ep["video"])
        max_tokens, reasoning = 700, False
    else:
        raise KeyError(variant)
    res = client.generate(prompt, video=video, images=images, max_tokens=max_tokens, reasoning=reasoning, tag=tag)
    if variant in ("A_track8", "D_grid8"):
        parsed, track = parse_track(res.raw_text, task, N_KEYFRAMES, "image" if variant == "A_track8" else "panel")
    elif variant == "C_vidcomp":
        parsed = parse_vidcomp(res.raw_text, task)
    else:
        parsed, _ = parse_diagnostic_repaired(res.raw_text)
    truth = {k: ep[k] for k in ("success", "failure_type", "grasped", "lifted", "dropped")}
    row = {
        "exploration": NAME, "split": split, "variant": variant, "backend": "nebius", "reasoning": reasoning,
        "checkpoint_id": ep["checkpoint_id"], "task": task, "env_seed": ep["env_seed"], "video": ep["video"],
        "video_frames": ep["video_frames"], "truth_success": ep["success"],
        "truth_failure_type": ep["failure_type"], "truth_grasped": ep["grasped"], "truth_lifted": ep["lifted"],
        "truth_dropped": ep["dropped"], **truth_extras(ep),
        "parse_status": parsed.parse_status, "parse_errors": "; ".join(parsed.errors),
        **{f"cosmos_{k}": parsed.fields.get(k) for k in ("task_success", "failure_type", "contact_observed",
                                                          "object_lifted", "object_dropped", "explanation")},
        "final_location": parsed.fields.get("final_location"), "facts": parsed.fields.get("facts"),
        "track": track, **agreement_row(parsed, truth),
        "input_images": [repo_relative(p) for p in images] if images else None,
        "max_tokens": max_tokens, "raw_text": res.raw_text, "model_id": res.model_id,
        "served_model": res.served_model, "request_id": res.request_id, "latency_s": res.latency_s,
        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
        "backend_error": res.error, "prompt_sha256": res.params.get("prompt_sha256"), "logged_utc": utc_now(),
    }
    return row


def cmd_run(a):
    from cosmos_plv.nebius_backend import NebiusVLMClient

    out = out_path(a.split, a.variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if not r.get("backend_error"):
                done.add((r["checkpoint_id"], r["env_seed"]))
    eps = [e for e in iter_episodes(a.split) if (e["checkpoint_id"], e["env_seed"]) not in done]
    if a.checkpoint_filter:
        eps = [e for e in eps if a.checkpoint_filter in e["checkpoint_id"]]
    if a.only_missing_from_frozen:
        have = {(r["checkpoint_id"], r["env_seed"]) for r in map(json.loads, open(FROZEN_V2[a.split]))
                if r.get("backend") == "nebius" and r.get("variant") == "v2_evidence_first"}
        eps = [e for e in eps if (e["checkpoint_id"], e["env_seed"]) not in have]
    if a.limit:
        eps = eps[: a.limit]
    used = calls_used()
    if used + len(eps) > CALL_BUDGET:
        print(f"budget: used {used}, need {len(eps)}, cap {CALL_BUDGET}; truncating", flush=True)
        eps = eps[: max(0, CALL_BUDGET - used)]
    print(f"{a.split}/{a.variant}: {len(eps)} episodes to run (calls used so far {used})", flush=True)
    client = NebiusVLMClient(evidence_path=str(EVIDENCE))

    def work(ep):
        try:
            row = run_one(client, ep, a.variant, a.split)
        except Exception as e:  # infrastructure failure: record, do not score as a model answer
            row = {"exploration": NAME, "split": a.split, "variant": a.variant, "checkpoint_id": ep["checkpoint_id"],
                   "env_seed": ep["env_seed"], "backend_error": f"{type(e).__name__}: {str(e)[:300]}"}
        with _lock:
            with open(out, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps({k: row.get(k) for k in ("checkpoint_id", "env_seed", "truth_success", "truth_failure_type",
                                                  "cosmos_task_success", "cosmos_failure_type", "final_location",
                                                  "parse_status", "latency_s", "backend_error")}), flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, eps))
    print(f"done; calls used now {calls_used()}", flush=True)


# ----------------------------------------------------------------------------- scoring


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 3), round(c + h, 3)]


def load_rows(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("backend_error") and "truth_success" not in r:
            continue  # infrastructure failure row, superseded by a retry if any
        rows[(r["checkpoint_id"], int(r["env_seed"]))] = r
    return rows


def base_variant(name: str) -> str:
    return name[:-3] if name.endswith("_r2") else name


def rederive_r2(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rule r2 (chosen on DEV after inspecting dev errors, applied unchanged to eval): "lifted" requires the
    object to be located "in_gripper" (carried in the air) in at least one keyframe; a "held" flag while the
    object still rests in place counts as contact only (annotator: grasp without a 5 cm lift = no_grasp)."""
    r = dict(row)
    r["variant"] = row["variant"] + "_r2"
    track = row.get("track") or []
    valid = [t for t in track if t.get("valid")]
    if not row.get("scored") or not valid:
        return r
    final = valid[-1]["location"]
    lifted = any(t["location"] == "in_gripper" for t in valid)
    held_last = bool(valid[-1]["held"]) or final == "in_gripper"
    success = final == GOAL_LOC[r["task"]]
    if success:
        ft = "none"
    elif not lifted:
        ft = "other" if final in FELL_LOCS else "no_grasp"
    elif held_last:
        ft = "not_placed"
    else:
        ft = "dropped"
    r["cosmos_failure_type"] = ft
    r["cosmos_object_lifted"] = lifted
    r["cosmos_object_dropped"] = ft == "dropped"
    r["failure_type_correct"] = ft == r["truth_failure_type"]
    r["lifted_correct"] = lifted == bool(r["truth_lifted"])
    r["dropped_correct"] = (ft == "dropped") == bool(r["truth_dropped"])
    return r


def load_variant(split: str, name: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows = load_rows(out_path(split, base_variant(name)))
    if name.endswith("_r2"):
        rows = {k: rederive_r2(r) for k, r in rows.items()}
    return rows


def load_frozen(split: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows = {}
    for line in open(FROZEN_V2[split]):
        r = json.loads(line)
        if r.get("backend") == "nebius" and r.get("variant") == "v2_evidence_first" and not r.get("reasoning"):
            rows[(r["checkpoint_id"], int(r["env_seed"]))] = r
    ext = load_rows(out_path(split, "V2_frozen"))  # same frozen settings, episodes recorded later
    for k, r in ext.items():
        rows.setdefault(k, r)
    return rows


def correct_flags(r: Dict[str, Any], harm: Optional[str] = None) -> Dict[str, Optional[bool]]:
    scored = bool(r.get("scored"))
    succ_ok = scored and bool(r.get("success_correct"))
    ft_ok = scored and r.get("cosmos_failure_type") == r["truth_failure_type"]
    out = {"scored": scored, "success_ok": succ_ok, "ft_ok": ft_ok}
    if harm is not None:
        out["ft_harm_ok"] = scored and r.get("cosmos_failure_type") == harm
    return out


def metrics(rows: List[Dict[str, Any]], harm: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"episodes": 0}
    sc = [r for r in rows if r.get("scored")]
    tp = sum(1 for r in sc if r["cosmos_task_success"] and r["truth_success"])
    tn = sum(1 for r in sc if not r["cosmos_task_success"] and not r["truth_success"])
    fp = sum(1 for r in sc if r["cosmos_task_success"] and not r["truth_success"])
    fn = sum(1 for r in sc if not r["cosmos_task_success"] and r["truth_success"])
    uns = [r for r in rows if not r.get("scored")]
    u_f = sum(1 for r in uns if not r["truth_success"])
    u_s = sum(1 for r in uns if r["truth_success"])
    fails_all = [r for r in rows if not r["truth_success"]]
    fails_sc = [r for r in fails_all if r.get("scored")]
    ft_ok_sc = sum(1 for r in fails_sc if r["cosmos_failure_type"] == r["truth_failure_type"])
    hk = lambda r: harm[(r["checkpoint_id"], int(r["env_seed"]))]  # noqa: E731
    h_fails = [r for r in rows if not r["truth_success"]]
    h_ok = sum(1 for r in h_fails if r.get("scored") and r["cosmos_failure_type"] == hk(r)["harmonized_failure_type"])
    conf = collections.defaultdict(collections.Counter)
    for r in rows:
        conf[r["truth_failure_type"]][r.get("cosmos_failure_type") if r.get("scored") else "UNPARSED"] += 1
    hard = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        hc = hk(r).get("hard_case")
        if hc:
            hard[hc][0] += 1
            hard[hc][1] += int(bool(r.get("scored")) and not r["cosmos_task_success"])
    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    pt = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    ct = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    return {
        "episodes": n, "scored": len(sc), "unparsed": n - len(sc),
        "success_acc": round((tp + tn) / len(sc), 3) if sc else None, "success_acc_ci95": wilson(tp + tn, len(sc)),
        "success_acc_all": round((tp + tn) / n, 3), "success_acc_all_ci95": wilson(tp + tn, n),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "balanced_acc": round(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), 3) if sc else None,
        "balanced_acc_all": round(0.5 * (tp / max(1, tp + fn + u_s) + tn / max(1, tn + fp + u_f)), 3),
        "n_failures": len(fails_all), "n_failures_scored": len(fails_sc),
        "failure_type_acc_on_failures": round(ft_ok_sc / len(fails_sc), 3) if fails_sc else None,
        "failure_type_acc_on_failures_ci95": wilson(ft_ok_sc, len(fails_sc)),
        "failure_type_acc_on_failures_all": round(ft_ok_sc / len(fails_all), 3) if fails_all else None,
        "failure_type_acc_on_failures_all_ci95": wilson(ft_ok_sc, len(fails_all)),
        "harmonized_failure_type_acc_on_failures_all": round(h_ok / len(h_fails), 3) if h_fails else None,
        "harmonized_failure_type_acc_on_failures_all_ci95": wilson(h_ok, len(h_fails)),
        "hard_cases_flagged_as_failure": {k: {"n": v[0], "flagged": v[1], "ci95": wilson(v[1], v[0])}
                                          for k, v in sorted(hard.items())},
        "confusion_truth_to_pred": {k: dict(v) for k, v in sorted(conf.items())},
        "mean_latency_s": round(statistics.mean(lat), 2) if lat else None,
        "median_latency_s": round(statistics.median(lat), 2) if lat else None,
        "mean_prompt_tokens": round(statistics.mean(pt), 1) if pt else None,
        "mean_completion_tokens": round(statistics.mean(ct), 1) if ct else None,
        "calls_per_episode": 1,
    }


def by_task(rows, harm):
    out = {"all": metrics(rows, harm)}
    for t in ("can", "square"):
        out[t] = metrics([r for r in rows if r["task"] == t], harm)
    return out


def paired(var: Dict, base: Dict, harm: Dict, n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Paired comparison on the intersection of episodes; unparsed counts as wrong for both."""
    from scipy.stats import binomtest

    keys = sorted(set(var) & set(base))
    out = {"n_common": len(keys)}
    rng = np.random.default_rng(seed)
    for metric, subset in (("success_ok", None), ("ft_ok", "fail"), ("ft_harm_ok", "fail")):
        ks = [k for k in keys if subset is None or not base[k]["truth_success"]]
        if not ks:
            continue
        a = np.array([correct_flags(var[k], harm[k]["harmonized_failure_type"])[metric] for k in ks], float)
        b = np.array([correct_flags(base[k], harm[k]["harmonized_failure_type"])[metric] for k in ks], float)
        d = a - b
        idx = rng.integers(0, len(ks), size=(n_boot, len(ks)))
        boots = d[idx].mean(axis=1)
        n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
        out[metric + ("_on_failures" if subset else "")] = {
            "n": len(ks), "variant": round(a.mean(), 3), "frozen_v2": round(b.mean(), 3),
            "diff": round(d.mean(), 3), "diff_ci95_bootstrap": [round(float(np.quantile(boots, 0.025)), 3),
                                                              round(float(np.quantile(boots, 0.975)), 3)],
            "variant_only_correct": n10, "v2_only_correct": n01, "mcnemar_exact_p": round(p, 4)}
    return out


def per_checkpoint(rows: Dict, base: Dict) -> Dict[str, Any]:
    """PLV angle: does the VLM-estimated per-checkpoint success rate track the simulator's?"""
    from cosmos_plv.ranking import spearman

    keys = sorted(set(rows) & set(base))
    ck = collections.defaultdict(lambda: {"truth": [], "var": [], "v2": []})
    for k in keys:
        r, b = rows[k], base[k]
        ck[k[0]]["truth"].append(float(r["truth_success"]))
        ck[k[0]]["var"].append(float(bool(r.get("scored")) and bool(r["cosmos_task_success"])))
        ck[k[0]]["v2"].append(float(bool(b.get("scored")) and bool(b["cosmos_task_success"])))
    ids = sorted(ck)
    t = [np.mean(ck[i]["truth"]) for i in ids]
    v = [np.mean(ck[i]["var"]) for i in ids]
    w = [np.mean(ck[i]["v2"]) for i in ids]
    return {"n_checkpoints": len(ids),
            "spearman_variant_vs_truth": round(spearman(v, t), 3), "spearman_v2_vs_truth": round(spearman(w, t), 3),
            "mae_variant": round(float(np.mean(np.abs(np.array(v) - t))), 3),
            "mae_v2": round(float(np.mean(np.abs(np.array(w) - t))), 3),
            "rows": [{"checkpoint_id": i, "truth": round(a, 2), "variant": round(b, 2), "v2": round(c, 2)}
                     for i, a, b, c in zip(ids, t, v, w)]}


def harm_table(split: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    cache = SCRATCH / f"{split}_truth_extras.json"
    have = {}
    if cache.exists():
        have = {(d["checkpoint_id"], d["env_seed"]): d for d in json.loads(cache.read_text())}
    for ep in iter_episodes(split):
        k = (ep["checkpoint_id"], int(ep["env_seed"]))
        if k not in have:
            have[k] = {"checkpoint_id": k[0], "env_seed": k[1], "task": ep["task"],
                       "truth_failure_type": ep["failure_type"], **truth_extras(ep)}
    cache.write_text(json.dumps(list(have.values())))
    return have


def dev_report() -> Dict[str, Any]:
    harm = harm_table("dev")
    base = load_frozen("dev")
    out = {"frozen_v2": by_task(list(base.values()), harm)}
    for v in SCORED_CANDIDATES:
        rows = load_variant("dev", v)
        if rows:
            out[v] = by_task(list(rows.values()), harm)
            out[v]["paired_vs_frozen_v2"] = paired(rows, base, harm)
    return out


def cmd_select(a):
    """Pre-declared dev rule: maximise failure-type accuracy on failures (unparsed = wrong) subject to
    balanced success accuracy (unparsed = wrong) >= frozen v2 dev value - 0.05; keep the best 2."""
    rep = dev_report()
    floor = rep["frozen_v2"]["all"]["balanced_acc_all"] - 0.05
    cands = []
    for v in SCORED_CANDIDATES:
        if v not in rep:
            continue
        m = rep[v]["all"]
        ok = m["episodes"] == 36 and m["balanced_acc_all"] >= floor
        cands.append((ok, m["failure_type_acc_on_failures_all"] or 0, m["balanced_acc_all"], v))
    cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    chosen, bases = [], set()
    for c in cands:  # best 2 with distinct underlying model calls (r2 is post-processing of the same calls)
        if c[0] and base_variant(c[3]) not in bases and len(chosen) < 2:
            chosen.append(c[3])
            bases.add(base_variant(c[3]))
    sel = {"rule": cmd_select.__doc__, "floor_balanced_acc_all": round(floor, 3),
           "ranking": [{"variant": c[3], "eligible": c[0], "ft_acc_on_failures_all": c[1], "balanced_acc_all": c[2]}
                       for c in cands], "chosen": chosen, "selected_utc": utc_now()}
    SELECTION.write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


def cmd_score(a):
    dev = dev_report()
    harm = harm_table("eval")
    base = load_frozen("eval")
    sel = json.loads(SELECTION.read_text()) if SELECTION.exists() else {"chosen": []}
    eval_eps = sorted(harm)
    ev: Dict[str, Any] = {"episodes_present": len(eval_eps),
                          "checkpoints_present": sorted({k[0] for k in eval_eps})}
    sens = [b + "_r2" if not b.endswith("_r2") else base_variant(b) for b in sel["chosen"]
            if base_variant(b) in ("A_track8", "D_grid8")]
    for v in list(sel["chosen"]) + sens:
        rows = load_variant("eval", v)
        if not rows:
            continue
        common = sorted(set(rows) & set(base))
        ev[v] = {"role": "chosen_on_dev" if v in sel["chosen"] else "sensitivity_other_rule", "run_on": len(rows), "common_with_frozen_v2": len(common),
                 "variant": by_task([rows[k] for k in common], harm),
                 "frozen_v2_same_episodes": by_task([base[k] for k in common], harm),
                 "paired_vs_frozen_v2": paired(rows, base, harm),
                 "per_checkpoint_success_rate": per_checkpoint(rows, base),
                 "variant_all_run_episodes": by_task(list(rows.values()), harm)}
    label_audit = collections.Counter((d["task"], d["truth_failure_type"], d["harmonized_failure_type"])
                                      for d in harm.values())
    hard = collections.Counter((d["task"], d.get("hard_case")) for d in harm.values() if d.get("hard_case"))
    res = {
        "name": NAME, "status": "EXPLORATORY / POST-HOC", "created_utc": utc_now(),
        "model": "nvidia/Nemotron-3.5-Super-VL via Nebius Token Factory",
        "nebius_calls_used": calls_used(), "nebius_call_budget": CALL_BUDGET,
        "selection_on_dev": sel, "dev": dev, "eval": ev,
        "label_audit_eval": [{"task": t, "official": o, "harmonized_settled_z0": h, "n": n}
                             for (t, o, h), n in sorted(label_audit.items())],
        "hard_cases_eval": [{"task": t, "case": c, "n": n} for (t, c), n in sorted(hard.items())],
        "notes": [
            "Variant choice used the dev split only (36 episodes, seeds 300000+).",
            "Frozen v2 baseline = eval_nebius_v2.jsonl rows; episodes recorded after it ran were scored with the "
            "same frozen settings (V2_frozen) in this exploration's scratch file.",
            "Harmonized failure type = failure_annotator applied after the first 10 sim steps (square nut spawns "
            "about 6 cm above the table and settles), a sim-state relabel that never uses model outputs.",
        ],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=("dev", "eval"), required=True)
    r.add_argument("--variant", choices=VARIANTS, required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--checkpoint-filter", default="")
    r.add_argument("--only-missing-from-frozen", action="store_true")
    sub.add_parser("select")
    sub.add_parser("score")
    sub.add_parser("calls")
    a = ap.parse_args()
    if a.cmd == "run":
        if a.split == "eval" and a.variant in CANDIDATES:
            sel = json.loads(SELECTION.read_text()) if SELECTION.exists() else {"chosen": []}
            if a.variant not in {base_variant(c) for c in sel["chosen"]}:
                sys.exit(f"{a.variant} was not chosen on dev ({sel['chosen']}); refusing to run it on eval")
        cmd_run(a)
    elif a.cmd == "select":
        cmd_select(a)
    elif a.cmd == "score":
        cmd_score(a)
    else:
        print(calls_used())


if __name__ == "__main__":
    main()
