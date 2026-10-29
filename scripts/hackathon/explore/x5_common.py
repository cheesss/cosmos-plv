"""x5_cosmos_modes: shared helpers (EXPLORATORY / post-hoc).

Episode loading, frame extraction (scratch dir only), ROI crops, pairwise prompt
and parser, aggregation rules, and metrics.  Stdlib + numpy/PIL/imageio (Env B).

Nothing here writes outside data/outputs/hackathon_explore/x5_cosmos_modes/.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ROOT / "data/outputs/hackathon_explore/x5_cosmos_modes"
DEV_ROOT = ROOT / "data/outputs/hackathon_rollouts_dev"
EVAL_ROOT = ROOT / "data/outputs/hackathon_rollouts_eval"
COSMOS_DIR = ROOT / "data/outputs/hackathon_cosmos"

import sys

sys.path.insert(0, str(ROOT))
from cosmos_plv.visual_diagnostics import (  # noqa: E402
    OBJECT_NAME,
    PERCEPTION_HELP,
    PERCEPTION_LOCATIONS,
    PERCEPTION_SCENE,
    SUCCESS_LOCATION,
    ParsedDiagnostic,
    _first_json_object,
    _repair_closing_bracket,
    split_reasoning,
)

# ---------------------------------------------------------------- variants
# Fixed a priori from the 448x448 agentview geometry (see docs report): (x0, y0, x1, y1).
# can: both bins plus the carrying corridor above them; square: table area with both pegs.
ROI_BOX = {"can": (0, 40, 448, 336), "square": (16, 96, 432, 448)}
ROI_UPSAMPLE = 2  # bicubic; Cosmos3-Edge processor keeps native resolution (patch 16, merge 2)

VARIANTS = ("a_roi_track8", "b_track15", "c_pairwise8", "d_reason_last")
SUCCESS_CANDIDATES = ("a_roi_track8", "c_pairwise8", "d_reason_last")
FTYPE_CANDIDATES = ("a_roi_track8", "b_track15", "c_pairwise8")

MOTIONS = ("up", "down", "stayed", "sideways", "not_visible")


def keyframe_indices(n_frames: int, k: int) -> List[int]:
    import numpy as np

    return [int(round(x)) for x in np.linspace(0, n_frames - 1, k)]


def iter_episodes(root: Path):
    for summ in sorted(Path(root).glob("*/summary.json")):
        data = json.loads(summ.read_text())
        for ep in data["episodes"]:
            if ep.get("video") and (ROOT / ep["video"]).exists():
                yield ep


def interleave(eps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Seed-major round robin across checkpoints so a time-capped prefix is spread over checkpoints."""
    by_ck: Dict[str, List[Dict[str, Any]]] = {}
    for e in eps:
        by_ck.setdefault(e["checkpoint_id"], []).append(e)
    for v in by_ck.values():
        v.sort(key=lambda e: e["env_seed"])
    out, i = [], 0
    while any(i < len(v) for v in by_ck.values()):
        for ck in sorted(by_ck):
            if i < len(by_ck[ck]):
                out.append(by_ck[ck][i])
        i += 1
    return out


_FRAME_CACHE: Dict[str, List[Any]] = {}


def video_frames(video_rel: str):
    if video_rel not in _FRAME_CACHE:
        import imageio.v2 as imageio

        _FRAME_CACHE.clear()
        _FRAME_CACHE[video_rel] = [f for f in imageio.get_reader(str(ROOT / video_rel))]
    return _FRAME_CACHE[video_rel]


def frame_png(ep: Dict[str, Any], j: int, crop: bool = False) -> Path:
    from PIL import Image

    frames = video_frames(ep["video"])
    j = j % len(frames)
    d = SCRATCH / "frames" / ep["checkpoint_id"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"seed{ep['env_seed']}_f{j:03d}{'_roi' if crop else ''}.png"
    if not p.exists():
        im = Image.fromarray(frames[j])
        if crop:
            x0, y0, x1, y1 = ROI_BOX[ep["task"]]
            im = im.crop((x0, y0, x1, y1))
            im = im.resize(((x1 - x0) * ROI_UPSAMPLE, (y1 - y0) * ROI_UPSAMPLE), Image.BICUBIC)
        im.save(p)
    return p


# ---------------------------------------------------------------- pairwise (c)
def build_pair_prompt(task: str, dt_s: float) -> str:
    obj = OBJECT_NAME[task]
    scene = PERCEPTION_SCENE[task].replace("The image shows", "Both images show", 1)
    locs = " | ".join(f'"{x}"' for x in PERCEPTION_LOCATIONS[task])
    surface = "bin floor" if task == "can" else "table"
    return (
        f"Image 1 and image 2 are two frames from the same fixed camera, taken {dt_s:.1f} seconds apart "
        f"(image 1 first). {scene}\n"
        f"Describe only what is visible. Compare the {obj} in the two images.\n"
        f'"object_motion": did the {obj} itself move "up" (higher above the {surface}), "down" (lower), '
        f'"stayed" (same place), "sideways" (moved but at the same height), or is it "not_visible"?\n'
        f'"held_in_image_1" / "held_in_image_2": is the {obj} between the gripper fingers AND lifted off the '
        f"{surface} in that image?\n"
        f'"location_in_image_2": use exactly one of: {PERCEPTION_HELP[task]}.\n'
        "Return exactly one JSON object and no other text after it:\n"
        "{\n"
        '  "object_motion": "up" | "down" | "stayed" | "sideways" | "not_visible",\n'
        '  "held_in_image_1": true or false,\n'
        '  "held_in_image_2": true or false,\n'
        f'  "location_in_image_2": {locs}\n'
        "}"
    )


PAIR_SCHEMA = {"object_motion": str, "held_in_image_1": bool, "held_in_image_2": bool, "location_in_image_2": str}


def parse_pair(raw: str, task: str) -> ParsedDiagnostic:
    reasoning, answer = split_reasoning(raw or "")
    blob = _first_json_object(answer)
    repaired = False
    if blob is None:
        blob = _repair_closing_bracket(answer)
        repaired = blob is not None
    if blob is None:
        return ParsedDiagnostic("no_json", {}, ["no JSON object in answer"], reasoning, answer)
    try:
        obj = json.loads(re.sub(r"^```(?:json)?|```$", "", blob.strip()))
    except json.JSONDecodeError as e:
        return ParsedDiagnostic("invalid_json", {}, [f"json: {e}"], reasoning, answer)
    if not isinstance(obj, dict):
        return ParsedDiagnostic("invalid_json", {}, ["top level is not an object"], reasoning, answer)
    missing = [k for k in PAIR_SCHEMA if k not in obj]
    bad = [k for k, t in PAIR_SCHEMA.items() if k in obj and not isinstance(obj[k], t)]
    enum = []
    if isinstance(obj.get("object_motion"), str) and obj["object_motion"] not in MOTIONS:
        enum.append("object_motion")
    if isinstance(obj.get("location_in_image_2"), str) and obj["location_in_image_2"] not in PERCEPTION_LOCATIONS[task]:
        enum.append("location_in_image_2")
    status = ("missing_fields" if missing else "bad_types" if bad else "bad_enum" if enum
              else ("ok_repaired" if repaired else "ok"))
    errs = [f"missing {k}" for k in missing] + [f"type {k}" for k in bad] + [f"enum {k}" for k in enum]
    return ParsedDiagnostic(status, {k: obj.get(k) for k in PAIR_SCHEMA}, errs, reasoning, answer)


OK = ("ok", "ok_repaired")


def aggregate_pairs(task: str, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-registered rule for (c).  pairs: list of {parse_status, fields} in time order.

    success     location_in_image_2 of the LAST pair == success location (last pair must parse)
    lift event  any parsed pair with object_motion == "up" and (held_in_image_2 or location_in_image_2 == in_gripper)
    held_last   last pair held_in_image_2 or location_in_image_2 == in_gripper
    type        success -> none; no lift event -> no_grasp; held_last -> not_placed; else dropped
    """
    if not pairs or pairs[-1]["parse_status"] not in OK:
        return {"parse_status": "last_pair_unparsed"}
    valid = [p["fields"] for p in pairs if p["parse_status"] in OK]
    last = pairs[-1]["fields"]
    success = last["location_in_image_2"] == SUCCESS_LOCATION[task]
    lift = any(f["object_motion"] == "up" and (f["held_in_image_2"] or f["location_in_image_2"] == "in_gripper")
               for f in valid)
    held_last = bool(last["held_in_image_2"]) or last["location_in_image_2"] == "in_gripper"
    if success:
        ft = "none"
    elif not lift:
        ft = "no_grasp"
    elif held_last:
        ft = "not_placed"
    else:
        ft = "dropped"
    return {"parse_status": "ok", "task_success": success, "failure_type": ft, "object_lifted": lift,
            "contact_observed": lift, "object_dropped": ft == "dropped"}


def aggregate_frames(task: str, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same rule as cosmos_plv.visual_diagnostics.aggregate_track (v7), on dict rows."""
    if not frames or frames[-1]["parse_status"] not in OK:
        return {"parse_status": "last_frame_unparsed"}
    valid = [f for f in frames if f["parse_status"] in OK]
    held = [bool(f["held_by_gripper"]) or f["object_location"] == "in_gripper" for f in valid]
    success = valid[-1]["object_location"] == SUCCESS_LOCATION[task]
    ever, held_last = any(held), held[-1]
    ft = "none" if success else "no_grasp" if not ever else "not_placed" if held_last else "dropped"
    return {"parse_status": "ok", "task_success": success, "failure_type": ft, "object_lifted": ever,
            "contact_observed": ever, "object_dropped": ft == "dropped"}


# ---------------------------------------------------------------- metrics
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def metrics(rows: List[Dict[str, Any]], has_ftype: bool = False) -> Dict[str, Any]:
    """rows: {truth_success, truth_failure_type, truth_lifted, task, pred (dict or None), latency_s}.

    Conventions follow scripts/hackathon/summarize_diagnostics.py: success_acc on scored rows;
    *_all variants count an unparsed output as a wrong prediction.
    """
    n = len(rows)
    sc = [r for r in rows if r["pred"] and r["pred"].get("parse_status") == "ok"]
    tp = sum(1 for r in sc if r["pred"]["task_success"] and r["truth_success"])
    tn = sum(1 for r in sc if not r["pred"]["task_success"] and not r["truth_success"])
    fp = sum(1 for r in sc if r["pred"]["task_success"] and not r["truth_success"])
    fn = sum(1 for r in sc if not r["pred"]["task_success"] and r["truth_success"])
    n_s = sum(1 for r in rows if r["truth_success"])
    n_f = n - n_s
    out: Dict[str, Any] = {
        "episodes": n, "scored": len(sc), "unparsed": n - len(sc),
        "n_true_success": n_s, "n_true_failure": n_f,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "success_acc": round((tp + tn) / len(sc), 3) if sc else None,
        "success_acc_ci95": wilson(tp + tn, len(sc)),
        "success_acc_all": round((tp + tn) / n, 3) if n else None,
        "success_acc_all_ci95": wilson(tp + tn, n),
        "balanced_acc_all": round(0.5 * (tp / max(1, n_s) + tn / max(1, n_f)), 3) if n else None,
        "fp_rate_on_failures_all": round((n_f - tn) / n_f, 3) if n_f else None,
        "fp_rate_ci95": wilson(n_f - tn, n_f),
        "fn_rate_on_successes_all": round((n_s - tp) / n_s, 3) if n_s else None,
        "fn_rate_ci95": wilson(n_s - tp, n_s),
        "mean_latency_s_per_episode": round(sum(r["latency_s"] or 0 for r in rows) / n, 2) if n else None,
    }
    if has_ftype and rows:
        fails = [r for r in rows if not r["truth_success"]]
        k = sum(1 for r in fails if r["pred"] and r["pred"].get("parse_status") == "ok"
                and r["pred"]["failure_type"] == r["truth_failure_type"])
        # secondary, clearly labelled: square 'other' is annotated from a below-start-height rule and
        # is visually a never-grasped episode, so treat other==no_grasp for square only.
        def merged(r):
            t, p = r["truth_failure_type"], r["pred"]["failure_type"]
            if r["task"] == "square" and t == "other":
                t = "no_grasp"
            return p == t
        km = sum(1 for r in fails if r["pred"] and r["pred"].get("parse_status") == "ok" and merged(r))
        lift_rows = [r for r in rows if r["pred"] and r["pred"].get("parse_status") == "ok"]
        kl = sum(1 for r in lift_rows if bool(r["pred"]["object_lifted"]) == bool(r["truth_lifted"]))
        conf: Dict[str, int] = {}
        for r in fails:
            p = r["pred"]["failure_type"] if r["pred"] and r["pred"].get("parse_status") == "ok" else "unscored"
            key = f"{r['truth_failure_type']}->{p}"
            conf[key] = conf.get(key, 0) + 1
        out.update({
            "failure_type_acc_on_failures_all": round(k / len(fails), 3) if fails else None,
            "failure_type_acc_ci95": wilson(k, len(fails)),
            "failure_type_acc_square_other_as_no_grasp": round(km / len(fails), 3) if fails else None,
            "failure_type_merged_ci95": wilson(km, len(fails)),
            "lifted_acc_scored": round(kl / len(lift_rows), 3) if lift_rows else None,
            "lifted_acc_ci95": wilson(kl, len(lift_rows)),
            "failure_confusion": dict(sorted(conf.items())),
        })
    return out


def correct_vec(rows, field="success"):
    out = []
    for r in rows:
        p = r["pred"]
        ok = bool(p) and p.get("parse_status") == "ok"
        if field == "success":
            out.append(1 if ok and bool(p["task_success"]) == bool(r["truth_success"]) else 0)
        else:
            if r["truth_success"]:
                continue
            out.append(1 if ok and p.get("failure_type") == r["truth_failure_type"] else 0)
    return out


def paired_diff(a_rows, b_rows, field="success", n_boot=2000, seed=0) -> Dict[str, Any]:
    """Paired (same episodes) accuracy difference a - b with episode bootstrap CI and discordant counts."""
    ka = {(r["checkpoint_id"], r["env_seed"]): r for r in a_rows}
    kb = {(r["checkpoint_id"], r["env_seed"]): r for r in b_rows}
    keys = sorted(set(ka) & set(kb))
    if field != "success":
        keys = [k for k in keys if not ka[k]["truth_success"]]
    va = correct_vec([ka[k] for k in keys], "success" if field == "success" else "ft")
    vb = correct_vec([kb[k] for k in keys], "success" if field == "success" else "ft")
    n = len(keys)
    if n == 0:
        return {"n": 0}
    d = [x - y for x, y in zip(va, vb)]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {"n": n, "acc_a": round(sum(va) / n, 3), "acc_b": round(sum(vb) / n, 3),
            "diff": round(sum(d) / n, 3), "diff_ci95_boot": (round(boots[int(0.025 * n_boot)], 3),
                                                             round(boots[int(0.975 * n_boot) - 1], 3)),
            "a_right_b_wrong": sum(1 for x in d if x > 0), "a_wrong_b_right": sum(1 for x in d if x < 0)}


# ---------------------------------------------------------------- baselines from existing JSONL
def load_baseline(path: Path, variant: str, backend: str = "local") -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("variant") != variant or r.get("backend") != backend:
            continue
        pred = None
        if r.get("scored"):
            pred = {"parse_status": "ok", "task_success": r["cosmos_task_success"]}
            if r.get("cosmos_failure_type") is not None:
                pred.update({"failure_type": r["cosmos_failure_type"], "object_lifted": r["cosmos_object_lifted"],
                             "contact_observed": r["cosmos_contact_observed"],
                             "object_dropped": r["cosmos_object_dropped"]})
        else:
            pred = {"parse_status": r.get("parse_status") or "unscored"}
        out[(r["checkpoint_id"], int(r["env_seed"]))] = {
            "checkpoint_id": r["checkpoint_id"], "env_seed": int(r["env_seed"]), "task": r["task"],
            "truth_success": r["truth_success"], "truth_failure_type": r["truth_failure_type"],
            "truth_lifted": r["truth_lifted"], "pred": pred, "latency_s": r["latency_s"], "_row": r,
        }
    return out


# ---------------------------------------------------------------- post-hoc format repair for (c)
# Added AFTER seeing dev outputs (format only, values untouched): Cosmos3-Edge often answers the pair
# prompt with a bracketed list, e.g. '[ "object_motion": "stayed", ... ]' or a list of one-key objects.
# This extracts the four "key": value pairs by regex.  Reported separately as c_pairwise8_fmtrepair;
# it was NOT used by the in-run (pre-registered) selection.
_KV = re.compile(r'"(object_motion|held_in_image_1|held_in_image_2|location_in_image_2)"\s*:\s*("([^"]*)"|true|false)')


def parse_pair_repaired(raw: str, task: str) -> ParsedDiagnostic:
    p = parse_pair(raw, task)
    if p.parse_status in OK:
        return p
    _, answer = split_reasoning(raw or "")
    found: Dict[str, Any] = {}
    for m in _KV.finditer(answer):
        k, v = m.group(1), m.group(2)
        if k in found:
            continue
        found[k] = True if v == "true" else False if v == "false" else m.group(3)
    if set(found) != set(PAIR_SCHEMA):
        return p
    bad = [k for k, t in PAIR_SCHEMA.items() if not isinstance(found[k], t)]
    enum = (found["object_motion"] not in MOTIONS if isinstance(found["object_motion"], str) else False) or \
        (found["location_in_image_2"] not in PERCEPTION_LOCATIONS[task] if isinstance(found["location_in_image_2"], str) else False)
    if bad or enum:
        return p
    return ParsedDiagnostic("ok_repaired", found, ["format repair: regex key-value extraction"], None, answer)
