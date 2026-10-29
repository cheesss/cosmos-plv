#!/usr/bin/env python3
"""x5_cosmos_modes analysis (EXPLORATORY / post-hoc; CPU only, Env B).

Reads the cached per-call outputs (data/outputs/hackathon_explore/x5_cosmos_modes/calls.jsonl), the
existing v6/v7 diagnostics and the simulator labels in the rollout summaries; writes
results/hackathon/explore/x5_cosmos_modes.json.  Also provides the pre-registered dev selection rule
used by x5_cosmos_modes.py.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_analyze.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x5_common as C  # noqa: E402

OUT = C.ROOT / "results/hackathon/explore/x5_cosmos_modes.json"
OUT_EP = C.ROOT / "results/hackathon/explore/x5_cosmos_modes_episodes.jsonl"
OK = C.OK
EXTRA = ("c_pairwise8_fmtrepair",)  # post-hoc format repair, not part of the in-run selection

SELECTION_RULE = (
    "Dev split only, episodes where all four variants completed. success_choice = argmax over "
    f"{list(C.SUCCESS_CANDIDATES)} of balanced_acc_all (unparsed = wrong), ties -> lower mean latency; "
    f"ftype_choice = argmax over {list(C.FTYPE_CANDIDATES)} of failure_type_acc_on_failures_all "
    "(unscored = wrong), ties -> lower mean latency. Both choices are applied unchanged to eval. "
    "Amended during the dev phase (before selection, never using eval): c_pairwise8_fmtrepair (same raw "
    "outputs, post-hoc format repair) added to both candidate sets after Cosmos3-Edge answered the pair "
    "prompt in a bracketed-list format on many dev calls."
)


_KV = re.compile(r'"(object_motion|held_in_image_1|held_in_image_2|location_in_image_2)"\s*:\s*("([^"]*)"|true|false)')


def parse_pair_repaired(raw: str, task: str):
    """Post-hoc FORMAT repair for c_pairwise8 (added after seeing dev outputs; values untouched).

    Cosmos3-Edge often answers the pair prompt with a bracketed list ('[ "object_motion": ... ]') or a
    list of one-key objects; the four "key": value pairs are extracted by regex.  Same as
    x5_common.parse_pair_repaired (duplicated here so the running GPU process can import it).
    """
    from cosmos_plv.visual_diagnostics import ParsedDiagnostic, split_reasoning

    p = C.parse_pair(raw, task)
    if p.parse_status in OK:
        return p
    _, answer = split_reasoning(raw or "")
    found: Dict[str, Any] = {}
    for m in _KV.finditer(answer):
        k, v = m.group(1), m.group(2)
        if k not in found:
            found[k] = True if v == "true" else False if v == "false" else m.group(3)
    if set(found) != set(C.PAIR_SCHEMA):
        return p
    if any(not isinstance(found[k], t) for k, t in C.PAIR_SCHEMA.items()):
        return p
    if found["object_motion"] not in C.MOTIONS or found["location_in_image_2"] not in C.PERCEPTION_LOCATIONS[task]:
        return p
    return ParsedDiagnostic("ok_repaired", found, ["format repair: regex key-value extraction"], None, answer)


def truth_map(root: Path):
    return {(e["checkpoint_id"], int(e["env_seed"])): e for e in C.iter_episodes(root)}


def _frame_pred(rec: Dict[str, Any], from_v7: bool) -> Dict[str, Any]:
    if from_v7:
        return {"parse_status": rec["parse_status"], "object_location": rec.get("p_object_location"),
                "held_by_gripper": rec.get("p_held_by_gripper"), "latency_s": rec.get("latency_s") or 0.0}
    f = rec["fields"] or {}
    return {"parse_status": rec["parse_status"], "object_location": f.get("object_location"),
            "held_by_gripper": f.get("held_by_gripper"), "latency_s": rec["latency_s"] or 0.0}


def episode_pred(variant: str, ep: Dict[str, Any], cache, v7) -> Tuple[Any, float, int]:
    """Returns (pred dict or None if incomplete, latency_s, n_model_calls)."""
    ck, sd, task = ep["checkpoint_id"], int(ep["env_seed"]), ep["task"]
    get = lambda unit: cache.get((variant, ck, sd, unit))  # noqa: E731
    if variant == "d_reason_last":
        r = get(f"f{ep['video_frames'] - 1:03d}")
        if r is None:
            return None, 0.0, 0
        ok = r["parse_status"] in OK
        return ({"parse_status": "ok" if ok else r["parse_status"],
                 "task_success": (r["fields"] or {}).get("task_success") if ok else None},
                r["latency_s"] or 0.0, 1)
    if variant in ("a_roi_track8", "b_track15"):
        k = 8 if variant == "a_roi_track8" else 15
        have = v7.get((ck, sd), {}) if variant == "b_track15" else {}
        frames = []
        for j in C.keyframe_indices(ep["video_frames"], k):
            if j in have:
                frames.append(_frame_pred(have[j], True))
                continue
            r = get(f"f{j:03d}")
            if r is None:
                return None, 0.0, 0
            frames.append(_frame_pred(r, False))
        return C.aggregate_frames(task, frames), sum(f["latency_s"] for f in frames), len(frames)
    if variant in ("c_pairwise8", "c_pairwise8_fmtrepair"):
        idx = C.keyframe_indices(ep["video_frames"], 8)
        pairs = []
        for i in range(len(idx) - 1):
            r = cache.get(("c_pairwise8", ck, sd, f"p{idx[i]:03d}_{idx[i + 1]:03d}"))
            if r is None:
                return None, 0.0, 0
            if variant == "c_pairwise8_fmtrepair":
                pp = parse_pair_repaired(r["raw_text"], task)
                r = {**r, "parse_status": pp.parse_status, "fields": pp.fields}
            pairs.append(r)
        return C.aggregate_pairs(task, pairs), sum(p["latency_s"] or 0 for p in pairs), len(pairs)
    raise KeyError(variant)


def variant_rows(cache, v7, keys: List[Tuple[str, int]], root: Path) -> Dict[str, List[Dict[str, Any]]]:
    tm = truth_map(root)
    split = "dev" if root == C.DEV_ROOT else "eval"
    base_file = {"dev": (C.COSMOS_DIR / "dev_prompt_selection.jsonl", C.COSMOS_DIR / "dev_prompt_selection.jsonl"),
                 "eval": (C.COSMOS_DIR / "eval_local_v6.jsonl", C.COSMOS_DIR / "eval_local_v7.jsonl")}[split]
    v6 = C.load_baseline(base_file[0], "v6_perception_last")
    v7b = C.load_baseline(base_file[1], "v7_perception_track")
    out: Dict[str, List[Dict[str, Any]]] = {v: [] for v in C.VARIANTS + EXTRA + ("v6_baseline", "v7_baseline")}
    for key in keys:
        key = (key[0], int(key[1]))
        ep = tm.get(key)
        if ep is None:
            continue
        base = {"checkpoint_id": key[0], "env_seed": key[1], "task": ep["task"], "truth_success": ep["success"],
                "truth_failure_type": ep["failure_type"], "truth_lifted": ep["lifted"]}
        for v in C.VARIANTS + EXTRA:
            pred, lat, ncalls = episode_pred(v, ep, cache, v7)
            if pred is not None:
                out[v].append({**base, "pred": pred, "latency_s": lat, "n_calls": ncalls})
        for name, src in (("v6_baseline", v6), ("v7_baseline", v7b)):
            if key in src:
                r = dict(src[key])
                r.pop("_row", None)
                out[name].append(r)
    return out


def _common(rows: Dict[str, List[Dict[str, Any]]], names) -> List[Tuple[str, int]]:
    sets = [{(r["checkpoint_id"], r["env_seed"]) for r in rows[n]} for n in names]
    return sorted(set.intersection(*sets)) if sets else []


def _subset(rs, keys):
    ks = set(keys)
    return [r for r in rs if (r["checkpoint_id"], r["env_seed"]) in ks]


def select(dev_rows) -> Dict[str, Any]:
    keys = _common(dev_rows, C.VARIANTS)
    sc = {}
    for v in C.VARIANTS + EXTRA:
        m = C.metrics(_subset(dev_rows[v], keys), has_ftype=v != "d_reason_last")
        sc[v] = {k: m.get(k) for k in ("balanced_acc_all", "success_acc_all", "fp", "fn",
                                       "failure_type_acc_on_failures_all", "mean_latency_s_per_episode")}
    s_cands = tuple(C.SUCCESS_CANDIDATES) + EXTRA
    f_cands = tuple(C.FTYPE_CANDIDATES) + EXTRA
    s_choice = sorted(s_cands, key=lambda v: (-(sc[v]["balanced_acc_all"] or 0),
                                              sc[v]["mean_latency_s_per_episode"] or 0))[0]
    f_choice = sorted(f_cands, key=lambda v: (-(sc[v]["failure_type_acc_on_failures_all"] or 0),
                                              sc[v]["mean_latency_s_per_episode"] or 0))[0]
    run_name = {"c_pairwise8_fmtrepair": "c_pairwise8"}
    return {"rule": SELECTION_RULE, "n_dev_episodes": len(keys), "dev_scores": sc,
            "candidates_success": list(s_cands), "candidates_ftype": list(f_cands),
            "success_choice": s_choice, "ftype_choice": f_choice,
            "chosen": sorted({run_name.get(s_choice, s_choice), run_name.get(f_choice, f_choice)})}


def split_report(rows, names) -> Dict[str, Any]:
    """Metrics for each variant on the episodes it shares with v6 (success) and v7 (failure type)."""
    rep: Dict[str, Any] = {}
    for v in names:
        if not rows[v]:
            continue
        ftype = v not in ("d_reason_last", "v6_baseline")
        own = C.metrics(rows[v], has_ftype=ftype)
        entry = {"own_episodes": own}
        if v != "v6_baseline" and rows["v6_baseline"]:
            keys = _common(rows, [v, "v6_baseline"])
            entry["vs_v6_same_episodes"] = {
                "n": len(keys),
                "variant": C.metrics(_subset(rows[v], keys), has_ftype=ftype),
                "v6": C.metrics(_subset(rows["v6_baseline"], keys)),
                "paired_success_diff_variant_minus_v6": C.paired_diff(rows[v], rows["v6_baseline"]),
            }
        if ftype and v != "v7_baseline" and rows["v7_baseline"]:
            keys = _common(rows, [v, "v7_baseline"])
            if keys:
                entry["vs_v7_same_episodes"] = {
                    "n": len(keys),
                    "variant": C.metrics(_subset(rows[v], keys), has_ftype=True),
                    "v7": C.metrics(_subset(rows["v7_baseline"], keys), has_ftype=True),
                    "paired_ftype_diff_variant_minus_v7": C.paired_diff(rows[v], rows["v7_baseline"], field="ft"),
                    "paired_success_diff_variant_minus_v7": C.paired_diff(rows[v], rows["v7_baseline"]),
                }
        rep[v] = entry
    return rep


def load_cache():
    cache = {}
    p = C.SCRATCH / "calls.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            cache[(r["variant"], r["checkpoint_id"], r["env_seed"], r["unit"])] = r
    return cache


def main():
    sys.path.insert(0, str(HERE))
    from x5_cosmos_modes import load_v7_frames

    cache = load_cache()
    v7 = load_v7_frames()
    state = json.loads((C.SCRATCH / "run_state.json").read_text()) if (C.SCRATCH / "run_state.json").exists() else {}
    dev_keys = sorted({(k[1], k[2]) for k in cache if k[1].startswith("dev_")})
    ev_keys = sorted({(k[1], k[2]) for k in cache if not k[1].startswith("dev_")})
    dev_rows = variant_rows(cache, v7, dev_keys, C.DEV_ROOT)
    ev_rows = variant_rows(cache, v7, ev_keys, C.EVAL_ROOT)
    sel = select(dev_rows)
    names = C.VARIANTS + EXTRA + ("v6_baseline", "v7_baseline")
    calls = list(cache.values())
    per_var_calls = {}
    for r in calls:
        d = per_var_calls.setdefault(r["variant"], {"calls": 0, "unparsed": 0, "latency_sum_s": 0.0,
                                                    "prompt_tokens_sum": 0, "completion_tokens_sum": 0,
                                                    "errors": 0})
        d["calls"] += 1
        d["unparsed"] += r["parse_status"] not in OK
        d["errors"] += bool(r["backend_error"])
        d["latency_sum_s"] = round(d["latency_sum_s"] + (r["latency_s"] or 0), 2)
        d["prompt_tokens_sum"] += r["prompt_tokens"] or 0
        d["completion_tokens_sum"] += r["completion_tokens"] or 0
    for d in per_var_calls.values():
        d["mean_latency_per_call_s"] = round(d["latency_sum_s"] / d["calls"], 3)
        d["mean_prompt_tokens"] = round(d["prompt_tokens_sum"] / d["calls"], 1)
        d["mean_completion_tokens"] = round(d["completion_tokens_sum"] / d["calls"], 1)
    res = {
        "name": "x5_cosmos_modes",
        "label": "EXPLORATORY / post-hoc; dev split (36 episodes) used for variant choice, eval split for reporting",
        "model": state.get("model"),
        "roi_box_xyxy_448": C.ROI_BOX, "roi_upsample": C.ROI_UPSAMPLE,
        "selection_in_run": state.get("selection"),
        "selection_recomputed": sel,
        "dev": split_report(dev_rows, names),
        "eval": split_report(ev_rows, names),
        "eval_episodes_done": state.get("eval_episodes_done"),
        "eval_episodes_available_at_run": state.get("eval_episodes_available"),
        "dev_episodes_done": state.get("dev_episodes_done"),
        "per_variant_call_stats": per_var_calls,
        "run_state": {k: state.get(k) for k in ("started_utc", "finished_utc", "dev_wall_s", "eval_wall_s",
                                                  "new_calls", "oom_events", "gpu_free_mib_checks", "env")},
        "notes": [
            "b_track15 reuses the existing v7 per-frame outputs for the 8 shared keyframes "
            "(identical frame indices, prompt, seed 0); only the 7 midpoint frames are new calls.",
            "Latency per episode = sum of model-call latencies (b_track15 includes the reused v7 frame latencies).",
            "failure_type metrics are on true failures; unscored episodes count as wrong (*_all).",
            "failure_type_acc_square_other_as_no_grasp is a secondary view: square 'other' labels come from a "
            "below-start-height rule on never-grasped nuts; no variant can output 'other'.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    with open(OUT_EP, "w") as f:
        for split, rows in (("dev", dev_rows), ("eval", ev_rows)):
            for v, rs in rows.items():
                for r in rs:
                    f.write(json.dumps({"split": split, "variant": v, **r}, default=str) + "\n")
    print(json.dumps({"selection": {k: sel[k] for k in ("success_choice", "ftype_choice", "n_dev_episodes")},
                      "dev_scores": sel["dev_scores"]}, indent=1))



# ---------------------------------------------------------------- frame-level diagnostics (added)
def frame_truth(ep: Dict[str, Any]):
    """Per-video-frame simulator truth. video frame k = sim step 2k (clipped to the last step).

    held(k)    finger-pad contact AND object >= 2 cm above its resting height (z at sim step 10;
               the square nut spawns ~6 cm above the table and settles during the first steps)
    success(k) simulator success flag at that step
    """
    import numpy as np

    e = np.load(C.ROOT / ep["episode_npz"])
    z = e["object_pos"][:, 2]
    rest = float(z[min(10, len(z) - 1)])
    T = len(z) - 1

    def at(k):
        s = min(2 * k, T)
        return {"held": bool(e["grasped"][s]) and float(z[s] - rest) >= 0.02,
                "success": bool(e["success"][s]), "dz": float(z[s] - rest)}
    return at


def frame_level(cache, v7, keys, root) -> Dict[str, Any]:
    tm = truth_map(root)
    acc: Dict[str, Dict[str, int]] = {}

    def add(name, pred_held, pred_succ, tr):
        d = acc.setdefault(name, {"frames": 0, "held_tp": 0, "held_fp": 0, "held_fn": 0, "held_tn": 0,
                                  "succloc_correct": 0})
        d["frames"] += 1
        d["held_tp" if pred_held and tr["held"] else "held_fp" if pred_held else
          "held_fn" if tr["held"] else "held_tn"] += 1
        d["succloc_correct"] += int(bool(pred_succ) == tr["success"])

    for key in keys:
        ep = tm.get((key[0], int(key[1])))
        if ep is None:
            continue
        at = frame_truth(ep)
        succ_loc = C.SUCCESS_LOCATION[ep["task"]]
        have = v7.get((ep["checkpoint_id"], int(ep["env_seed"])), {})
        for j in C.keyframe_indices(ep["video_frames"], 8):
            tr = at(j)
            if j in have and have[j]["parse_status"] in OK:
                f = have[j]
                add("v7_frames(8kf,full)", bool(f.get("p_held_by_gripper")) or f.get("p_object_location") == "in_gripper",
                    f.get("p_object_location") == succ_loc, tr)
                add("v7_frames(8kf,full)[held:=location==in_gripper only]", f.get("p_object_location") == "in_gripper",
                    f.get("p_object_location") == succ_loc, tr)
            r = cache.get(("a_roi_track8", ep["checkpoint_id"], int(ep["env_seed"]), f"f{j:03d}"))
            if r and r["parse_status"] in OK:
                f = r["fields"]
                add("a_roi_frames(8kf,crop)", bool(f.get("held_by_gripper")) or f.get("object_location") == "in_gripper",
                    f.get("object_location") == succ_loc, tr)
                add("a_roi_frames(8kf,crop)[held:=location==in_gripper only]", f.get("object_location") == "in_gripper",
                    f.get("object_location") == succ_loc, tr)
        for j in C.keyframe_indices(ep["video_frames"], 15):
            r = cache.get(("b_track15", ep["checkpoint_id"], int(ep["env_seed"]), f"f{j:03d}"))
            if r and r["parse_status"] in OK:
                f = r["fields"]
                add("b_midpoint_frames(full)", bool(f.get("held_by_gripper")) or f.get("object_location") == "in_gripper",
                    f.get("object_location") == succ_loc, at(j))
        idx = C.keyframe_indices(ep["video_frames"], 8)
        for i in range(len(idx) - 1):
            r = cache.get(("c_pairwise8", ep["checkpoint_id"], int(ep["env_seed"]), f"p{idx[i]:03d}_{idx[i + 1]:03d}"))
            if r and r["parse_status"] in OK:
                f = r["fields"]
                t0, t1 = at(idx[i]), at(idx[i + 1])
                add("c_pair_image2", bool(f["held_in_image_2"]) or f["location_in_image_2"] == "in_gripper",
                    f["location_in_image_2"] == succ_loc, t1)
                d = acc.setdefault("c_pair_motion_up", {"pairs": 0, "up_tp": 0, "up_fp": 0, "up_fn": 0, "up_tn": 0})
                d["pairs"] += 1
                true_up = (t1["dz"] - t0["dz"]) >= 0.03
                pu = f["object_motion"] == "up"
                d["up_tp" if pu and true_up else "up_fp" if pu else "up_fn" if true_up else "up_tn"] += 1
    for name, d in acc.items():
        if "held_tp" in d:
            npos = d["held_tp"] + d["held_fn"]
            npred = d["held_tp"] + d["held_fp"]
            d["held_precision"] = round(d["held_tp"] / npred, 3) if npred else None
            d["held_precision_ci95"] = C.wilson(d["held_tp"], npred)
            d["held_recall"] = round(d["held_tp"] / npos, 3) if npos else None
            d["held_recall_ci95"] = C.wilson(d["held_tp"], npos)
            d["held_fp_rate_on_not_held"] = round(d["held_fp"] / max(1, d["held_fp"] + d["held_tn"]), 3)
            d["succloc_acc"] = round(d["succloc_correct"] / d["frames"], 3)
        else:
            npos = d["up_tp"] + d["up_fn"]
            npred = d["up_tp"] + d["up_fp"]
            d["up_precision"] = round(d["up_tp"] / npred, 3) if npred else None
            d["up_recall"] = round(d["up_tp"] / npos, 3) if npos else None
    return acc


def main_with_frames():
    main()
    from x5_cosmos_modes import load_v7_frames

    res = json.loads(OUT.read_text())
    cache = load_cache()
    v7 = load_v7_frames()
    dev_keys = sorted({(k[1], k[2]) for k in cache if k[1].startswith("dev_")})
    ev_keys = sorted({(k[1], k[2]) for k in cache if not k[1].startswith("dev_")})
    res["frame_level"] = {"truth_rule": frame_truth.__doc__.strip(),
                          "dev": frame_level(cache, v7, dev_keys, C.DEV_ROOT),
                          "eval": frame_level(cache, v7, ev_keys, C.EVAL_ROOT)}
    res["majority_failure_type_baseline"] = {
        "note": "reference only: accuracy on true failures of always answering 'no_grasp' (majority type)",
        "dev": majority_baseline(dev_keys, C.DEV_ROOT), "eval": majority_baseline(ev_keys, C.EVAL_ROOT)}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res["frame_level"], indent=1)[:3000])
    print(json.dumps(res["majority_failure_type_baseline"]))


def majority_baseline(keys, root):
    tm = truth_map(root)
    fails = [tm[(k[0], int(k[1]))]["failure_type"] for k in keys
             if (k[0], int(k[1])) in tm and not tm[(k[0], int(k[1]))]["success"]]
    cnt: Dict[str, int] = {}
    for f in fails:
        cnt[f] = cnt.get(f, 0) + 1
    k = cnt.get("no_grasp", 0)
    return {"n_failures": len(fails), "counts": cnt,
            "always_no_grasp_acc": round(k / len(fails), 3) if fails else None,
            "ci95": C.wilson(k, len(fails))}


if __name__ == "__main__":
    main_with_frames()
