#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION of exploration x4_nemotron_modes (EXPLORATORY / POST-HOC).

Read-only with respect to x4's files: re-parses x4's saved raw model outputs, recomputes every headline number with
independent code, and adds checks x4 did not run (clip-length leakage baseline, checkpoint-cluster bootstrap,
per-task Spearman, sim-state audit of the "square other" artifact and of the "lift hallucination" mechanism).

  rescore  CPU only, no API calls -> results/hackathon/explore/x4_nemotron_modes_verification.json
  probe    <= 30 Nebius calls (clip-length probe + run-to-run repeat), rows in
           data/outputs/hackathon_explore/x4_nemotron_modes_verify/probe.jsonl

Run from the repo root with Env B:
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/x4_nemotron_modes_verify.py rescore
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))

import x4_nemotron_modes as X  # noqa: E402  (import has no side effects besides chdir(ROOT))
from cosmos_plv.failure_annotator import annotate_episode  # noqa: E402
from cosmos_plv.ranking import spearman  # noqa: E402
from cosmos_plv.visual_diagnostics import parse_diagnostic, build_prompt  # noqa: E402

NAME = "x4_nemotron_modes_verify"
X4 = ROOT / "data/outputs/hackathon_explore/x4_nemotron_modes"
SCRATCH = ROOT / "data/outputs/hackathon_explore" / NAME
OUT = ROOT / "results/hackathon/explore/x4_nemotron_modes_verification.json"
EVIDENCE = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
PROBE = SCRATCH / "probe.jsonl"
PROBE_BUDGET = 30
GOAL = {"can": "right_bin_front_right", "square": "on_square_peg_low"}


# ----------------------------------------------------------------------------- helpers

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 3), round(c + h, 3)]


def episodes(split):
    out = {}
    for s in sorted(glob.glob(str(ROOT / f"data/outputs/hackathon_rollouts_{split}/*/summary.json"))):
        for ep in json.load(open(s))["episodes"]:
            out[(ep["checkpoint_id"], int(ep["env_seed"]))] = ep
    return out


def jl(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def key(r):
    return (r["checkpoint_id"], int(r["env_seed"]))


def reparse(row, variant):
    """Independent re-derivation of (scored, success, failure_type, track) from raw_text."""
    task, raw = row["task"], row.get("raw_text") or ""
    if variant in ("A_track8", "D_grid8"):
        p, track = X.parse_track(raw, task, 8, "image" if variant == "A_track8" else "panel")
        return p, track
    if variant == "C_vidcomp":
        return X.parse_vidcomp(raw, task), None
    if variant in ("V2_frozen", "B_v2reason"):
        p, _ = X.parse_diagnostic_repaired(raw)
        return p, None
    return parse_diagnostic(raw), None  # frozen file rows: the original parser


def my_r2(task, track):
    """Own re-implementation of rule r2 from the report text (lifted := some keyframe location == in_gripper)."""
    valid = [t for t in track if t.get("valid")]
    final = valid[-1]["location"]
    lifted = any(t["location"] == "in_gripper" for t in valid)
    held_last = bool(valid[-1]["held"]) or final == "in_gripper"
    if final == GOAL[task]:
        return True, "none"
    if not lifted:
        return False, ("other" if final in ("fell_to_floor", "off_table") else "no_grasp")
    return False, ("not_placed" if held_last else "dropped")


def predictions(rows, variant, truth_eps, r2=False):
    """key -> dict(scored, succ, ft, final, track) recomputed from raw text; also returns mismatch counts."""
    pred, mism = {}, collections.Counter()
    for r in rows:
        if r.get("backend_error") and "truth_success" not in r:
            continue
        k = key(r)
        p, track = reparse(r, variant)
        scored = p.parse_status in ("ok", "ok_repaired") and bool(p.fields)
        succ = bool(p.fields.get("task_success")) if scored else None
        ft = p.fields.get("failure_type") if scored else None
        if r2 and scored and track:
            succ, ft = my_r2(r["task"], track)
        if not r2:
            mism["stored_scored_mismatch"] += int(bool(r.get("scored")) != scored)
            mism["stored_success_mismatch"] += int(scored and bool(r.get("cosmos_task_success")) != succ)
            mism["stored_ft_mismatch"] += int(scored and r.get("cosmos_failure_type") != ft)
        ep = truth_eps.get(k)
        if ep is None:
            mism["row_not_in_current_summary"] += 1
            continue
        mism["truth_success_mismatch_vs_summary"] += int(bool(r["truth_success"]) != bool(ep["success"]))
        mism["truth_ft_mismatch_vs_summary"] += int(r["truth_failure_type"] != ep["failure_type"])
        pred[k] = {"scored": scored, "succ": succ, "ft": ft, "track": track,
                   "final": (p.fields.get("final_location") if scored else None), "status": p.parse_status,
                   "latency": r.get("latency_s"), "pt": r.get("prompt_tokens"), "ct": r.get("completion_tokens")}
    return pred, dict(mism)


def frozen_preds(split, truth_eps):
    src = X.FROZEN_V2[split]
    rows = [r for r in jl(src) if r.get("backend") == "nebius" and r.get("variant") == "v2_evidence_first"
            and not r.get("reasoning")]
    p1, m1 = predictions(rows, "frozen_file", truth_eps)
    out = {"from_frozen_file": len(p1), "mismatch_frozen_file": m1}
    if split == "eval":
        p2, m2 = predictions(jl(X4 / "eval_V2_frozen.jsonl"), "V2_frozen", truth_eps)
        out["from_V2_frozen_rerun"] = len([k for k in p2 if k not in p1])
        out["mismatch_V2_frozen"] = m2
        for k, v in p2.items():
            p1.setdefault(k, v)
    return p1, out


def ok_flags(pred, eps, keys, label=None):
    s = np.array([pred[k]["scored"] and pred[k]["succ"] == bool(eps[k]["success"]) for k in keys], float)
    lab = label or (lambda k: eps[k]["failure_type"])
    f = np.array([pred[k]["scored"] and pred[k]["ft"] == lab(k) for k in keys], float)
    return s, f


def summarize(pred, eps, keys, harm=None):
    keys = [k for k in keys if k in pred]
    n = len(keys)
    sc = [k for k in keys if pred[k]["scored"]]
    tp = sum(1 for k in sc if pred[k]["succ"] and eps[k]["success"])
    tn = sum(1 for k in sc if not pred[k]["succ"] and not eps[k]["success"])
    fp = sum(1 for k in sc if pred[k]["succ"] and not eps[k]["success"])
    fn = sum(1 for k in sc if not pred[k]["succ"] and eps[k]["success"])
    npos = sum(1 for k in keys if eps[k]["success"])
    nneg = n - npos
    fails = [k for k in keys if not eps[k]["success"]]
    ftok = sum(1 for k in fails if pred[k]["scored"] and pred[k]["ft"] == eps[k]["failure_type"])
    out = {"n": n, "unparsed": n - len(sc), "success_acc": round((tp + tn) / n, 3) if n else None,
           "success_acc_wilson": wilson(tp + tn, n), "fp": fp, "fn": fn,
           "balanced_acc": round(0.5 * (tp / max(1, npos) + tn / max(1, nneg)), 3),
           "n_failures": len(fails), "ft_acc_on_failures": round(ftok / len(fails), 3) if fails else None,
           "ft_acc_wilson": wilson(ftok, len(fails)), "ft_correct": ftok}
    if harm is not None:
        hok = sum(1 for k in fails if pred[k]["scored"] and pred[k]["ft"] == harm[k])
        out["harmonized_ft_acc"] = round(hok / len(fails), 3) if fails else None
        out["harmonized_ft_wilson"] = wilson(hok, len(fails))
    lat = [pred[k]["latency"] for k in keys if pred[k].get("latency")]
    pt = [pred[k]["pt"] for k in keys if pred[k].get("pt")]
    ct = [pred[k]["ct"] for k in keys if pred[k].get("ct")]
    out.update({"mean_latency_s": round(float(np.mean(lat)), 2) if lat else None,
                "mean_prompt_tokens": round(float(np.mean(pt)), 1) if pt else None,
                "mean_completion_tokens": round(float(np.mean(ct)), 1) if ct else None})
    return out


def paired(pa, pb, eps, keys, label=None, n_boot=10000, seed=0):
    """Episode bootstrap (as x4) AND checkpoint-cluster bootstrap of mean(a_correct - b_correct)."""
    from scipy.stats import binomtest
    res = {}
    for metric in ("success", "ft_on_failures"):
        ks = [k for k in keys if k in pa and k in pb and (metric == "success" or not eps[k]["success"])]
        sa, fa = ok_flags(pa, eps, ks, label)
        sb, fb = ok_flags(pb, eps, ks, label)
        a, b = (sa, sb) if metric == "success" else (fa, fb)
        d = a - b
        rng = np.random.default_rng(seed)
        boots = d[rng.integers(0, len(ks), size=(n_boot, len(ks)))].mean(axis=1)
        ck = sorted({k[0] for k in ks})
        by = {c: np.array([d[i] for i, k in enumerate(ks) if k[0] == c]) for c in ck}
        rng = np.random.default_rng(seed + 1)
        cb = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(ck), size=len(ck))
            v = np.concatenate([by[ck[i]] for i in pick])
            cb.append(v.mean() if len(v) else 0.0)
        n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
        res[metric] = {"n": len(ks), "n_checkpoints": len(ck), "a": round(a.mean(), 3), "b": round(b.mean(), 3),
                       "diff": round(d.mean(), 3),
                       "episode_bootstrap_ci95": [round(float(np.quantile(boots, q)), 3) for q in (0.025, 0.975)],
                       "checkpoint_cluster_bootstrap_ci95": [round(float(np.quantile(cb, q)), 3)
                                                             for q in (0.025, 0.975)],
                       "a_only": n10, "b_only": n01, "mcnemar_exact_p": round(p, 4)}
    return res


def ckpt_view(preds, eps, keys, n_boot=5000, seed=0):
    """Per-checkpoint VLM success rate vs simulator success rate; overall and per task; bootstrap over checkpoints."""
    ck = sorted({k[0] for k in keys})
    rate = lambda p, c: float(np.mean([bool(p[k]["scored"]) and bool(p[k]["succ"]) for k in keys if k[0] == c]))  # noqa
    truth = {c: float(np.mean([bool(eps[k]["success"]) for k in keys if k[0] == c])) for c in ck}
    out = {}
    for scope in ("all", "can", "square"):
        cs = [c for c in ck if scope == "all" or c.startswith(scope)]
        t = [truth[c] for c in cs]
        row = {"n_checkpoints": len(cs)}
        est = {name: [rate(p, c) for c in cs] for name, p in preds.items()}
        for name, v in est.items():
            row[f"spearman_{name}"] = round(spearman(v, t), 3)
            row[f"mae_{name}"] = round(float(np.mean(np.abs(np.array(v) - np.array(t)))), 3)
        if len(preds) >= 2 and len(cs) >= 4:
            names = list(preds)
            rng = np.random.default_rng(seed)
            diffs = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(cs), size=len(cs))
                tt = [t[i] for i in idx]
                if len(set(tt)) < 2:
                    continue
                s0 = spearman([est[names[0]][i] for i in idx], tt)
                s1 = spearman([est[names[1]][i] for i in idx], tt)
                if np.isfinite(s0) and np.isfinite(s1):
                    diffs.append(s1 - s0)
            row[f"spearman_diff_{names[1]}_minus_{names[0]}_ci95"] = (
                [round(float(np.quantile(diffs, q)), 3) for q in (0.025, 0.975)] if diffs else None)
        out[scope] = row
    return out


# ----------------------------------------------------------------------------- sim-state audits

def square_audit(eps):
    rows = []
    for k, ep in sorted(eps.items()):
        if ep["task"] != "square":
            continue
        z = np.load(ROOT / ep["episode_npz"])
        obj, eef, g = z["object_pos"], z["eef_pos"], z["grasped"].astype(bool)
        tr = {kk: z[kk][X.SETTLE_STEPS:] for kk in ("eef_pos", "object_pos", "grasped", "success")}
        rows.append({
            "checkpoint_id": k[0], "env_seed": k[1], "official": ep["failure_type"],
            "harmonized": annotate_episode(tr)["failure_type"],
            "z0": round(float(obj[0, 2]), 4), "z5": round(float(obj[5, 2]), 4), "z10": round(float(obj[10, 2]), 4),
            "z_end": round(float(obj[-1, 2]), 4), "ever_grasped": bool(g.any()),
            "xy_move_after_settle_m": round(float(np.linalg.norm(obj[-1, :2] - obj[10, :2])), 4),
            "max_rise_above_settled_m": round(float((obj[10:, 2] - obj[10, 2]).max()), 4),
            "min_eef_obj_dist_m": round(float(np.linalg.norm(eef - obj, axis=1).min()), 4)})
    others = [r for r in rows if r["official"] == "other"]
    return {
        "n_square": len(rows),
        "spawn_minus_settled_z_m_all_square": [round(float(np.min([r["z0"] - r["z10"] for r in rows])), 4),
                                               round(float(np.max([r["z0"] - r["z10"] for r in rows])), 4)],
        "z5_minus_z10_max_m": round(float(np.max([abs(r["z5"] - r["z10"]) for r in rows])), 4),
        "other_n": len(others),
        "other_ever_grasped": sum(r["ever_grasped"] for r in others),
        "other_harmonized": dict(collections.Counter(r["harmonized"] for r in others)),
        "other_xy_move_after_settle_m_max": max([r["xy_move_after_settle_m"] for r in others], default=None),
        "other_max_rise_above_settled_m_max": max([r["max_rise_above_settled_m"] for r in others], default=None),
        "other_min_eef_obj_dist_m_min": min([r["min_eef_obj_dist_m"] for r in others], default=None),
        "other_rows": others,
        "label_changes": [r for r in rows if r["official"] != r["harmonized"]],
    }


def lift_audit(pred_r2, eps):
    """For truth no_grasp episodes that A_track8_r2 called 'dropped': is the object physically raised at the keyframe
    the model calls in_gripper?  Video frame k = sim step 2k; keyframe index j is in the kfNNN.jpg filename order."""
    out = []
    for k, p in sorted(pred_r2.items()):
        ep = eps[k]
        if ep["failure_type"] != "no_grasp" or p["ft"] != "dropped":
            continue
        z = np.load(ROOT / ep["episode_npz"])
        obj, eef, g = z["object_pos"], z["eef_pos"], z["grasped"].astype(bool)
        n_f = int(ep["video_frames"])
        idx = [int(round(x)) for x in np.linspace(0, n_f - 1, 8)]
        base_z = float(obj[min(X.SETTLE_STEPS, len(obj) - 1), 2])
        flagged = []
        for t in p["track"]:
            if t.get("valid") and t["location"] == "in_gripper":
                s = min(2 * idx[t["i"]], len(obj) - 1)
                flagged.append({"keyframe": t["i"] + 1, "video_frame": idx[t["i"]], "sim_step": s,
                                "rise_m": round(float(obj[s, 2] - base_z), 4),
                                "eef_obj_dist_m": round(float(np.linalg.norm(eef[s] - obj[s])), 4),
                                "grasped_at_step": bool(g[s])})
        out.append({"checkpoint_id": k[0], "env_seed": k[1], "task": ep["task"],
                    "max_rise_above_settled_m": round(float((obj[X.SETTLE_STEPS:, 2] - base_z).max()), 4),
                    "ever_grasped": bool(g.any()), "flagged_in_gripper_keyframes": flagged})
    raised = [o for o in out if any(f["rise_m"] >= 0.02 for f in o["flagged_in_gripper_keyframes"])]
    near = [o for o in out if all(f["eef_obj_dist_m"] < 0.08 for f in o["flagged_in_gripper_keyframes"])]
    return {"n": len(out), "n_object_raised_ge_2cm_at_a_flagged_keyframe": len(raised),
            "n_ever_grasped_contact": sum(o["ever_grasped"] for o in out),
            "n_all_flagged_keyframes_eef_within_8cm": len(near), "rows": out}


# ----------------------------------------------------------------------------- rescore

def cmd_rescore(a):
    res = {"name": NAME, "status": "ADVERSARIAL VERIFICATION, EXPLORATORY / POST-HOC", "verifies": "x4_nemotron_modes"}
    ev_now, dev_now = episodes("eval"), episodes("dev")
    # The repo's failure_annotator was changed (settled-height fix, commit baa3967, 2026-10-29 05:15 KST) and every
    # summary.json was re-annotated WHILE this verification ran.  The main reproduction therefore uses the labels
    # x4 scored against (truth_* stored in x4's rows); a separate block re-scores with the current labels.
    # x4's own scratch rows were NOT touched by refresh_truth.py (it rewrites data/outputs/hackathon_cosmos/*.jsonl only)
    lab_rows = jl(X4 / "eval_A_track8.jsonl") + jl(X4 / "eval_C_vidcomp.jsonl") + jl(X4 / "eval_V2_frozen.jsonl")
    dev_rows = jl(X4 / "dev_A_track8.jsonl")

    def x4_labels(now, rows):
        out = {k: dict(v) for k, v in now.items()}
        for r in rows:
            if "truth_failure_type" in r and key(r) in out:
                out[key(r)]["failure_type"] = r["truth_failure_type"]
                out[key(r)]["success"] = bool(r["truth_success"])
        return out
    ev_eps, dev_eps = x4_labels(ev_now, lab_rows), x4_labels(dev_now, dev_rows)
    res["label_versions"] = {
        "main_tables_use": "labels stored in x4's rows (annotator before commit baa3967)",
        "eval_label_changes_since_x4": dict(collections.Counter(
            f"{ev_eps[k]['failure_type']}->{ev_now[k]['failure_type']}" for k in ev_now
            if ev_eps[k]["failure_type"] != ev_now[k]["failure_type"])),
        "eval_success_changes_since_x4": sum(ev_eps[k]["success"] != ev_now[k]["success"] for k in ev_now),
        "dev_label_changes_since_x4": dict(collections.Counter(
            f"{dev_eps[k]['failure_type']}->{dev_now[k]['failure_type']}" for k in dev_now
            if dev_eps[k]["failure_type"] != dev_now[k]["failure_type"])),
    }
    res["eval_episodes_now"] = len(ev_eps)
    res["eval_checkpoints_now"] = len({k[0] for k in ev_eps})

    # --- integrity: re-parse every saved raw output
    A, mA = predictions(jl(X4 / "eval_A_track8.jsonl"), "A_track8", ev_eps)
    A2, _ = predictions(jl(X4 / "eval_A_track8.jsonl"), "A_track8", ev_eps, r2=True)
    C, mC = predictions(jl(X4 / "eval_C_vidcomp.jsonl"), "C_vidcomp", ev_eps)
    V, mV = frozen_preds("eval", ev_eps)
    res["integrity_eval"] = {"A_track8": mA, "C_vidcomp": mC, "frozen_v2": mV}
    # r2 vs x4's own rederive_r2
    x4r2 = {key(r): X.rederive_r2(r) for r in jl(X4 / "eval_A_track8.jsonl")}
    res["integrity_eval"]["r2_mismatch_vs_x4_rederive"] = sum(
        1 for k in A2 if x4r2[k].get("scored") and x4r2[k]["cosmos_failure_type"] != A2[k]["ft"])

    common = sorted(set(A) & set(C) & set(V))
    # x4's harmonized label as stored in its rows (old annotator on trace[10:]); recomputed (current annotator on
    # trace[10:]) only for episodes x4 never scored
    harm = {k: annotate_episode({kk: np.load(ROOT / ev_eps[k]["episode_npz"])[kk][X.SETTLE_STEPS:]
                                 for kk in ("eef_pos", "object_pos", "grasped", "success")})["failure_type"]
            for k in ev_eps}
    harm_recomputed_now = dict(harm)
    for r in jl(X4 / "eval_A_track8.jsonl"):
        if r.get("harmonized_failure_type"):
            harm[key(r)] = r["harmonized_failure_type"]
    res["harmonized_stored_vs_recomputed_now_disagree"] = sum(harm[k] != harm_recomputed_now[k] for k in harm)
    res["eval_common_episodes"] = len(common)
    res["eval_common_checkpoints"] = len({k[0] for k in common})
    res["eval_common_failures_by_checkpoint"] = dict(collections.Counter(k[0] for k in common
                                                                         if not ev_eps[k]["success"]))
    res["eval_excluded_from_paired"] = sorted({k[0] for k in ev_eps if k not in set(common)})
    preds = {"frozen_v2": V, "A_track8_r2": A2, "A_track8_r1": A, "C_vidcomp": C}
    tab = {}
    for name, p in preds.items():
        tab[name] = {"all": summarize(p, ev_eps, common, harm),
                     "can": summarize(p, ev_eps, [k for k in common if ev_eps[k]["task"] == "can"], harm),
                     "square": summarize(p, ev_eps, [k for k in common if ev_eps[k]["task"] == "square"], harm)}
    res["eval_tables_recomputed"] = tab
    res["eval_paired"] = {
        n: {"official": paired(preds[n], V, ev_eps, common),
            "harmonized": paired(preds[n], V, ev_eps, common, label=lambda k: harm[k])}
        for n in ("A_track8_r2", "A_track8_r1", "C_vidcomp")}
    # confusion
    res["eval_confusion"] = {n: {t: dict(collections.Counter(
        (preds[n][k]["ft"] if preds[n][k]["scored"] else "UNPARSED") for k in common
        if ev_eps[k]["failure_type"] == t)) for t in ("no_grasp", "dropped", "other", "not_placed")}
        for n in ("frozen_v2", "A_track8_r2")}

    # --- clip-length leakage baseline: success := clip shorter than the 400-step timeout clip
    def clip_rule(k):
        return int(ev_eps[k]["video_frames"]) < 201
    clip = {k: {"scored": True, "succ": clip_rule(k), "ft": None} for k in ev_eps}
    n_ok = sum(clip[k]["succ"] == bool(ev_eps[k]["success"]) for k in ev_eps)
    d_ok = sum((int(e["video_frames"]) < 201) == bool(e["success"]) for e in dev_eps.values())
    res["clip_length_baseline"] = {
        "rule": "predict success iff video_frames < 201 (episodes stop 16 steps after the simulator's first success)",
        "eval_all": {"n": len(ev_eps), "correct": n_ok, "acc": round(n_ok / len(ev_eps), 3),
                     "wilson": wilson(n_ok, len(ev_eps))},
        "eval_common": summarize(clip, ev_eps, common),
        "dev": {"n": len(dev_eps), "correct": d_ok, "acc": round(d_ok / len(dev_eps), 3)},
        "failure_frames": sorted({int(e["video_frames"]) for e in ev_eps.values() if not e["success"]}),
        "success_frames_range": [min(int(e["video_frames"]) for e in ev_eps.values() if e["success"]),
                                 max(int(e["video_frames"]) for e in ev_eps.values() if e["success"])],
        "prompts_reveal_duration": {
            "frozen_v2": "'The clip lasts about {d:.1f} seconds.' in build_prompt",
            "A_track8": "per-image timestamps 'image k at t=..s' (last one = clip length)",
            "C_vidcomp": "'(about {d:.1f} s)' in build_vidcomp_prompt",
        },
    }
    res["clip_length_baseline"]["checkpoint_view"] = ckpt_view({"clip_rule": clip, "frozen_v2": V}, ev_eps, common)

    # --- checkpoint (PLV) view, per task, with bootstrap over checkpoints
    res["checkpoint_view"] = {
        "A_track8_r2_vs_v2": ckpt_view({"frozen_v2": V, "A_track8_r2": A2}, ev_eps, common),
        "C_vidcomp_vs_v2": ckpt_view({"frozen_v2": V, "C_vidcomp": C}, ev_eps, common),
    }

    # --- hard cases (x4's sim-state definitions, recomputed)
    hard = {}
    for k in common:
        hc = X.truth_extras(ev_eps[k])["hard_case"]
        if hc:
            hard.setdefault(hc, []).append(k)
    res["hard_cases"] = {}
    for hc, ks in hard.items():
        res["hard_cases"][hc] = {
            "n": len(ks), "checkpoints": dict(collections.Counter(k[0] for k in ks)),
            "flagged_as_failure": {n: sum(1 for k in ks if preds[n][k]["scored"] and not preds[n][k]["succ"])
                                   for n in preds},
            "final_location_C": [C[k]["final"] for k in ks], "final_location_A": [A[k]["final"] for k in ks]}
    can_fp_v2 = [k for k in common if ev_eps[k]["task"] == "can" and V[k]["succ"] and not ev_eps[k]["success"]]
    res["hard_cases"]["v2_can_false_positives"] = {
        "n": len(can_fp_v2), "of_which_wrong_compartment": sum(k in set(hard.get("wrong_compartment", []))
                                                              for k in can_fp_v2)}
    # A can FN, C square FP final locations
    res["A_can_false_negatives_final_location"] = dict(collections.Counter(
        A2[k]["final"] for k in common if ev_eps[k]["task"] == "can" and ev_eps[k]["success"] and not A2[k]["succ"]))
    sq_fp = [k for k in common if ev_eps[k]["task"] == "square" and C[k]["succ"] and not ev_eps[k]["success"]]
    res["C_square_false_positives"] = {
        "n": len(sq_fp), "truth_failure_type": dict(collections.Counter(ev_eps[k]["failure_type"] for k in sq_fp)),
        "truth_end_z_range": [round(min(X.truth_extras(ev_eps[k])["object_end_xyz"][2] for k in sq_fp), 3),
                              round(max(X.truth_extras(ev_eps[k])["object_end_xyz"][2] for k in sq_fp), 3)]
        if sq_fp else None,
        "truth_end_dist_to_square_peg_m": sorted(round(float(np.hypot(X.truth_extras(ev_eps[k])["object_end_xyz"][0]
                                                                      - 0.23, X.truth_extras(ev_eps[k])
                                                                      ["object_end_xyz"][1] - 0.10)), 3)
                                                 for k in sq_fp)}
    # --- design-on-eval check: which hard-case episodes were already listed in experiment2_error_analysis.json
    #     (written before x4 ran; it is where the "wrong compartment" blind spot was identified)
    ea_path = ROOT / "results/hackathon/experiment2_error_analysis.json"
    ea = {(e["checkpoint_id"], int(e["env_seed"])) for e in json.load(open(ea_path))}
    clean = {}
    for hc, ks in hard.items():
        ks_clean = [k for k in ks if k not in ea]
        clean[hc] = {"n_in_error_analysis": len(ks) - len(ks_clean), "n_not_in_error_analysis": len(ks_clean),
                     "flagged_on_clean": {n: sum(1 for k in ks_clean if preds[n][k]["scored"] and
                                                 not preds[n][k]["succ"]) for n in preds}}
    res["hard_cases_design_on_eval_check"] = {"error_analysis_file": str(ea_path.relative_to(ROOT)),
                                              "error_analysis_episodes": len(ea), "by_case": clean}
    # dev hard cases (x4 sim-state definition), per method
    dev_hard = {}
    for k, ep in dev_eps.items():
        hc = X.truth_extras(ep)["hard_case"]
        if hc:
            dev_hard.setdefault(hc, []).append(k)
    res["dev_hard_cases_keys"] = {hc: [list(k) for k in ks] for hc, ks in dev_hard.items()}
    # parser-repair use on eval
    res["eval_parser_repairs"] = {v: sum(1 for r in jl(X4 / f"eval_{v}.jsonl") if r.get("parse_status") == "ok_repaired")
                                  for v in ("A_track8", "C_vidcomp", "V2_frozen")}
    # failure-type accuracy without the three epoch-0 checkpoints (30 of the 59 failures)
    late = [k for k in common if not k[0].endswith("_e0000")]
    res["eval_without_epoch0"] = {
        "n": len(late), "n_failures": sum(1 for k in late if not ev_eps[k]["success"]),
        "tables": {n: {kk: summarize(p, ev_eps, late, harm)[kk] for kk in
                       ("success_acc", "fp", "fn", "ft_acc_on_failures", "ft_acc_wilson", "harmonized_ft_acc")}
                   for n, p in preds.items()},
        "paired_A_track8_r2": paired(A2, V, ev_eps, late)["ft_on_failures"],
        "paired_C_vidcomp": paired(C, V, ev_eps, late)["ft_on_failures"]}
    e0 = [k for k in common if k[0].endswith("_e0000")]
    res["eval_epoch0_only"] = {
        "n": len(e0), "n_failures": sum(1 for k in e0 if not ev_eps[k]["success"]),
        "truth_failure_types": dict(collections.Counter(ev_eps[k]["failure_type"] for k in e0
                                                        if not ev_eps[k]["success"])),
        "tables": {n: {kk: summarize(p, ev_eps, e0, harm)[kk] for kk in
                       ("ft_acc_on_failures", "ft_correct", "ft_acc_wilson", "harmonized_ft_acc")}
                   for n, p in preds.items()},
        "paired_A_track8_r2": paired(A2, V, ev_eps, e0)["ft_on_failures"],
        "paired_C_vidcomp": paired(C, V, ev_eps, e0)["ft_on_failures"]}
    by_type = {}
    for ftype in ("no_grasp", "dropped", "other"):
        sub = [kx for kx in common if ev_eps[kx]["failure_type"] == ftype]
        by_type[ftype] = {}
        for n in ("A_track8_r2", "C_vidcomp"):
            pr = paired(preds[n], V, ev_eps, sub)["ft_on_failures"]
            by_type[ftype][n] = {kk: pr[kk] for kk in ("n", "a", "b", "diff", "a_only", "b_only", "mcnemar_exact_p")}
    res["eval_paired_by_truth_failure_type"] = by_type
    res["eval_without_epoch0"]["truth_failure_types"] = dict(collections.Counter(
        ev_eps[k]["failure_type"] for k in late if not ev_eps[k]["success"]))
    res["square_label_audit"] = square_audit(ev_eps)
    res["square_label_audit_dev"] = {kk: v for kk, v in square_audit(dev_eps).items() if kk != "other_rows"}
    res["lift_hallucination_audit_A_r2"] = lift_audit({k: A2[k] for k in common}, ev_eps)

    # --- dev reproduction + selection rule
    dev = {}
    Vd, mVd = frozen_preds("dev", dev_eps)
    dev["integrity_frozen"] = mVd
    dev_preds = {"frozen_v2": Vd}
    for v in ("A_track8", "B_v2reason", "C_vidcomp", "D_grid8"):
        rows = jl(X4 / f"dev_{v}.jsonl")
        p, m = predictions(rows, v, dev_eps)
        dev_preds[v] = p
        dev[f"integrity_{v}"] = m
        if v in ("A_track8", "D_grid8"):
            dev_preds[v + "_r2"], _ = predictions(rows, v, dev_eps, r2=True)
        if v == "B_v2reason":
            dev["B_repaired_count"] = sum(1 for r in rows if "missing opening quote" in (r.get("parse_errors") or ""))
    dk = sorted(dev_eps)
    dev["tables"] = {n: summarize(p, dev_eps, dk) for n, p in dev_preds.items()}
    dev["hard_cases_flagged_as_failure"] = {
        hc: {n: sum(1 for k in ks if p[k]["scored"] and not p[k]["succ"]) for n, p in dev_preds.items()}
        for hc, ks in dev_hard.items()}
    dev["paired_B_vs_v2"] = paired(dev_preds["B_v2reason"], Vd, dev_eps, dk)
    floor = dev["tables"]["frozen_v2"]["balanced_acc"] - 0.05
    elig = sorted(((n, t["ft_acc_on_failures"], t["balanced_acc"]) for n, t in dev["tables"].items()
                   if n != "frozen_v2" and t["n"] == 36 and t["balanced_acc"] >= floor),
                  key=lambda x: (x[1], x[2]), reverse=True)
    dev["selection_reproduced"] = {"floor": round(floor, 3), "eligible_ranked": elig}
    elig_no_r2 = [e for e in elig if not e[0].endswith("_r2")]
    dev["selection_if_r2_never_added"] = [e[0] for e in elig_no_r2[:2]]
    res["dev"] = dev

    # --- call accounting
    tags = collections.Counter()
    errs = 0
    for r in jl(EVIDENCE):
        t = r.get("tag", "")
        if t.startswith("explore/x4_nemotron_modes/"):
            parts = t.split("/")
            tags["/".join(parts[2:4])] += 1
            errs += int(bool(r.get("error")))
    res["x4_calls_by_split_variant"] = dict(tags)
    res["x4_calls_total"] = sum(tags.values())
    res["x4_calls_with_backend_error"] = errs

    # --- re-score with the CURRENT official labels (annotator after the settled-height fix)
    cur = {"tables": {}, "paired": {}, "confusion": {}}
    for name, p in preds.items():
        cur["tables"][name] = {kk: summarize(p, ev_now, common)[kk] for kk in
                               ("success_acc", "fp", "fn", "n_failures", "ft_acc_on_failures", "ft_acc_wilson")}
        if name != "frozen_v2":
            pr = paired(p, V, ev_now, common)["ft_on_failures"]
            cur["paired"][name] = {kk: pr[kk] for kk in ("n", "diff", "episode_bootstrap_ci95",
                                                         "checkpoint_cluster_bootstrap_ci95", "a_only", "b_only",
                                                         "mcnemar_exact_p")}
        cur["confusion"][name] = {t: dict(collections.Counter((p[k]["ft"] if p[k]["scored"] else "UNPARSED")
                                                              for k in common if ev_now[k]["failure_type"] == t))
                                  for t in ("no_grasp", "dropped", "not_placed", "other")}
    cur["truth_failure_types_common"] = dict(collections.Counter(ev_now[k]["failure_type"] for k in common
                                                                 if not ev_now[k]["success"]))
    cur["agreement_x4_harmonized_vs_new_official"] = sum(harm[k] == ev_now[k]["failure_type"] for k in common
                                                         if not ev_now[k]["success"])
    res["rescored_with_current_labels"] = cur

    # --- what x4's own `score` would print if re-run now (read-only re-use of x4's functions; no cache writes):
    #     load_frozen() reads the refreshed eval_nebius_v2.jsonl (new labels) while x4's variant rows keep old labels
    cache = json.load(open(X4 / "eval_truth_extras.json"))
    xh = {(d["checkpoint_id"], int(d["env_seed"])): d for d in cache}
    base_now = X.load_frozen("eval")
    rerun = {}
    for v in ("A_track8_r2", "C_vidcomp"):
        rows = X.load_variant("eval", v)
        kk = sorted(set(rows) & set(base_now) & set(xh))
        rerun[v] = {"variant_ft_acc": X.metrics([rows[k] for k in kk], xh)["failure_type_acc_on_failures_all"],
                    "frozen_v2_ft_acc": X.metrics([base_now[k] for k in kk], xh)["failure_type_acc_on_failures_all"],
                    "paired_ft": X.paired(rows, base_now, xh)["ft_ok_on_failures"]}
    rerun["frozen_v2_rows_with_label_differing_from_variant_rows"] = sum(
        1 for k, r in X.load_variant("eval", "A_track8").items() if k in base_now
        and base_now[k]["truth_failure_type"] != r["truth_failure_type"])
    res["x4_score_if_rerun_now_mixed_labels"] = rerun

    # --- probe results (if run)
    if PROBE.exists():
        res["probe"] = probe_summary(ev_eps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {OUT}")


# ----------------------------------------------------------------------------- probe (Nebius, <= 30 calls)

def padded_video(ep, n_frames=201):
    """Copy of the success clip padded to the timeout clip length by repeating its final frame."""
    import imageio.v2 as imageio
    out = SCRATCH / "padded" / ep["checkpoint_id"] / f"seed{ep['env_seed']}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fr = [f for f in imageio.get_reader(str(ROOT / ep["video"]))]
    fr = fr + [fr[-1]] * max(0, n_frames - len(fr))
    with imageio.get_writer(str(out), fps=float(ep["video_fps"]), codec="libx264", quality=8,
                            macro_block_size=16, pixelformat="yuv420p") as w:
        for f in fr:
            w.append_data(f)
    return out


def probe_plan(ev_eps):
    """Deterministic plan: 10 eval SUCCESS episodes (7 can, 3 square; first by sorted key over distinct checkpoints)
    run with v2 frozen and C_vidcomp on a padded 20.1 s clip; 10 eval FAILURES re-run with A_track8 unchanged."""
    A_rows = {key(r): r for r in jl(X4 / "eval_A_track8.jsonl")}
    C_rows = {key(r): r for r in jl(X4 / "eval_C_vidcomp.jsonl")}
    V_rows = {key(r): r for r in jl(X.FROZEN_V2["eval"]) if r.get("variant") == "v2_evidence_first"}
    succ = [k for k in sorted(ev_eps) if ev_eps[k]["success"] and k in A_rows and k in C_rows and k in V_rows]
    pick, seen = [], set()
    for task, n in (("can", 7), ("square", 3)):
        for k in succ:
            if ev_eps[k]["task"] == task and k[0] not in seen and sum(ev_eps[p]["task"] == task for p in pick) < n:
                pick.append(k)
                seen.add(k[0])
    fails = [k for k in sorted(ev_eps) if not ev_eps[k]["success"] and k in A_rows]
    rng = np.random.default_rng(12345)
    rep = [fails[i] for i in sorted(rng.choice(len(fails), size=10, replace=False))]
    plan = [("v2_padded", k) for k in pick] + [("C_padded", k) for k in pick] + [("A_repeat", k) for k in rep]
    return plan[:PROBE_BUDGET], A_rows


def cmd_probe(a):
    from cosmos_plv.nebius_backend import NebiusVLMClient
    ev_eps = episodes("eval")
    plan, A_rows = probe_plan(ev_eps)
    done = {(r["probe"], r["checkpoint_id"], r["env_seed"]) for r in jl(PROBE) if not r.get("backend_error")}
    used = sum(1 for r in jl(EVIDENCE) if r.get("tag", "").startswith(f"explore/{NAME}/"))
    todo = [(p, k) for p, k in plan if (p, k[0], k[1]) not in done]
    todo = todo[: max(0, PROBE_BUDGET - used)]
    print(f"probe: {len(todo)} calls to make (used {used} / {PROBE_BUDGET})", flush=True)
    client = NebiusVLMClient(evidence_path=str(EVIDENCE))
    lock = threading.Lock()
    SCRATCH.mkdir(parents=True, exist_ok=True)

    def work(item):
        probe, k = item
        ep = ev_eps[k]
        task = ep["task"]
        tag = f"explore/{NAME}/{probe}/{k[0]}/seed{k[1]}"
        try:
            if probe in ("v2_padded", "C_padded"):
                vid = padded_video(ep)
                dur = 201 / float(ep["video_fps"])
                if probe == "v2_padded":
                    prompt = build_prompt(task, 4.0, dur, False, "v2_evidence_first")
                    res = client.generate(prompt, video=str(vid), max_tokens=700, reasoning=False, tag=tag)
                    p, _ = X.parse_diagnostic_repaired(res.raw_text)
                else:
                    last = X4 / "frames" / k[0] / f"seed{k[1]}" / "last.png"
                    prompt = X.build_vidcomp_prompt(task, dur)
                    res = client.generate(prompt, video=str(vid), images=[str(last)], max_tokens=500,
                                          reasoning=False, tag=tag)
                    p = X.parse_vidcomp(res.raw_text, task)
                track = None
            else:
                r0 = A_rows[k]
                imgs = [str(ROOT / im) for im in r0["input_images"]]
                times = [int(Path(im).stem[2:]) / float(ep["video_fps"]) for im in r0["input_images"]]
                prompt = X.build_track_prompt(task, times, grid=False)
                res = client.generate(prompt, images=imgs, max_tokens=900, reasoning=False, tag=tag)
                p, track = X.parse_track(res.raw_text, task, 8, "image")
                if res.params.get("prompt_sha256") != r0.get("prompt_sha256"):
                    print(f"WARNING prompt hash differs for {k}", flush=True)
            row = {"probe": probe, "checkpoint_id": k[0], "env_seed": k[1], "task": task,
                   "truth_success": ep["success"], "truth_failure_type": ep["failure_type"],
                   "parse_status": p.parse_status, "pred_success": p.fields.get("task_success"),
                   "pred_failure_type": p.fields.get("failure_type"), "final_location": p.fields.get("final_location"),
                   "track": track, "raw_text": res.raw_text, "latency_s": res.latency_s,
                   "prompt_sha256": res.params.get("prompt_sha256"), "backend_error": res.error}
        except Exception as e:  # infrastructure failure
            row = {"probe": probe, "checkpoint_id": k[0], "env_seed": k[1], "backend_error": f"{type(e).__name__}"}
        with lock:
            with open(PROBE, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        print(json.dumps({kk: row.get(kk) for kk in ("probe", "checkpoint_id", "env_seed", "pred_success",
                                                      "pred_failure_type", "final_location", "parse_status",
                                                      "backend_error")}), flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(work, todo))


def probe_summary(ev_eps):
    rows = [r for r in jl(PROBE) if not r.get("backend_error")]
    A_rows = {key(r): r for r in jl(X4 / "eval_A_track8.jsonl")}
    C_rows = {key(r): r for r in jl(X4 / "eval_C_vidcomp.jsonl")}
    V_rows = {key(r): r for r in jl(X.FROZEN_V2["eval"]) if r.get("variant") == "v2_evidence_first"}
    out = {}
    for probe, orig in (("v2_padded", V_rows), ("C_padded", C_rows)):
        rs = [r for r in rows if r["probe"] == probe]
        ok = [r for r in rs if r["parse_status"] in ("ok", "ok_repaired")]
        out[probe] = {
            "n": len(rs), "parsed": len(ok),
            "pred_success_on_padded": sum(bool(r["pred_success"]) for r in ok),
            "pred_success_on_original_clip": sum(bool(orig[key(r)].get("cosmos_task_success")) for r in ok),
            "flips_success_to_failure": sum(bool(orig[key(r)].get("cosmos_task_success")) and not r["pred_success"]
                                            for r in ok),
            "rows": [{"checkpoint_id": r["checkpoint_id"], "env_seed": r["env_seed"], "task": r["task"],
                      "orig_success": orig[key(r)].get("cosmos_task_success"), "padded_success": r["pred_success"],
                      "padded_ft": r["pred_failure_type"], "padded_final_location": r.get("final_location")}
                     for r in rs]}
    rs = [r for r in rows if r["probe"] == "A_repeat"]
    agree_ft, agree_ft_r2, agree_succ, hits = 0, 0, 0, collections.Counter()
    det = []
    for r in rs:
        o = A_rows[key(r)]
        o2 = X.rederive_r2(o)
        new_r2 = my_r2(r["task"], r["track"])[1] if r.get("track") and r["track"][-1].get("valid") else None
        agree_ft += int(r["pred_failure_type"] == o["cosmos_failure_type"])
        agree_ft_r2 += int(new_r2 == o2["cosmos_failure_type"])
        agree_succ += int(bool(r["pred_success"]) == bool(o["cosmos_task_success"]))
        hits["orig_r2_correct"] += int(o2["cosmos_failure_type"] == o["truth_failure_type"])
        hits["repeat_r2_correct"] += int(new_r2 == o["truth_failure_type"])
        det.append({"checkpoint_id": r["checkpoint_id"], "env_seed": r["env_seed"], "truth": o["truth_failure_type"],
                    "orig_r2": o2["cosmos_failure_type"], "repeat_r2": new_r2,
                    "same_prompt_hash": r.get("prompt_sha256") == o.get("prompt_sha256")})
    out["A_repeat"] = {"n": len(rs), "r1_failure_type_agreement": agree_ft, "r2_failure_type_agreement": agree_ft_r2,
                       "success_agreement": agree_succ, **dict(hits), "rows": det}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("rescore", "probe", "plan"))
    a = ap.parse_args()
    if a.cmd == "rescore":
        cmd_rescore(a)
    elif a.cmd == "probe":
        cmd_probe(a)
    else:
        plan, _ = probe_plan(episodes("eval"))
        print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
