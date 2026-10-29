#!/usr/bin/env python3
"""Adversarial verification of x5_cosmos_modes (EXPLORATORY / post-hoc; CPU only, Env B, no model calls).

What it does (nothing outside the verification scratch dir / its results JSON is written):
  1. Re-runs x5_analyze.main_with_frames() and x5_rule_ablation.main() with their OUTPUT PATHS
     redirected to data/outputs/hackathon_explore/x5_cosmos_modes_verification/ (the original
     results files are never touched) and diffs the headline numbers against the stored JSON.
  2. Re-scores every saved raw Cosmos output (calls.jsonl) with the current parsers.
  3. Frame-level 'held' check: recomputes on ALL format-repaired pair outputs (x5 used strictly-parsed
     pairs only), truth-definition sensitivity, video-frame/sim-step offset sensitivity, denominators.
  4. Baseline fairness: the always-no_grasp reference uses oracle knowledge of which episodes failed;
     adds a same-success-predictor reference and macro (per-class) recall on failures.
  5. Nebius v2 failure type vs majority class on the SAME failures.
  6. Clip-length leakage (episodes stop 16 steps after success): length-only success classifier and the
     time gap written into the c_pairwise8 prompt.
  7. Dependence: checkpoint-cluster bootstrap; seed composition; which episodes drive the dev/eval gaps.
  8. Label-artifact claim (square z0) against the CURRENT failure_annotator and refreshed labels.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_cosmos_modes_verify.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x5_common as C  # noqa: E402
import x5_analyze as A  # noqa: E402
import x5_rule_ablation as RA  # noqa: E402
from x5_cosmos_modes import load_v7_frames  # noqa: E402

VSCR = C.ROOT / "data/outputs/hackathon_explore/x5_cosmos_modes_verification"
OUT = C.ROOT / "results/hackathon/explore/x5_cosmos_modes_verification.json"
ORIG_MAIN = C.ROOT / "results/hackathon/explore/x5_cosmos_modes.json"
ORIG_RA = C.ROOT / "results/hackathon/explore/x5_cosmos_modes_rule_ablation.json"
FT_CLASSES = ("no_grasp", "dropped", "not_placed", "other")


def wilson(k, n):
    return C.wilson(k, n)


# ---------------------------------------------------------------- 1. reproduction
def rerun_original():
    VSCR.mkdir(parents=True, exist_ok=True)
    A.OUT = VSCR / "rerun_x5_cosmos_modes.json"
    A.OUT_EP = VSCR / "rerun_x5_cosmos_modes_episodes.jsonl"
    A.main_with_frames()
    RA.OUT = VSCR / "rerun_x5_cosmos_modes_rule_ablation.json"
    RA.main()
    new_main = json.loads(A.OUT.read_text())
    new_ra = json.loads(RA.OUT.read_text())
    old_main = json.loads(ORIG_MAIN.read_text())
    old_ra = json.loads(ORIG_RA.read_text())

    keys = ("episodes", "unparsed", "tp", "tn", "fp", "fn", "success_acc_all", "success_acc_all_ci95",
            "failure_type_acc_on_failures_all", "failure_type_acc_ci95", "mean_latency_s_per_episode")
    diffs = []
    for sp in ("dev", "eval"):
        for v in old_main[sp]:
            o = old_main[sp][v]["own_episodes"]
            n = new_main[sp].get(v, {}).get("own_episodes", {})
            for k in keys:
                if o.get(k) != n.get(k):
                    diffs.append({"split": sp, "variant": v, "key": k, "stored": o.get(k), "rerun": n.get(k)})
            for cmp in ("vs_v6_same_episodes", "vs_v7_same_episodes"):
                if cmp in old_main[sp][v]:
                    for pk, pv in old_main[sp][v][cmp].items():
                        if pk.startswith("paired") and pv != new_main[sp][v].get(cmp, {}).get(pk):
                            diffs.append({"split": sp, "variant": v, "key": f"{cmp}.{pk}", "stored": pv,
                                          "rerun": new_main[sp][v].get(cmp, {}).get(pk)})
    fl_diff = old_main.get("frame_level") != new_main.get("frame_level")
    ra_diff = []
    for sp in ("dev", "eval"):
        for r in RA.RULES:
            for k in ("episodes", "failure_type_acc_on_failures_all", "failure_type_acc_ci95"):
                if old_ra[sp][r].get(k) != new_ra[sp][r].get(k):
                    ra_diff.append({"split": sp, "rule": r, "key": k, "stored": old_ra[sp][r].get(k),
                                    "rerun": new_ra[sp][r].get(k)})
    if old_ra["eval_paired_ftype_diff_choice_minus_R0"] != new_ra["eval_paired_ftype_diff_choice_minus_R0"]:
        ra_diff.append({"key": "eval_paired_ftype_diff_choice_minus_R0",
                        "stored": old_ra["eval_paired_ftype_diff_choice_minus_R0"],
                        "rerun": new_ra["eval_paired_ftype_diff_choice_minus_R0"]})
    return {
        "rerun_outputs": [str(A.OUT.relative_to(C.ROOT)), str(RA.OUT.relative_to(C.ROOT))],
        "main_metric_diffs": diffs, "frame_level_differs": fl_diff,
        "selection_recomputed_same": old_main["selection_recomputed"]["dev_scores"] == new_main["selection_recomputed"]["dev_scores"]
        and old_main["selection_in_run"]["dev_scores"] == new_main["selection_recomputed"]["dev_scores"],
        "selection_choice": {k: new_main["selection_recomputed"][k] for k in ("success_choice", "ftype_choice")},
        "rule_ablation_diffs": ra_diff, "rule_ablation_dev_choice": new_ra["dev_choice"],
        "majority_rerun": new_main["majority_failure_type_baseline"],
    }, new_main, new_ra


# ---------------------------------------------------------------- helpers
def load_calls():
    return [json.loads(line) for line in (C.SCRATCH / "calls.jsonl").read_text().splitlines()]


def truth_all():
    tm = {}
    for root in (C.DEV_ROOT, C.EVAL_ROOT):
        for e in C.iter_episodes(root):
            tm[(e["checkpoint_id"], int(e["env_seed"]))] = e
    return tm


def frame_truth_fn(ep, rest_step=10, contact_required=True, dz_min=0.02, offset_steps=0):
    e = np.load(C.ROOT / ep["episode_npz"])
    z = e["object_pos"][:, 2]
    rest = float(z[min(rest_step, len(z) - 1)])
    T = len(z) - 1

    def at(k):
        s = int(np.clip(2 * k + offset_steps, 0, T))
        dz = float(z[s] - rest)
        held = (bool(e["grasped"][s]) if contact_required else True) and dz >= dz_min
        return {"held": held, "dz": dz, "contact": bool(e["grasped"][s])}
    return at


# ---------------------------------------------------------------- 2. re-score raw outputs
def rescore_raw(calls):
    from cosmos_plv.visual_diagnostics import parse_perception, split_reasoning

    out = {}
    cs = [r for r in calls if r["variant"] == "c_pairwise8"]
    strict = Counter(C.parse_pair(r["raw_text"], r["task"]).parse_status for r in cs)
    cached = Counter(r["parse_status"] for r in cs)
    rep = Counter()
    motion = Counter()
    first_char = Counter()
    for r in cs:
        q = A.parse_pair_repaired(r["raw_text"], r["task"])
        rep[q.parse_status] += 1
        if q.parse_status in C.OK:
            motion[q.fields["object_motion"]] += 1
        if r["parse_status"] not in C.OK:
            _, ans = split_reasoning(r["raw_text"] or "")
            s = ans.strip().lstrip("`").lstrip("json").strip()
            first_char[s[:1] or "<empty>"] += 1
    out["c_pairwise8"] = {"calls": len(cs), "strict_status_cached": dict(cached), "strict_status_reparsed": dict(strict),
                          "strict_unparsed": sum(v for k, v in strict.items() if k not in C.OK),
                          "repaired_status": dict(rep), "repaired_parsed": sum(v for k, v in rep.items() if k in C.OK),
                          "motion_on_repaired": dict(motion),
                          "first_char_of_payload_in_strict_failures": dict(first_char)}
    for v in ("a_roi_track8", "b_track15", "d_reason_last"):
        rows = [r for r in calls if r["variant"] == v]
        mism = 0
        for r in rows:
            p = parse_perception(r["raw_text"], r["task"])
            if p.parse_status != r["parse_status"] or (
                    p.parse_status in C.OK and (p.fields.get("object_location") != (r["fields"] or {}).get("object_location")
                                                or p.fields.get("held_by_gripper") != (r["fields"] or {}).get("held_by_gripper"))):
                mism += 1
        out[v] = {"calls": len(rows), "status": dict(Counter(r["parse_status"] for r in rows)),
                  "reparse_mismatches": mism}
    ds = [r for r in calls if r["variant"] == "d_reason_last"]
    out["d_reason_last"]["completion_tokens_at_cap_1024"] = sum(1 for r in ds if (r["completion_tokens"] or 0) >= 1024)
    return out


# ---------------------------------------------------------------- 3. frame level
def frame_level_checks(calls, v7, tm):
    cache = {(r["variant"], r["checkpoint_id"], int(r["env_seed"]), r["unit"]): r for r in calls}
    eps = {"dev": sorted({(r["checkpoint_id"], int(r["env_seed"])) for r in calls if r["checkpoint_id"].startswith("dev_")}),
           "eval": sorted({(r["checkpoint_id"], int(r["env_seed"])) for r in calls if not r["checkpoint_id"].startswith("dev_")})}
    truth_defs = {
        "x5_contact_and_dz2cm": dict(contact_required=True, dz_min=0.02),
        "contact_only": dict(contact_required=True, dz_min=-1.0),
        "contact_and_dz1cm": dict(contact_required=True, dz_min=0.01),
        "dz2cm_no_contact_needed": dict(contact_required=False, dz_min=0.02),
        "x5_def_offset_minus2steps": dict(contact_required=True, dz_min=0.02, offset_steps=-2),
        "x5_def_offset_plus2steps": dict(contact_required=True, dz_min=0.02, offset_steps=2),
    }
    res = {}
    for sp, keys in eps.items():
        res[sp] = {}
        for tname, kw in truth_defs.items():
            acc = defaultdict(lambda: Counter())
            for key in keys:
                ep = tm.get(key)
                if ep is None:
                    continue
                at = frame_truth_fn(ep, **kw)
                have = v7.get(key, {})
                idx8 = C.keyframe_indices(ep["video_frames"], 8)
                for j in idx8:
                    tr = at(j)
                    if j in have and have[j]["parse_status"] in C.OK:
                        f = have[j]
                        ph = bool(f.get("p_held_by_gripper")) or f.get("p_object_location") == "in_gripper"
                        acc["v7_full"][("tp" if ph and tr["held"] else "fp" if ph else "fn" if tr["held"] else "tn")] += 1
                    r = cache.get(("a_roi_track8", key[0], key[1], f"f{j:03d}"))
                    if r and r["parse_status"] in C.OK:
                        f = r["fields"]
                        ph = bool(f.get("held_by_gripper")) or f.get("object_location") == "in_gripper"
                        acc["a_crop"][("tp" if ph and tr["held"] else "fp" if ph else "fn" if tr["held"] else "tn")] += 1
                for j in C.keyframe_indices(ep["video_frames"], 15):
                    r = cache.get(("b_track15", key[0], key[1], f"f{j:03d}"))
                    if r and r["parse_status"] in C.OK:
                        f = r["fields"]
                        tr = at(j)
                        ph = bool(f.get("held_by_gripper")) or f.get("object_location") == "in_gripper"
                        acc["b_midpoints"][("tp" if ph and tr["held"] else "fp" if ph else "fn" if tr["held"] else "tn")] += 1
                for i in range(len(idx8) - 1):
                    r = cache.get(("c_pairwise8", key[0], key[1], f"p{idx8[i]:03d}_{idx8[i + 1]:03d}"))
                    if not r:
                        continue
                    for parser_name, parsed in (("strict", C.parse_pair(r["raw_text"], r["task"])),
                                                ("repaired", A.parse_pair_repaired(r["raw_text"], r["task"]))):
                        if parsed.parse_status not in C.OK:
                            continue
                        f = parsed.fields
                        t0, t1 = at(idx8[i]), at(idx8[i + 1])
                        ph = bool(f["held_in_image_2"]) or f["location_in_image_2"] == "in_gripper"
                        acc[f"c_image2_{parser_name}"][("tp" if ph and t1["held"] else "fp" if ph else "fn" if t1["held"] else "tn")] += 1
                        true_up = (t1["dz"] - t0["dz"]) >= 0.03
                        pu = f["object_motion"] == "up"
                        acc[f"c_motion_up_{parser_name}"][("tp" if pu and true_up else "fp" if pu else "fn" if true_up else "tn")] += 1
            out = {}
            for src, c in acc.items():
                n = sum(c.values())
                npred, npos, nneg = c["tp"] + c["fp"], c["tp"] + c["fn"], c["fp"] + c["tn"]
                out[src] = {"n": n, **dict(c),
                            "precision": round(c["tp"] / npred, 3) if npred else None,
                            "precision_ci95": wilson(c["tp"], npred),
                            "recall": round(c["tp"] / npos, 3) if npos else None,
                            "fp_rate_on_negatives": round(c["fp"] / nneg, 3) if nneg else None,
                            "fp_share_of_all_frames": round(c["fp"] / n, 3) if n else None,
                            "positive_prevalence": round(npos / n, 3) if n else None}
            res[sp][tname] = out
    return res


def v7_all_frames_false_held(v7, tm):
    """Attempt to reproduce the report's 'earlier CPU-only check' (47.1% dev / 48.4% eval false held)."""
    out = {}
    for sp in ("dev", "eval"):
        c = Counter()
        n_eps = 0
        for key, frs in v7.items():
            if key[0].startswith("dev_") != (sp == "dev") or key not in tm:
                continue
            n_eps += 1
            at = frame_truth_fn(tm[key])
            for j, f in frs.items():
                if f["parse_status"] not in C.OK:
                    continue
                ph = bool(f.get("p_held_by_gripper")) or f.get("p_object_location") == "in_gripper"
                tr = at(j)
                c["tp" if ph and tr["held"] else "fp" if ph else "fn" if tr["held"] else "tn"] += 1
        nneg = c["fp"] + c["tn"]
        out[sp] = {"episodes": n_eps, "frames": sum(c.values()), **dict(c),
                   "false_held_rate_on_not_held": round(c["fp"] / nneg, 3) if nneg else None}
    return out


# ---------------------------------------------------------------- 4. baseline fairness
def ft_views(rows, label):
    """rows: list with truth_success, truth_failure_type, pred.  Accuracy on true failures plus:
    - oracle_majority: always no_grasp on every TRUE failure (x5's reference; knows which episodes failed)
    - same_detector_majority: no_grasp whenever THIS variant predicts failure, 'none' when it predicts success
    - macro_recall over failure classes present
    """
    fails = [r for r in rows if not r["truth_success"]]
    n = len(fails)
    if n == 0:
        return {"label": label, "n_failures": 0}

    def pred_ft(r):
        p = r["pred"]
        return p.get("failure_type") if p and p.get("parse_status") == "ok" else "unscored"

    def pred_fail(r):
        p = r["pred"]
        return bool(p) and p.get("parse_status") == "ok" and not p.get("task_success")

    k = sum(pred_ft(r) == r["truth_failure_type"] for r in fails)
    k_orc = sum(r["truth_failure_type"] == "no_grasp" for r in fails)
    k_same = sum(pred_fail(r) and r["truth_failure_type"] == "no_grasp" for r in fails)
    classes = sorted({r["truth_failure_type"] for r in fails})
    per_cls = {}
    for c in classes:
        rc = [r for r in fails if r["truth_failure_type"] == c]
        per_cls[c] = {"n": len(rc), "variant_recall": round(sum(pred_ft(r) == c for r in rc) / len(rc), 3),
                      "always_no_grasp_recall": 1.0 if c == "no_grasp" else 0.0}
    macro_v = round(sum(d["variant_recall"] for d in per_cls.values()) / len(per_cls), 3)
    macro_m = round(sum(d["always_no_grasp_recall"] for d in per_cls.values()) / len(per_cls), 3)
    return {"label": label, "n_failures": n, "variant_acc": round(k / n, 3), "variant_ci95": wilson(k, n),
            "oracle_always_no_grasp_acc": round(k_orc / n, 3), "oracle_ci95": wilson(k_orc, n),
            "same_detector_no_grasp_acc": round(k_same / n, 3), "same_detector_ci95": wilson(k_same, n),
            "per_class": per_cls, "macro_recall_variant": macro_v, "macro_recall_always_no_grasp": macro_m,
            "pred_distribution_on_failures": dict(Counter(pred_ft(r) for r in fails))}


def baseline_fairness(new_main_rows_path, ra_rows):
    rows = defaultdict(list)
    for line in Path(new_main_rows_path).read_text().splitlines():
        r = json.loads(line)
        rows[(r["split"], r["variant"])].append(r)
    out = {"x5_variants": {}, "rule_ablation": {}}
    for (sp, v), rs in sorted(rows.items()):
        if v in ("d_reason_last", "v6_baseline"):
            continue
        if not any(r["pred"] and r["pred"].get("failure_type") for r in rs):
            continue
        out["x5_variants"][f"{sp}/{v}"] = ft_views(rs, f"{sp}/{v}")
    for sp, d in ra_rows.items():
        for rule, rs in d.items():
            out["rule_ablation"][f"{sp}/{rule}"] = ft_views(rs, f"{sp}/{rule}")
    return out


def ra_rows_now():
    v7 = load_v7_frames()
    return {"dev": RA.rows_for(C.DEV_ROOT, v7, True), "eval": RA.rows_for(C.EVAL_ROOT, v7, False)}


# ---------------------------------------------------------------- 5. Nebius majority check
def nebius_majority(tm):
    out = {}
    for fname in ("eval_nebius_v2.jsonl", "eval_nebius_v2.orig.jsonl", "eval_nebius_v2_pad201.jsonl"):
        p = C.COSMOS_DIR / fname
        if not p.exists():
            continue
        rows = []
        keys_seen = set()
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("backend") not in (None, "nebius") and "nebius" not in str(r.get("backend")):
                continue
            key = (r["checkpoint_id"], int(r["env_seed"]))
            if key not in tm or key in keys_seen:
                continue
            keys_seen.add(key)
            e = tm[key]
            pred = ({"parse_status": "ok", "task_success": r["cosmos_task_success"], "failure_type": r["cosmos_failure_type"]}
                    if r.get("scored") else {"parse_status": "unscored"})
            rows.append({"truth_success": e["success"], "truth_failure_type": e["failure_type"], "pred": pred,
                         "variant": r.get("variant")})
        out[fname] = {"rows": len(rows), "variants": dict(Counter(r["variant"] for r in rows)),
                      **ft_views(rows, fname)}
    return out


# ---------------------------------------------------------------- 6. clip-length leakage
def clip_length(calls, tm):
    out = {}
    for sp, root in (("dev", C.DEV_ROOT), ("eval", C.EVAL_ROOT)):
        eps = list(C.iter_episodes(root))
        maxf = max(e["video_frames"] for e in eps)
        pred = [e["video_frames"] < maxf for e in eps]
        acc = sum(p == bool(e["success"]) for p, e in zip(pred, eps)) / len(eps)
        out[sp] = {"episodes": len(eps), "max_video_frames": maxf,
                   "length_only_success_acc": round(acc, 3),
                   "failures_all_have_max_frames": all(e["video_frames"] == maxf for e in eps if not e["success"]),
                   "successes_with_max_frames": sum(1 for e in eps if e["success"] and e["video_frames"] == maxf)}
    dts = defaultdict(list)
    for r in calls:
        if r["variant"] != "c_pairwise8":
            continue
        e = tm[(r["checkpoint_id"], int(r["env_seed"]))]
        dt = float(r["prompt"].split("taken ")[1].split(" seconds")[0])
        dts["success" if e["success"] else "failure"].append(dt)
    out["c_pair_prompt_dt_s"] = {k: {"n": len(v), "min": min(v), "max": max(v), "distinct": sorted(set(v))[:12]}
                                 for k, v in dts.items()}
    s_max = max(dts["success"]) if dts["success"] else None
    f_min = min(dts["failure"]) if dts["failure"] else None
    out["c_pair_prompt_dt_separates_outcome"] = (s_max is not None and f_min is not None and s_max < f_min)
    return out


# ---------------------------------------------------------------- 7. dependence
def paired_vectors(rows_a, rows_b):
    ka = {(r["checkpoint_id"], r["env_seed"]): r for r in rows_a}
    kb = {(r["checkpoint_id"], r["env_seed"]): r for r in rows_b}
    keys = sorted(set(ka) & set(kb))
    va = C.correct_vec([ka[k] for k in keys])
    vb = C.correct_vec([kb[k] for k in keys])
    return keys, [x - y for x, y in zip(va, vb)]


def cluster_boot(keys, d, n_boot=4000, seed=1):
    by = defaultdict(list)
    for k, x in zip(keys, d):
        by[k[0]].append(x)
    cl = list(by.values())
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        s = [cl[rng.randrange(len(cl))] for _ in range(len(cl))]
        flat = [x for c in s for x in c]
        bs.append(sum(flat) / len(flat))
    bs.sort()
    return {"n_clusters": len(cl), "ci95": (round(bs[int(0.025 * n_boot)], 3), round(bs[int(0.975 * n_boot) - 1], 3))}


def dependence(new_main_rows_path):
    rows = defaultdict(list)
    for line in Path(new_main_rows_path).read_text().splitlines():
        r = json.loads(line)
        rows[(r["split"], r["variant"])].append(r)
    out = {}
    for sp in ("dev", "eval"):
        for v in ("a_roi_track8", "c_pairwise8_fmtrepair", "d_reason_last"):
            if not rows.get((sp, v)):
                continue
            keys, d = paired_vectors(rows[(sp, v)], rows[(sp, "v6_baseline")])
            disc = [{"episode": list(k), "variant_minus_v6": x} for k, x in zip(keys, d) if x != 0]
            out[f"{sp}/{v}_minus_v6"] = {"n": len(keys), "diff": round(sum(d) / len(d), 3),
                                         "checkpoint_cluster_bootstrap": cluster_boot(keys, d),
                                         "discordant": disc}
        seeds = Counter(r["env_seed"] for r in rows[(sp, "a_roi_track8")])
        cks = Counter(r["checkpoint_id"] for r in rows[(sp, "a_roi_track8")])
        out[f"{sp}_composition"] = {"seeds": dict(seeds), "checkpoints": dict(cks),
                                    "outcomes_by_checkpoint": {ck: dict(Counter(
                                        "S" if r["truth_success"] else r["truth_failure_type"]
                                        for r in rows[(sp, "a_roi_track8")] if r["checkpoint_id"] == ck)) for ck in cks}}
    return out


def eval_a_breakers(calls, tm):
    """For eval episodes where a_roi_track8 is wrong but v6 right: last-frame answers of both."""
    v6 = C.load_baseline(C.COSMOS_DIR / "eval_local_v6.jsonl", "v6_perception_last")
    out = []
    for r in calls:
        if r["variant"] != "a_roi_track8" or r["checkpoint_id"].startswith("dev_"):
            continue
        key = (r["checkpoint_id"], int(r["env_seed"]))
        e = tm[key]
        if r["unit"] != f"f{e['video_frames'] - 1:03d}":
            continue
        a_succ = (r["fields"] or {}).get("object_location") == C.SUCCESS_LOCATION[e["task"]]
        b = v6.get(key)
        v6_succ = b["pred"].get("task_success") if b else None
        if a_succ != e["success"] or (v6_succ is not None and v6_succ != e["success"]):
            out.append({"episode": list(key), "truth_success": e["success"], "truth_ft": e["failure_type"],
                        "a_last_location": (r["fields"] or {}).get("object_location"),
                        "a_last_explanation": (r["fields"] or {}).get("explanation"),
                        "a_image": r["images"][0],
                        "v6_success": v6_succ,
                        "v6_raw": (b["_row"].get("raw_text") or "")[:300] if b else None})
    return out


# ---------------------------------------------------------------- 8. square z0 claim
def square_z0(tm):
    from cosmos_plv.failure_annotator import SETTLE_STEPS, annotate_episode

    drops, changed, n = [], Counter(), 0
    for key, e in tm.items():
        if e["task"] != "square":
            continue
        d = np.load(C.ROOT / e["episode_npz"])
        z = d["object_pos"][:, 2]
        drops.append(float(z[0] - z[: SETTLE_STEPS + 1].min()))
        ann = annotate_episode({k: d[k] for k in ("eef_pos", "object_pos", "grasped", "success")})
        n += 1
        if ann["failure_type"] != e["failure_type"]:
            changed[(e["failure_type"], ann["failure_type"])] += 1
    orig_vs_now = Counter()
    for root in (C.DEV_ROOT, C.EVAL_ROOT):
        for summ in root.glob("*/summary.orig.json"):
            now = {(x["checkpoint_id"], x["env_seed"]): x for x in json.loads(summ.with_name("summary.json").read_text())["episodes"]}
            for x in json.loads(summ.read_text())["episodes"]:
                y = now.get((x["checkpoint_id"], x["env_seed"]))
                if y and x["failure_type"] != y["failure_type"]:
                    orig_vs_now[(x["task"], x["failure_type"], y["failure_type"])] += 1
    labels_now = Counter((e["task"], e["failure_type"]) for e in tm.values())
    return {"current_annotator_z0": "min(object z over steps 0..SETTLE_STEPS) with SETTLE_STEPS=%d" % SETTLE_STEPS,
            "square_episodes": n,
            "square_spawn_drop_m": {"median": round(float(np.median(drops)), 4), "min": round(float(np.min(drops)), 4),
                                    "max": round(float(np.max(drops)), 4)},
            "current_annotator_vs_summary_label_mismatches": {f"{a}->{b}": c for (a, b), c in changed.items()},
            "summary_orig_vs_now_changes": {f"{t}:{a}->{b}": c for (t, a, b), c in orig_vs_now.items()},
            "labels_now": {f"{t}:{f}": c for (t, f), c in sorted(labels_now.items())}}


def main():
    res = {"name": "x5_cosmos_modes_verification",
           "label": "EXPLORATORY / post-hoc adversarial verification; CPU only; no new model or Nebius calls"}
    rep, new_main, new_ra = rerun_original()
    res["reproduction"] = rep
    calls = load_calls()
    tm = truth_all()
    v7 = load_v7_frames()
    res["raw_rescore"] = rescore_raw(calls)
    res["frame_level"] = frame_level_checks(calls, v7, tm)
    res["v7_all_frames_false_held"] = v7_all_frames_false_held(v7, tm)
    ep_path = VSCR / "rerun_x5_cosmos_modes_episodes.jsonl"
    res["baseline_fairness"] = baseline_fairness(ep_path, ra_rows_now())
    res["nebius_majority"] = nebius_majority(tm)
    res["clip_length"] = clip_length(calls, tm)
    res["dependence"] = dependence(ep_path)
    res["eval_a_vs_v6_errors"] = eval_a_breakers(calls, tm)
    res["square_z0"] = square_z0(tm)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
