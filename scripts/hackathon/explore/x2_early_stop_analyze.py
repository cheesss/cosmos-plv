#!/usr/bin/env python3
"""x2_early_stop (EXPLORATORY, post-hoc): can a rollout be aborted once failure is evident?

Re-runnable on whatever rollouts / VLM rows exist when it is run.

Part 1  privileged-state upper bound from eval npz traces (object height, finger
        contact, success flag) + one deployable proprio-only rule (gripper close
        command) + the trivial time-only baseline (= shorter horizon).
Part 2  Cosmos3-Edge per-keyframe perception (eval_local_v7.jsonl, rows still
        arriving) turned into the same kind of rule, vision only.
Part 3  Nemotron-3.5-Super-VL "will this attempt succeed?" on absolute-time video
        prefixes (calls made by x2_early_stop_nebius.py).

Online semantics.  A rule may abort an episode at step a only if the episode is
still running at a (a < last step and success not yet reached at a) and uses only
information up to a.  Aborting a failure saves (last_step - a) control steps and
leaves the success estimate unchanged; aborting a success ("false abort") turns
it into a failure and biases the checkpoint's success estimate downward.

Held-out discipline.  Rule parameters are never fit on the episodes they are
scored on: (a) dev split (seeds 300000+) -> eval split, and (b) 2-fold cross-fit
on eval by env seed (200000-200004 <-> 200005-200009).  Curves vs t are
descriptive (no selection).  Nemotron prompt is chosen on dev only.

Height reference: object z at the settled steps 5..10 (the square nut spawns
~6 cm above the table and falls in the first 5 steps, so step 0 is not the rest
height).

Env B:
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x2_early_stop_analyze.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/hackathon"))

from cosmos_plv.ranking import spearman  # noqa: E402

NAME = "x2_early_stop"
SIM_HZ = 20.0
LIFT_M = 0.05
LOW_M = 0.02
SETTLE = 10  # ignore the first steps while the object settles (square nut spawns ~6 cm high)
SPLIT_ROOT = {"dev": ROOT / "data/outputs/hackathon_rollouts_dev", "eval": ROOT / "data/outputs/hackathon_rollouts_eval"}
V7 = {"eval": ROOT / "data/outputs/hackathon_cosmos/eval_local_v7.jsonl",
      "dev": ROOT / "data/outputs/hackathon_cosmos/dev_prompt_selection.jsonl"}
NEB = ROOT / "data/outputs/hackathon_explore" / NAME / "nebius_prefix_calls.jsonl"
SUCCESS_LOC = {"can": "right_bin", "square": "on_peg"}


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def ratio(k, n):
    return round(k / n, 4) if n else None


def auroc(score, y):
    """P(score_pos > score_neg) + 0.5 ties; positives = y==1."""
    s = np.asarray(score, float)
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def boot(fn, n, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(rng.integers(0, n, n))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return [None, None]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# ---------------------------------------------------------------- data
def load_split(split):
    eps = []
    for summ in sorted(SPLIT_ROOT[split].glob("*/summary.json")):
        for ep in json.loads(summ.read_text())["episodes"]:
            p = ROOT / ep["episode_npz"]
            if not p.exists():
                continue
            z = np.load(p)
            oz = z["object_pos"][:, 2].astype(float)
            zref = float(np.median(oz[5:11])) if len(oz) > 11 else float(oz[0])
            ck = ep["checkpoint_id"].replace("dev_", "")
            run, epoch = ck.rsplit("_e", 1)
            succ = z["success"].astype(bool)
            eps.append({
                "split": split, "checkpoint_id": ep["checkpoint_id"], "run": run, "epoch": int(epoch),
                "task": ep["task"], "env_seed": int(ep["env_seed"]), "success": bool(ep["success"]),
                "failure_type": ep["failure_type"], "end": int(len(oz) - 1),
                "annot_lifted": bool(ep.get("lifted")), "z0_step0": float(oz[0]), "z_settled": zref,
                "z_final": float(oz[-1]),
                "fss": int(np.argmax(succ)) if succ.any() else -1,
                "h": oz - zref, "grasped": z["grasped"].astype(bool),
                "grip_cmd": z["action"][:, 6].astype(float),
            })
    return eps


def running(ep, a):
    return a < ep["end"] and not (ep["fss"] >= 0 and ep["fss"] <= a)


# ---------------------------------------------------------------- rules (return abort step or None)
def r_time(T):
    def f(ep):
        return T if running(ep, T) else None
    f.desc = f"time_only(T={T})"
    return f


def r_nolift(T):
    def f(ep):
        return T if running(ep, T) and ep["h"][SETTLE: T + 1].max(initial=-1.0) < LIFT_M else None
    f.desc = f"nolift(T={T})"
    return f


def r_noclose(T):
    def f(ep):
        return T if running(ep, T) and not (ep["grip_cmd"][:T] > 0).any() else None
    f.desc = f"noclose(T={T})"
    return f


def r_drop(k):
    def f(ep):
        cache = ep.setdefault("_drop_cache", {})
        if k in cache:
            return cache[k]
        h, g = ep["h"], ep["grasped"]
        lifted = np.maximum.accumulate((h >= LIFT_M) & (np.arange(len(h)) >= SETTLE))
        low = (h < LOW_M) & ~g
        run_len = 0
        hit = None
        for t in range(1, ep["end"]):
            run_len = run_len + 1 if (low[t] and lifted[t - 1]) else 0
            if run_len >= k and running(ep, t):
                hit = t
                break
        cache[k] = hit
        return hit
    f.desc = f"drop(k={k})"
    return f


def combo(*rules):
    def f(ep):
        steps = [s for s in (r(ep) for r in rules) if s is not None]
        return min(steps) if steps else None
    f.desc = " + ".join(r.desc for r in rules)
    return f


# ---------------------------------------------------------------- evaluation of an abort rule
def evaluate(eps, aborts, with_ranking=True):
    """aborts: list (aligned with eps) of abort step or None."""
    n = len(eps)
    fail = [not e["success"] for e in eps]
    ab = [a is not None for a in aborts]
    tp = sum(1 for f, x in zip(fail, ab) if f and x)
    fp = sum(1 for f, x in zip(fail, ab) if (not f) and x)
    n_fail = sum(fail)
    n_succ = n - n_fail
    steps = np.array([e["end"] for e in eps], float)
    saved = np.array([(e["end"] - a) if a is not None else 0 for e, a in zip(eps, aborts)], float)
    abort_steps = [a for a in aborts if a is not None]
    out = {
        "n_episodes": n, "n_failures": n_fail, "n_successes": n_succ, "n_aborted": tp + fp,
        "true_aborts": tp, "false_aborts": fp,
        "precision": ratio(tp, tp + fp), "precision_ci95": wilson(tp, tp + fp),
        "recall": ratio(tp, n_fail), "recall_ci95": wilson(tp, n_fail),
        "false_abort_rate": ratio(fp, n_succ), "false_abort_rate_ci95": wilson(fp, n_succ),
        "steps_total": int(steps.sum()), "steps_saved": int(saved.sum()),
        "saved_frac": round(float(saved.sum() / steps.sum()), 4) if steps.sum() else None,
        "saved_frac_ci95": boot(lambda ix: float(saved[ix].sum() / steps[ix].sum()), n),
        "robot_time_saved_s": round(float(saved.sum()) / SIM_HZ, 1),
        "robot_time_total_s": round(float(steps.sum()) / SIM_HZ, 1),
        "median_abort_step": float(np.median(abort_steps)) if abort_steps else None,
    }
    if with_ranking:
        out["ranking"] = ranking_effect(eps, aborts)
    return out


def ranking_effect(eps, aborts):
    by = defaultdict(lambda: [0, 0, 0])  # n, full successes, early-stopped successes
    for e, a in zip(eps, aborts):
        b = by[e["checkpoint_id"]]
        b[0] += 1
        b[1] += int(e["success"])
        b[2] += int(e["success"] and a is None)
    ck = sorted(by)
    full = {c: by[c][1] / by[c][0] for c in ck}
    est = {c: by[c][2] / by[c][0] for c in ck}
    deltas = [est[c] - full[c] for c in ck]
    runs = defaultdict(list)
    for c in ck:
        runs[c.replace("dev_", "").rsplit("_e", 1)[0]].append(c)
    per_run = {}
    for r, cs in sorted(runs.items()):
        f = [full[c] for c in cs]
        s = [est[c] for c in cs]
        best_full = max(f)
        top_est = max(s)
        # pessimistic tie-break: among checkpoints tied at the top of the estimate, take the worst true one
        picked = min(full[c] for c in cs if est[c] == top_est)
        per_run[r] = {"n_checkpoints": len(cs),
                      "spearman_est_vs_full": None if not np.isfinite(spearman(s, f)) else round(spearman(s, f), 3),
                      "regret_top1_pessimistic": round(best_full - picked, 3)}
    return {"n_checkpoints": len(ck), "checkpoints_changed": int(sum(1 for d in deltas if d != 0)),
            "max_abs_delta_success": round(max((abs(d) for d in deltas), default=0.0), 3),
            "mean_abs_delta_success": round(float(np.mean(np.abs(deltas))) if deltas else 0.0, 4),
            "per_checkpoint": {c: {"full": round(full[c], 3), "early_stop": round(est[c], 3)} for c in ck},
            "per_run": per_run}


def apply(rule, eps):
    return [rule(e) for e in eps]


# ---------------------------------------------------------------- Part 1
GRID_T1 = list(range(40, 201, 10))
GRID_K = [5, 10, 20, 40]
GRID_T2 = list(range(160, 401, 20))  # 400 = never fires (no time cut)


def _never(ep):
    return None


_never.desc = "off"


def configs():
    """Every component can be switched off (None); T2=400 means no time cut."""
    for T1, k, T2 in itertools.product(GRID_T1 + [None], GRID_K + [None], GRID_T2):
        rules = [r_nolift(T1) if T1 else _never, r_drop(k) if k else _never, r_time(T2)]
        yield ("state_combo", (T1, k, T2), combo(*rules))
    for T1, T2 in itertools.product(GRID_T1 + [None], GRID_T2):
        yield ("proprio_combo", (T1, T2), combo(r_noclose(T1) if T1 else _never, r_time(T2)))
    for T2 in GRID_T2:
        yield ("time_only", (T2,), r_time(T2))


def select(eps, family, max_fp=0):
    """Max saved steps subject to <= max_fp false aborts on the selection set.

    Ties -> most conservative (components off / largest thresholds)."""
    best = None
    for fam, params, rule in configs():
        if fam != family:
            continue
        ab = apply(rule, eps)
        fp = sum(1 for e, a in zip(eps, ab) if e["success"] and a is not None)
        saved = sum((e["end"] - a) for e, a in zip(eps, ab) if a is not None)
        cons = tuple(10**6 if p is None else p for p in params)
        key = (fp <= max_fp, saved, cons)
        if best is None or key > best[0]:
            best = (key, params, rule)
    return best[1], best[2]


def part1(dev, ev):
    res = {}
    # descriptive curves on eval (no selection)
    curves = defaultdict(list)
    for task in ("can", "square"):
        E = [e for e in ev if e["task"] == task]
        for T in range(20, 401, 20):
            for nm, rule in (("time_only", r_time(T)), ("nolift", r_nolift(T)), ("noclose", r_noclose(T)),
                             ("nolift+drop10", combo(r_nolift(T), r_drop(10)))):
                m = evaluate(E, apply(rule, E), with_ranking=False)
                curves[f"{task}/{nm}"].append({"T_step": T, "T_s": T / SIM_HZ, **{k: m[k] for k in (
                    "n_aborted", "true_aborts", "false_aborts", "precision", "precision_ci95", "recall",
                    "recall_ci95", "saved_frac")}})
    res["curves_eval"] = curves
    # oracle ceiling: abort every failure at step 0
    res["oracle_abort_all_failures_at_0"] = {
        t: evaluate([e for e in ev if e["task"] == t] if t != "all" else ev,
                    [0 if not e["success"] else None for e in ev if (t == "all" or e["task"] == t)],
                    with_ranking=False)["saved_frac"] for t in ("can", "square", "all")}
    # (a) dev -> eval, per task
    sel = {}
    for fam, max_fp in itertools.product(("state_combo", "proprio_combo", "time_only"), (0, 1)):
        aborts = [None] * len(ev)
        chosen = {}
        for task in ("can", "square"):
            D = [e for e in dev if e["task"] == task]
            params, rule = select(D, fam, max_fp)
            chosen[task] = {"params": params, "rule": rule.desc,
                            "dev": evaluate(D, apply(rule, D), with_ranking=False)}
            for i, e in enumerate(ev):
                if e["task"] == task:
                    aborts[i] = rule(e)
        sel[f"{fam}/maxfp{max_fp}"] = {"chosen_on_dev": chosen, "eval_all": evaluate(ev, aborts),
                    "eval_by_task": {t: evaluate([e for e in ev if e["task"] == t],
                                                 [a for e, a in zip(ev, aborts) if e["task"] == t], with_ranking=False)
                                     for t in ("can", "square")},
                    "eval_excluding_epoch0": evaluate([e for e in ev if e["epoch"] > 0],
                                                      [a for e, a in zip(ev, aborts) if e["epoch"] > 0],
                                                      with_ranking=False)}
    res["dev_to_eval"] = sel
    # (b) 2-fold cross-fit on eval by env seed (per task)
    folds = (set(range(200000, 200005)), set(range(200005, 200010)))
    cf = {}
    for fam, max_fp in itertools.product(("state_combo", "proprio_combo", "time_only"), (0, 1)):
        aborts = [None] * len(ev)
        chosen = []
        for task in ("can", "square"):
            for fi in (0, 1):
                fit = [e for e in ev if e["task"] == task and e["env_seed"] in folds[fi]]
                params, rule = select(fit, fam, max_fp)
                chosen.append({"task": task, "fit_fold": fi, "params": params, "rule": rule.desc})
                for i, e in enumerate(ev):
                    if e["task"] == task and e["env_seed"] in folds[1 - fi]:
                        aborts[i] = rule(e)
        cf[f"{fam}/maxfp{max_fp}"] = {"chosen": chosen, "eval_all_crossfit": evaluate(ev, aborts),
                                      "eval_by_task": {t: evaluate([e for e in ev if e["task"] == t],
                                                                   [a for e, a in zip(ev, aborts) if e["task"] == t],
                                                                   with_ranking=False) for t in ("can", "square")}}
    res["crossfit_eval_by_seed"] = cf
    return res


# ---------------------------------------------------------------- Part 2: Cosmos3-Edge v7 keyframe perceptions
def load_v7(split, eps):
    idx = {(e["checkpoint_id"], e["env_seed"]): e for e in eps}
    rows = []
    p = V7[split]
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("variant") != "v7_perception_track" or not r.get("track"):
            continue
        e = idx.get((r["checkpoint_id"], int(r["env_seed"])))
        if e is None:
            continue
        frames = []
        for t, f in zip(r["track"]["times_s"], r["track"]["frames"]):
            if f.get("parse_status") not in ("ok", "ok_repaired"):
                continue
            loc = f.get("p_object_location")
            frames.append({"step": int(round(t * SIM_HZ)), "loc": loc,
                           "held": bool(f.get("p_held_by_gripper")) or loc == "in_gripper"})
        rows.append((e, frames))
    return rows


def v_nohold(T):
    def f(e, frames):
        seen = [fr for fr in frames if fr["step"] <= T]
        return T if running(e, T) and not any(fr["held"] for fr in seen) else None
    f.desc = f"vis_nohold(T={T})"
    return f


def v_drop():
    def f(e, frames):
        was_held = False
        for fr in frames:
            if was_held and not fr["held"] and fr["loc"] != SUCCESS_LOC[e["task"]] and running(e, fr["step"]):
                return fr["step"]
            was_held = was_held or fr["held"]
        return None
    f.desc = "vis_drop"
    return f


def v_combo(*rules):
    def f(e, frames):
        s = [x for x in (r(e, frames) for r in rules) if x is not None]
        return min(s) if s else None
    f.desc = " + ".join(r.desc for r in rules)
    return f


def v_time(T):
    def f(e, frames):
        return T if running(e, T) else None
    f.desc = f"time_only(T={T})"
    return f


def s_nolift_at_frames(T):
    """Privileged state rule, sampled only at the SAME keyframe steps the VLM saw."""
    def f(e, frames):
        seen = [fr["step"] for fr in frames if fr["step"] <= T]
        return T if running(e, T) and not any(e["h"][s] >= LIFT_M for s in seen if s >= SETTLE) else None
    f.desc = f"state_nolift_at_keyframes(T={T})"
    return f


START_LOC = {"can": "left_bin", "square": "on_table"}
# keyframe steps of a full-length (401-step) episode: linspace(0, 200, 8) video frames * 2
FAIL_KF_STEPS = [int(2 * round(x)) for x in np.linspace(0, 200, 8)][1:-1]


def latest_frame(frames, T):
    seen = [fr for fr in frames if fr["step"] <= T]
    return seen[-1] if seen else None


def v_snapshot(T, use_held):
    """Time-matched snapshot rule: look only at the most recent keyframe <= T.

    Abort if the object is still at its start location (and, if use_held, not flagged as held)."""
    def f(e, frames):
        fr = latest_frame(frames, T)
        if fr is None or not running(e, T):
            return None
        bad = fr["loc"] == START_LOC[e["task"]] and (not fr["held"] if use_held else True)
        return T if bad else None
    f.desc = f"vis_snapshot(T={T},use_held={use_held})"
    return f


def s_snapshot(T):
    """Privileged twin of v_snapshot: object height at the SAME keyframe < 5 cm."""
    def f(e, frames):
        fr = latest_frame(frames, T)
        if fr is None or not running(e, T):
            return None
        s = fr["step"]
        return T if (s < SETTLE or e["h"][s] < LIFT_M) else None
    f.desc = f"state_snapshot_same_keyframe(T={T})"
    return f


def _v_off(e, frames):
    return None


_v_off.desc = "off"


def _sel_v(D, make, grid):
    best = None
    for params in grid:
        rule = make(*params)
        ab = [rule(e, fr) for e, fr in D]
        fp = sum(1 for (e, _), a in zip(D, ab) if e["success"] and a is not None)
        saved = sum(e["end"] - a for (e, _), a in zip(D, ab) if a is not None)
        cons = tuple(10**6 if x is None else (int(x) if not isinstance(x, bool) else int(x)) for x in params)
        key = (fp == 0, saved, cons)
        if best is None or key > best[0]:
            best = (key, params, rule)
    return best[1], best[2]


def part2(dev, ev):
    out = {"note": ("v7 keyframes are 8 frames evenly spaced over the WHOLE episode, so their timing depends on the "
                    "(future) episode length: successes are sampled ~3x denser than failures. History rules "
                    "(vis_nohold, vis_drop) are therefore optimistically biased; the snapshot rule uses only the "
                    "single latest keyframe <= T at failure-keyframe times, which is approximately time-matched.")}
    rows_ev = load_v7("eval", ev)
    rows_dev = load_v7("dev", dev)
    out["n_eval_rows"] = len(rows_ev)
    out["n_dev_rows"] = len(rows_dev)
    out["eval_checkpoints"] = dict(Counter(e["checkpoint_id"] for e, _ in rows_ev))
    if not rows_ev:
        return out
    E = [e for e, _ in rows_ev]
    # per-frame perception vs privileged state
    agree, loc_agree = Counter(), Counter()
    for e, frames in rows_ev:
        for fr in frames:
            s = min(fr["step"], e["end"])
            truth = bool(s >= SETTLE and e["h"][s] >= LIFT_M and e["grasped"][s])
            agree[(truth, fr["held"])] += 1
            at_start = bool(s < SETTLE or e["h"][s] < LIFT_M) and not (e["fss"] >= 0 and s >= e["fss"])
            loc_agree[(at_start, fr["loc"] == START_LOC[e["task"]])] += 1
    out["frame_held_vs_state_lifted_and_grasped"] = {f"truth_{a}_pred_{b}": v for (a, b), v in sorted(agree.items())}
    out["frame_startloc_vs_state_not_lifted_not_succeeded"] = {f"truth_{a}_pred_{b}": v for (a, b), v in sorted(loc_agree.items())}
    out["mean_keyframe_spacing_steps"] = {
        "successes": round(float(np.mean([np.diff([f["step"] for f in fr]).mean() for e, fr in rows_ev if e["success"] and len(fr) > 1] or [np.nan])), 1),
        "failures": round(float(np.mean([np.diff([f["step"] for f in fr]).mean() for e, fr in rows_ev if not e["success"] and len(fr) > 1] or [np.nan])), 1)}
    keep = ("n_aborted", "true_aborts", "false_aborts", "precision", "precision_ci95", "recall", "recall_ci95",
            "false_abort_rate", "false_abort_rate_ci95", "saved_frac")
    curves = []
    for T in FAIL_KF_STEPS:
        stale = {}
        for lab, cond in (("successes", True), ("failures", False)):
            v = [T - latest_frame(fr, T)["step"] for e, fr in rows_ev if e["success"] == cond and running(e, T)]
            stale[lab] = round(float(np.mean(v)), 1) if v else None
        row = {"T_step": T, "T_s": T / SIM_HZ, "mean_staleness_of_latest_keyframe_steps_running": stale}
        for nm, ab in (
            ("vis_snapshot_loc", [v_snapshot(T, False)(e, fr) for e, fr in rows_ev]),
            ("vis_snapshot_loc_and_not_held", [v_snapshot(T, True)(e, fr) for e, fr in rows_ev]),
            ("state_snapshot_same_keyframe", [s_snapshot(T)(e, fr) for e, fr in rows_ev]),
            ("state_nolift_dense", [r_nolift(T)(e) for e in E]),
            ("time_only", [r_time(T)(e) for e in E]),
            ("LEAKY_vis_nohold_history", [v_nohold(T)(e, fr) for e, fr in rows_ev]),
            ("LEAKY_vis_nohold+vis_drop_history", [v_combo(v_nohold(T), v_drop())(e, fr) for e, fr in rows_ev]),
        ):
            m = evaluate(E, ab, with_ranking=False)
            row[nm] = {k: m[k] for k in keep}
        curves.append(row)
    out["curves_eval_subset"] = curves
    # dev -> eval operating point: snapshot(T1, use_held) + time_only(T2), per task, zero dev false aborts
    grid = list(itertools.product(FAIL_KF_STEPS + [None], (False, True), GRID_T2))

    def make(T1, use_held, T2):
        return v_combo(v_snapshot(T1, use_held) if T1 else _v_off, v_time(T2))

    chosen, rules = {}, {}
    for task in ("can", "square"):
        D = [(e, fr) for e, fr in rows_dev if e["task"] == task]
        if not D:
            continue
        params, rule = _sel_v(D, make, grid)
        chosen[task] = {"T1": params[0], "use_held": params[1], "T2": params[2], "n_dev": len(D)}
        rules[task] = rule
    sub = [(e, fr) for e, fr in rows_ev if e["task"] in rules]
    Es = [e for e, _ in sub]
    out["dev_to_eval"] = {
        "chosen_on_dev": chosen,
        "vision_snapshot_rule_eval": evaluate(Es, [rules[e["task"]](e, fr) for e, fr in sub]),
        "time_only_same_T2_eval": evaluate(Es, [r_time(chosen[e["task"]]["T2"])(e) for e in Es]),
        "state_snapshot_same_T1_T2_eval": evaluate(Es, [v_combo(s_snapshot(chosen[e["task"]]["T1"]) if chosen[e["task"]]["T1"] else _v_off,
                                                                 v_time(chosen[e["task"]]["T2"]))(e, fr) for e, fr in sub]),
    }
    return out


# ---------------------------------------------------------------- Part 3: Nemotron prefix predictions
def load_neb():
    rows = []
    if NEB.exists():
        for line in NEB.read_text().splitlines():
            rows.append(json.loads(line))
    return rows


def pred_summary(rows):
    ok = [r for r in rows if r["parse_status"] == "ok"]
    y = [int(r["truth_success"]) for r in ok]
    pred = [int(bool(r["will_succeed"])) for r in ok]
    sc = [r["score_p_success"] for r in ok]
    tp = sum(1 for a, b in zip(y, pred) if a and b)
    tn = sum(1 for a, b in zip(y, pred) if not a and not b)
    fp = sum(1 for a, b in zip(y, pred) if not a and b)
    fn = sum(1 for a, b in zip(y, pred) if a and not b)
    npos, nneg = tp + fn, tn + fp
    n = len(ok)
    ya, sa = np.array(y), np.array(sc, float)
    au = auroc(sa, ya)
    # failure-prediction view: "predict failure" = will_succeed false
    return {"n_rows": len(rows), "n_parsed": n, "parse_fail": len(rows) - n,
            "n_success": npos, "n_failure": nneg,
            "accuracy": ratio(tp + tn, n), "accuracy_ci95": wilson(tp + tn, n),
            "balanced_accuracy": round(0.5 * (tp / npos + tn / nneg), 4) if npos and nneg else None,
            "auroc_ordinal": None if not np.isfinite(au) else round(au, 4),
            "auroc_ci95": boot(lambda ix: auroc(sa[ix], ya[ix]), n) if n and npos and nneg else [None, None],
            "tp_pred_success": tp, "tn_pred_failure": tn, "fp_pred_success_but_failed": fp,
            "fn_pred_failure_but_succeeded": fn,
            "fail_pred_precision": ratio(tn, tn + fn), "fail_pred_precision_ci95": wilson(tn, tn + fn),
            "fail_pred_recall": ratio(tn, nneg), "fail_pred_recall_ci95": wilson(tn, nneg),
            "confidence_counts": dict(Counter(f"{r['will_succeed']}/{r['confidence']}" for r in ok)),
            "mean_latency_s": round(float(np.mean([r["latency_s"] for r in rows])), 2) if rows else None}


def part3(dev, ev, time_T2_by_task):
    rows = load_neb()
    out = {"n_calls_rows": len(rows)}
    if not rows:
        return out
    out["call_stats"] = {"n": len(rows), "by_split_cut": dict(Counter(f"{r['split']}/t{r['cut_step']}" for r in rows)),
                         "parse_ok": sum(r["parse_status"] == "ok" for r in rows),
                         "backend_errors": sum(bool(r.get("backend_error")) for r in rows),
                         "mean_latency_s": round(float(np.mean([r["latency_s"] for r in rows])), 2),
                         "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] or 0 for r in rows])), 0),
                         "mean_completion_tokens": round(float(np.mean([r["completion_tokens"] or 0 for r in rows])), 0)}
    devr = [r for r in rows if r["split"] == "dev"]
    out["dev"] = {}
    for v in sorted({r["variant"] for r in devr}):
        rr = [r for r in devr if r["variant"] == v]
        out["dev"][v] = {"all": pred_summary(rr),
                         "by_task": {t: pred_summary([r for r in rr if r["task"] == t]) for t in ("can", "square")}}
    if out["dev"]:
        def key(v):
            s = out["dev"][v]["all"]
            return (s["balanced_accuracy"] or 0, s["auroc_ordinal"] or 0)
        chosen = max(out["dev"], key=key)
    else:
        chosen = None
    out["chosen_variant_on_dev"] = chosen
    evr = [r for r in rows if r["split"] == "eval" and r["variant"] == chosen]
    out["eval"] = {}
    for cut in sorted({r["cut_step"] for r in evr}):
        rr = [r for r in evr if r["cut_step"] == cut]
        out["eval"][f"t{cut:03d}"] = {"cut_s": cut / SIM_HZ, "all": pred_summary(rr),
                                      "by_task": {t: pred_summary([r for r in rr if r["task"] == t]) for t in ("can", "square")},
                                      "excluding_epoch0": pred_summary([r for r in rr if not r["checkpoint_id"].endswith("_e0000")]),
                                      "by_failure_type_pred_failure_rate": {
                                          ft: ratio(sum(1 for r in rr if r["truth_failure_type"] == ft and r["parse_status"] == "ok" and not r["will_succeed"]),
                                                    sum(1 for r in rr if r["truth_failure_type"] == ft and r["parse_status"] == "ok"))
                                          for ft in sorted({r["truth_failure_type"] for r in rr})}}
    # sequential abort policy on eval: check at each available cut; abort if will_succeed is False
    pred = {(r["checkpoint_id"], r["env_seed"], r["cut_step"]): r for r in evr if r["parse_status"] == "ok"}
    cuts = sorted({r["cut_step"] for r in evr})
    called = {(r["checkpoint_id"], r["env_seed"]) for r in evr}
    E = [e for e in ev if (e["checkpoint_id"], e["env_seed"]) in called]

    def nem_rule(conf_ok, use_time):
        def f(e):
            for c in cuts:
                r = pred.get((e["checkpoint_id"], e["env_seed"], c))
                if r is not None and running(e, c) and (not r["will_succeed"]) and r["confidence"] in conf_ok:
                    return c
            if use_time:
                return r_time(time_T2_by_task[e["task"]])(e)
            return None
        return f

    pol = {}
    for nm, conf_ok in (("abort_if_pred_fail_any_conf", CONF_ALL), ("abort_if_pred_fail_high_conf", ("high",))):
        pol[nm] = evaluate(E, [nem_rule(conf_ok, False)(e) for e in E])
        pol[nm + "+time_only_dev_T2"] = evaluate(E, [nem_rule(conf_ok, True)(e) for e in E])
    pol["reference_time_only_dev_T2"] = evaluate(E, [r_time(time_T2_by_task[e["task"]])(e) for e in E])
    # privileged state rule restricted to the same check times (apples to apples)
    def st_rule(e):
        for c in cuts:
            d = r_drop(10)(e)
            if running(e, c) and (e["h"][SETTLE: c + 1].max(initial=-1.0) < LIFT_M or (d is not None and d <= c)):
                return c
        return None
    pol["reference_state_nolift_or_dropped_at_same_cuts"] = evaluate(E, [st_rule(e) for e in E])
    out["eval_policies"] = {"cuts_steps": cuts, "n_episodes": len(E), **pol}
    E0 = [e for e in E if e["epoch"] > 0]
    out["eval_policies_excluding_epoch0"] = {
        "n_episodes": len(E0),
        "abort_if_pred_fail_any_conf": evaluate(E0, [nem_rule(CONF_ALL, False)(e) for e in E0]),
        "abort_if_pred_fail_high_conf": evaluate(E0, [nem_rule(("high",), False)(e) for e in E0]),
        "reference_state_nolift_or_dropped_at_same_cuts": evaluate(E0, [st_rule(e) for e in E0]),
    }
    hc = [r for r in evr if r["parse_status"] == "ok" and not r["will_succeed"] and r["confidence"] == "high"]
    out["high_conf_failure_predictions"] = {
        "n": len(hc), "by_cut_and_epoch0": dict(Counter(f"t{r['cut_step']}/epoch0={r['checkpoint_id'].endswith('_e0000')}/truth_success={r['truth_success']}" for r in hc))}
    return out


CONF_ALL = ("low", "medium", "high")


def strip(o):
    if isinstance(o, dict):
        return {str(k): strip(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/hackathon/explore" / f"{NAME}.json"))
    a = ap.parse_args()
    dev = load_split("dev")
    ev = load_split("eval")
    res = {
        "exploration": NAME, "label": "EXPLORATORY / post-hoc",
        "data": {"eval_episodes": len(ev), "eval_checkpoints": len({e["checkpoint_id"] for e in ev}),
                 "eval_failures": sum(not e["success"] for e in ev), "dev_episodes": len(dev),
                 "eval_by_task": dict(Counter(e["task"] for e in ev)),
                 "eval_last_success_step_by_task": {t: max([e["fss"] for e in ev if e["task"] == t and e["success"]] or [None]) for t in ("can", "square")},
                 "dev_last_success_step_by_task": {t: max([e["fss"] for e in dev if e["task"] == t and e["success"]] or [None]) for t in ("can", "square")}},
    }
    # side check: square nut spawns above the table; failure_annotator uses step-0 height as reference
    sq = [e for e in dev + ev if e["task"] == "square"]
    res["side_check_square_annotation"] = {
        "n_square_episodes_dev_plus_eval": len(sq),
        "median_z_step0_m": round(float(np.median([e["z0_step0"] for e in sq])), 3) if sq else None,
        "median_z_settled_steps5to10_m": round(float(np.median([e["z_settled"] for e in sq])), 3) if sq else None,
        "label_other_count": sum(e["failure_type"] == "other" for e in sq),
        "label_other_with_final_height_within_1cm_of_settled": sum(
            e["failure_type"] == "other" and abs(e["z_final"] - e["z_settled"]) < 0.01 for e in sq),
        "annotated_not_lifted_but_lifted_5cm_above_settled": sum(
            (not e["annot_lifted"]) and bool((e["h"][SETTLE: (e["fss"] + 1 if e["fss"] >= 0 else e["end"] + 1)] >= LIFT_M).any())
            for e in sq),
    }
    res["part1_state"] = part1(dev, ev)
    t2 = {t: res["part1_state"]["dev_to_eval"]["time_only/maxfp0"]["chosen_on_dev"][t]["params"][0] for t in ("can", "square")}
    res["part2_cosmos_v7"] = part2(dev, ev)
    res["part3_nemotron_prefix"] = part3(dev, ev, t2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(strip(res), indent=1))
    print(json.dumps({"wrote": a.out, "eval_episodes": len(ev), "v7_rows": res["part2_cosmos_v7"].get("n_eval_rows"),
                      "nebius_rows": res["part3_nemotron_prefix"].get("n_calls_rows")}))


if __name__ == "__main__":
    main()
