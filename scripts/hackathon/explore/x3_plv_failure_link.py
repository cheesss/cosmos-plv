#!/usr/bin/env python3
"""EXPLORATORY / POST-HOC (x3_plv_failure_link): link PLV replanning diagnostics to physical failure modes.

Question: when a PLV user sees high execution-matched replanning disagreement (or a bad cheap
score) for a checkpoint / episode, WHAT kind of physical failure does it predict
(no_grasp, dropped, other) -- and does disagreement spike BEFORE a drop?

Parts
  (1) episode level: per eval episode, execution-matched replanning statistics on REAL sim
      feedback (cosmos_plv.cheap_metrics.replan_metrics_from_plans, scale = per-dim std of the
      10-D predicted actions pooled over all eval episodes of the task; outcome-free), per-chunk
      (time-local) versions, executed smoothness, gripper chatter.  AUROC within task for
      success-vs-failure and each failure type, checkpoint-clustered bootstrap CI.
      Outcome-free "high" thresholds (75th pct of the feature distribution, from the OTHER run for
      can; pooled for square) -> failure composition of high vs low episodes (Wilson CI).
  (2) temporal: event-triggered average of per-boundary disagreement around drop events (from
      npz traces) vs a within-episode random-time permutation null and vs a phase-matched
      control (intentional release in successful episodes).  Also with Nemotron-timed events.
  (3) checkpoint level: failure composition (sim labels and Nemotron v2 predictions) vs epoch,
      training-time success, val_loss, train_action_mse_error, real-feedback PLV metrics.

Nothing is fitted on outcomes: features are used raw with a pre-declared direction
(higher = worse); thresholds are outcome-free quantiles.  Runs in Environment B, CPU only, no
model / API calls.  Re-runnable: works on whatever rollouts/logs exist when run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cosmos_plv.cheap_metrics import executed_smoothness, replan_alignment, replan_metrics_from_plans  # noqa: E402
from cosmos_plv.checkpoint_adapter import discover_checkpoints  # noqa: E402
from cosmos_plv.provenance import git_commit, utc_now, write_json  # noqa: E402
from cosmos_plv.ranking import spearman  # noqa: E402

NAME = "x3_plv_failure_link"
LIFT_M, TABLE_M, REACH_M = 0.05, 0.02, 0.05  # same thresholds as cosmos_plv.failure_annotator
PREFIX_STEPS = 80  # early window (10 chunks = 4.0 s); 96 was the a-priori choice but the earliest success
# (any split) is at step 85, so 80 is the longest chunk-aligned window that precedes every success (checked below)
DEBOUNCE = 3  # grasp contact must be absent for 3 consecutive steps to count as "lost"
DIM_GROUPS = {"all": slice(0, 10), "pos": slice(0, 3), "rot": slice(3, 9), "grip": slice(9, 10)}
SIM_HZ = 20.0


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def auroc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_fast(x, y):
    """AUROC of score x for label y (1 = positive) incl. ties; x finite."""
    p, n = x[y == 1], x[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def clustered_boot_auroc(x, y, clusters, n_boot=2000, seed=0):
    """Percentile CI of AUROC resampling whole clusters (checkpoints) with replacement."""
    rng = np.random.default_rng(seed)
    keys = np.unique(clusters)
    idx = [np.nonzero(clusters == k)[0] for k in keys]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([idx[i] for i in pick])
        v = auroc_fast(x[ix], y[ix])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return [float("nan"), float("nan")], len(vals)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], len(vals)


def _strat_parts(x, y, cl):
    parts = {}
    for k in np.unique(cl):
        m = cl == k
        p, n = x[m & (y == 1)], x[m & (y == 0)]
        if len(p) and len(n):
            conc = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            parts[k] = (float(conc), float(len(p) * len(n)))
    return parts


def strat_auroc(x, y, cl):
    """Within-checkpoint AUROC: only (pos, neg) pairs from the SAME checkpoint are compared, so
    checkpoint-level quality (training progress) cannot drive it.  Returns (auroc, n_pairs, n_ckpts)."""
    parts = _strat_parts(x, y, cl)
    tot = sum(v[1] for v in parts.values())
    return (sum(v[0] for v in parts.values()) / tot if tot else float("nan")), int(tot), len(parts)


def clustered_boot_strat(x, y, cl, n_boot=2000, seed=0):
    parts = _strat_parts(x, y, cl)
    keys = np.unique(cl)
    conc = np.array([parts.get(k, (0.0, 0.0))[0] for k in keys])
    pairs = np.array([parts.get(k, (0.0, 0.0))[1] for k in keys])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), (n_boot, len(keys)))
    c, pr = conc[pick].sum(1), pairs[pick].sum(1)
    v = c[pr > 0] / pr[pr > 0]
    if len(v) < 50:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ----------------------------------------------------------------------------- episode loading
def load_episodes(rollout_dirs):
    eps = []
    for rd in rollout_dirs:
        for summ in sorted(Path(rd).glob("*/summary.json")):
            s = json.loads(summ.read_text())
            for ep in s["episodes"]:
                npz = ROOT / ep["episode_npz"]
                if not npz.is_file():
                    continue
                d = dict(np.load(npz))
                eps.append({"meta": ep, "d": d, "split": Path(rd).name})
    return eps


def task_scales(eps):
    """Per-dim std of the 10-D predicted actions pooled over ALL eval episodes of a task (outcome-free)."""
    pooled = {}
    for e in eps:
        if e["split"] != "hackathon_rollouts_eval":
            continue
        pooled.setdefault(e["meta"]["task"], []).append(e["d"]["action_pred"].reshape(-1, 10))
    return {t: np.concatenate(v).std(axis=0) + 1e-6 for t, v in pooled.items()}


def per_boundary(ap, plan_t, scale, H, n_obs, n_act):
    """Same alignment as cheap_metrics.replan_metrics_from_plans, but kept per boundary and per dim group."""
    al = replan_alignment(H, n_obs, n_act)
    rows = []
    for c in range(len(ap) - 1):
        if int(plan_t[c + 1]) - int(plan_t[c]) != n_act:
            continue
        old = ap[c, al["old_slice"][0]: al["old_slice"][1]] / scale
        new = ap[c + 1, al["new_slice"][0]: al["new_slice"][1]] / scale
        bo = ap[c, al["boundary_old_index"]] / scale
        bn = ap[c + 1, al["boundary_new_index"]] / scale
        r = {"b": int(plan_t[c + 1])}
        for g, sl in DIM_GROUPS.items():
            r[f"ov_{g}"] = float(np.abs(old[:, sl] - new[:, sl]).mean())
            r[f"jump_{g}"] = float(np.abs(bo[sl] - bn[sl]).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def grip_flips(a):
    s = np.sign(np.asarray(a, float))
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum()) if len(s) > 1 else 0


def lost_grasp_time(grasped, start, h=None, min_h=None):
    """First t >= start with grasped[t-1] and no contact for DEBOUNCE steps (optionally object still >= min_h)."""
    T = len(grasped)
    for t in range(max(1, start), T - DEBOUNCE + 1):
        if grasped[t - 1] and not grasped[t:t + DEBOUNCE].any():
            if min_h is None or h[t] >= min_h:
                return t
    return -1


def events(d, meta):
    obj, eef = d["object_pos"].astype(float), d["eef_pos"].astype(float)
    g = d["grasped"].astype(bool)
    h = obj[:, 2] - obj[0, 2]
    T = len(h)
    lifted = np.nonzero(h >= LIFT_M)[0]
    first_lift = int(lifted[0]) if len(lifted) else -1
    near = np.nonzero(np.linalg.norm(eef - obj, axis=1) < REACH_M)[0]
    first_reach = int(near[0]) if len(near) else -1
    ev = {"first_lift": first_lift, "first_reach": first_reach, "drop_t": -1, "drop_rule": None,
          "release_t": -1, "n_steps_trace": T}
    if first_lift >= 0:
        below = np.nonzero(h[first_lift:] < TABLE_M)[0]
        t_below = int(first_lift + below[0]) if len(below) else -1
        t_lost = lost_grasp_time(g, first_lift + 1, h, TABLE_M)
        if meta["failure_type"] == "dropped":
            cands = [(t, r) for t, r in ((t_below, "fell_below_2cm"), (t_lost, "contact_lost_while_lifted")) if t >= 0]
            if cands:
                ev["drop_t"], ev["drop_rule"] = min(cands)
            else:  # lifted, never held after lift and never fell below 2 cm: first no-contact step
                nc = np.nonzero(~g[first_lift:])[0]
                ev["drop_t"], ev["drop_rule"] = (int(first_lift + nc[0]), "no_contact_after_lift") if len(nc) else (-1, None)
        if meta["success"]:
            t_rel = lost_grasp_time(g, first_lift + 1)
            fs = int(meta["first_success_step"])
            ev["release_t"] = t_rel if 0 <= t_rel <= fs + 1 else fs
    return ev


def episode_table(eps, scales):
    rows, series = [], {}
    for e in eps:
        m, d = e["meta"], e["d"]
        H, n_obs, n_act = int(m["horizon"]), int(m["n_obs_steps"]), int(m["n_action_steps"])
        sc = scales[m["task"]]
        rp = replan_metrics_from_plans(d["action_pred"], d["plan_t"], H, n_obs, n_act, sc)
        pb = per_boundary(d["action_pred"].astype(np.float64), d["plan_t"], sc, H, n_obs, n_act)
        ev = events(d, m)
        key = (m["checkpoint_id"], int(m["env_seed"]))
        series[key] = pb
        act = d["action"]
        n = len(act)
        r = {"split": e["split"], "checkpoint_id": m["checkpoint_id"], "env_seed": int(m["env_seed"]),
             "task": m["task"], "run": m["checkpoint_id"].rsplit("_e", 1)[0],
             "epoch": int(m["checkpoint_id"].rsplit("_e", 1)[1]), "success": bool(m["success"]),
             "failure_type": m["failure_type"], "n_steps": int(m["n_steps"]),
             "first_success_step": int(m["first_success_step"]),
             "sim_replan_overlap_l1": rp["sim_replan_overlap_l1"], "sim_boundary_jump_l1": rp["sim_boundary_jump_l1"],
             "replan_pairs": rp["sim_replan_pairs"], **ev}
        # consistency with the shared helper
        r["check_overlap_absdiff"] = abs(float(pb["ov_all"].mean()) - rp["sim_replan_overlap_l1"]) if len(pb) else float("nan")
        for g in DIM_GROUPS:
            r[f"ov_{g}_mean"] = float(pb[f"ov_{g}"].mean()) if len(pb) else float("nan")
            r[f"jump_{g}_mean"] = float(pb[f"jump_{g}"].mean()) if len(pb) else float("nan")
        r["ov_all_max"] = float(pb["ov_all"].max()) if len(pb) else float("nan")
        r["ov_all_p90"] = float(np.percentile(pb["ov_all"], 90)) if len(pb) else float("nan")
        r["jump_all_max"] = float(pb["jump_all"].max()) if len(pb) else float("nan")
        r["ov_grip_max"] = float(pb["ov_grip"].max()) if len(pb) else float("nan")
        r["exec_smoothness"] = executed_smoothness(act)
        r["grip_flips"] = grip_flips(act[:, 6])
        r["grip_flip_rate_per100"] = 100.0 * r["grip_flips"] / max(n, 1)
        # early prefix window (identical length for every episode, precedes every success)
        pre = pb[pb.b <= PREFIX_STEPS]
        r["ov_all_pre"] = float(pre["ov_all"].mean()) if len(pre) else float("nan")
        r["jump_all_pre"] = float(pre["jump_all"].mean()) if len(pre) else float("nan")
        r["ov_grip_pre"] = float(pre["ov_grip"].mean()) if len(pre) else float("nan")
        r["ov_pos_pre"] = float(pre["ov_pos"].mean()) if len(pre) else float("nan")
        r["exec_smoothness_pre"] = executed_smoothness(act[:PREFIX_STEPS])
        r["grip_flips_pre"] = grip_flips(act[:PREFIX_STEPS, 6])
        # strictly pre-lift (approach + grasp) disagreement: boundaries at or before the first lift.  The lift
        # happens in successes AND drops, so this is a precursor window that never contains the drop itself.
        pl = pb[pb.b <= ev["first_lift"]] if ev["first_lift"] >= 0 else pb.iloc[0:0]
        for g in ("all", "pos", "grip"):
            r[f"ov_{g}_prelift"] = float(pl[f"ov_{g}"].mean()) if len(pl) else float("nan")
        # phase-local windows (+-16 steps = +-2 chunks around first reach / first lift)
        for evn, col in (("first_reach", "reach"), ("first_lift", "lift")):
            t0 = ev[evn]
            w = pb[(pb.b >= t0 - 16) & (pb.b < t0 + 16)] if t0 >= 0 else pb.iloc[0:0]
            r[f"ov_all_{col}win"] = float(w["ov_all"].mean()) if len(w) else float("nan")
            r[f"ov_grip_{col}win"] = float(w["ov_grip"].mean()) if len(w) else float("nan")
        rows.append(r)
    return pd.DataFrame(rows), series


FEATURES = [
    ("sim_replan_overlap_l1", "replan overlap L1 (episode mean, real feedback)"),
    ("sim_boundary_jump_l1", "boundary jump L1 (episode mean)"),
    ("ov_pos_mean", "overlap, position dims"),
    ("ov_rot_mean", "overlap, rotation-6d dims"),
    ("ov_grip_mean", "overlap, gripper dim"),
    ("ov_all_max", "overlap, max over boundaries"),
    ("ov_all_p90", "overlap, p90 over boundaries"),
    ("jump_all_max", "boundary jump, max"),
    ("exec_smoothness", "executed smoothness (mean sq. 2nd diff)"),
    ("grip_flips", "gripper sign flips (count)"),
    ("grip_flip_rate_per100", "gripper flips per 100 steps"),
    ("ov_all_pre", "overlap, first 96 steps only"),
    ("jump_all_pre", "boundary jump, first 96 steps only"),
    ("ov_grip_pre", "gripper overlap, first 96 steps only"),
    ("exec_smoothness_pre", "executed smoothness, first 96 steps"),
    ("grip_flips_pre", "gripper flips, first 96 steps"),
    ("ov_all_prelift", "overlap, boundaries up to the first lift (never contains a drop)"),
    ("ov_pos_prelift", "position overlap, up to the first lift"),
    ("ov_grip_prelift", "gripper overlap, up to the first lift"),
    ("ov_all_reachwin", "overlap, +-16 steps around first reach"),
    ("ov_all_liftwin", "overlap, +-16 steps around first lift"),
    ("ov_grip_liftwin", "gripper overlap, +-16 steps around first lift"),
    ("n_steps", "episode length (reference: trivially separates; not a PLV metric)"),
]


def contrasts(task):
    c = [("failure_vs_success", lambda f: f != "none", lambda f: f == "none"),
         ("dropped_vs_success", lambda f: f == "dropped", lambda f: f == "none"),
         ("no_grasp_vs_success", lambda f: f == "no_grasp", lambda f: f == "none"),
         ("dropped_vs_no_grasp", lambda f: f == "dropped", lambda f: f == "no_grasp"),
         ("dropped_vs_other_failures", lambda f: f == "dropped", lambda f: f not in ("none", "dropped"))]
    if task == "square":
        c.append(("other_vs_success", lambda f: f == "other", lambda f: f == "none"))
    return c


def auroc_table(E, n_boot):
    out = []
    cohorts = {"eval_all": E[E.split == "hackathon_rollouts_eval"],
               "eval_trained_epoch_gt0": E[(E.split == "hackathon_rollouts_eval") & (E.epoch > 0)],
               "dev_all": E[E.split == "hackathon_rollouts_dev"]}
    for coh, C in cohorts.items():
        for task, T in C.groupby("task"):
            for cname, fpos, fneg in contrasts(task):
                sub = T[T.failure_type.map(fpos) | T.failure_type.map(fneg)].copy()
                sub["y"] = sub.failure_type.map(fpos).astype(int)
                npos, nneg = int(sub.y.sum()), int((1 - sub.y).sum())
                if npos < 3 or nneg < 3:
                    continue
                for feat, label in FEATURES:
                    s2 = sub[np.isfinite(sub[feat].astype(float))]
                    a = auroc(s2[s2.y == 1][feat], s2[s2.y == 0][feat])
                    if coh.startswith("dev") or s2.checkpoint_id.nunique() < 3:
                        ci, nb = [float("nan"), float("nan")], 0
                    else:
                        ci, nb = clustered_boot_auroc(s2[feat].values.astype(float), s2.y.values,
                                                      s2.checkpoint_id.values, n_boot=n_boot)
                    x, yy, cl = s2[feat].values.astype(float), s2.y.values, s2.checkpoint_id.values
                    sa, npairs, nck = strat_auroc(x, yy, cl)
                    if coh.startswith("dev") or nck < 3:
                        sci = [float("nan"), float("nan")]
                    else:
                        sci = clustered_boot_strat(x, yy, cl, n_boot=n_boot)
                    out.append({"cohort": coh, "task": task, "contrast": cname, "feature": feat, "label": label,
                                "n_pos": int(s2.y.sum()), "n_neg": int((1 - s2.y).sum()),
                                "n_checkpoints": int(s2.checkpoint_id.nunique()), "auroc": a,
                                "ci_lo": ci[0], "ci_hi": ci[1], "n_boot_valid": nb,
                                "within_ckpt_auroc": sa, "within_ckpt_ci_lo": sci[0], "within_ckpt_ci_hi": sci[1],
                                "within_ckpt_pairs": npairs, "within_ckpt_n_checkpoints_both_classes": nck})
    return pd.DataFrame(out)


def high_low_table(E, feats=("sim_replan_overlap_l1", "sim_boundary_jump_l1", "ov_grip_mean", "ov_all_pre",
                              "grip_flip_rate_per100", "exec_smoothness", "ov_all_liftwin")):
    """Outcome-free threshold: 75th pct of the feature. can: from the OTHER run (leave-one-run-out); square: pooled."""
    out = []
    ev = E[E.split == "hackathon_rollouts_eval"]
    for coh, C in {"eval_all": ev, "eval_trained_epoch_gt0": ev[ev.epoch > 0]}.items():
        for task, T in C.groupby("task"):
            runs = sorted(T.run.unique())
            for feat in feats:
                parts = []
                for run in runs:
                    R = T[T.run == run]
                    ref = T[T.run != run] if len(runs) > 1 else T
                    thr = float(np.nanpercentile(ref[feat].astype(float), 75))
                    R = R[np.isfinite(R[feat].astype(float))].copy()
                    R["high"] = R[feat] > thr
                    R["thr"] = thr
                    parts.append(R)
                P = pd.concat(parts)
                row = {"cohort": coh, "task": task, "feature": feat,
                       "threshold_rule": "q75 of other run (outcome-free)" if len(runs) > 1 else "q75 pooled (outcome-free)",
                       "thresholds": {r: float(p.thr.iloc[0]) for r, p in zip(runs, parts) if len(p)}}
                for grp, G in (("high", P[P.high]), ("low", P[~P.high])):
                    n = len(G)
                    row[f"{grp}_n"] = n
                    for ft in ("none", "no_grasp", "dropped", "other", "not_placed"):
                        k = int((G.failure_type == ft).sum())
                        row[f"{grp}_{ft}_k"] = k
                        row[f"{grp}_{ft}_rate"] = k / n if n else float("nan")
                        row[f"{grp}_{ft}_wilson"] = wilson(k, n)
                out.append(row)
    return out


# ----------------------------------------------------------------------------- temporal
def norm_series(pb, col):
    v = np.log(pb[col].values + 1e-6)
    return v - np.median(v)


def eta(E, series, event_col, cols, lags=range(-6, 5), win=(-3, -1), n_perm=5000, seed=0, mask=None):
    """Event-triggered average of within-episode log-normalised disagreement.

    lag k = boundary b with floor((b - t_e)/8) == k.  Pre-event statistic = mean over lags in `win`.
    Null: same statistic with t_e replaced by a uniformly random step of the same episode at which the
    full window exists (permutation of event time within episode), averaged over events.
    """
    rng = np.random.default_rng(seed)
    sel = E[(E[event_col] >= 0)] if mask is None else E[mask & (E[event_col] >= 0)]
    res = {"n_events": int(len(sel)), "n_checkpoints": int(sel.checkpoint_id.nunique()), "by_col": {}}
    for col in cols:
        lag_vals = {k: [] for k in lags}
        lag_raw = {k: [] for k in lags}
        pre_obs, ep_data = [], []
        for _, r in sel.iterrows():
            pb = series[(r.checkpoint_id, r.env_seed)]
            if len(pb) < 6:
                continue
            z = norm_series(pb, col)
            b = pb.b.values
            te = int(r[event_col])
            lagidx = np.floor((b - te) / 8).astype(int)
            for k in lags:
                hit = np.nonzero(lagidx == k)[0]
                if len(hit):
                    lag_vals[k].append(float(z[hit[0]]))
                    lag_raw[k].append(float(pb[col].values[hit[0]]))
            w = (lagidx >= win[0]) & (lagidx <= win[1])
            if w.sum() == (win[1] - win[0] + 1):
                pre_obs.append(float(z[w].mean()))
                ep_data.append((b, z))
        if not pre_obs:
            continue
        obs = float(np.mean(pre_obs))
        draws = []
        for b, z in ep_data:
            lo, hi = int(b.min()) - 8 * win[0], int(b.max()) - 8 * win[1]  # te such that window exists
            wm = []
            for te in range(lo, max(lo, hi) + 1):
                li = np.floor((b - te) / 8).astype(int)
                w = (li >= win[0]) & (li <= win[1])
                wm.append(z[w].mean() if w.any() else np.nan)
            wm = np.asarray(wm)
            draws.append(wm[rng.integers(0, len(wm), n_perm)])
        null = np.nanmean(np.vstack(draws), axis=0)
        boot = []
        pv = np.array(pre_obs)
        for _ in range(2000):
            boot.append(pv[rng.integers(0, len(pv), len(pv))].mean())
        res["by_col"][col] = {
            "lag_mean_lognorm": {int(k): (float(np.mean(v)) if v else None) for k, v in lag_vals.items()},
            "lag_n": {int(k): len(v) for k, v in lag_vals.items()},
            "lag_median_raw": {int(k): (float(np.median(v)) if v else None) for k, v in lag_raw.items()},
            "pre_window_lags": list(win), "pre_stat_obs": obs, "n_events_with_window": len(pre_obs),
            "pre_stat_boot_ci_events": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "null_mean": float(np.nanmean(null)), "null_ci": [float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5))],
            "p_one_sided_obs_ge_null": float((1 + (null >= obs).sum()) / (1 + n_perm)),
            "p_one_sided_obs_le_null": float((1 + (null <= obs).sum()) / (1 + n_perm)),
        }
    return res


def pre_window_raw(E, series, event_col, mask, col, win=(-3, -1)):
    rows = []
    for _, r in E[mask & (E[event_col] >= 0)].iterrows():
        pb = series[(r.checkpoint_id, r.env_seed)]
        li = np.floor((pb.b.values - int(r[event_col])) / 8).astype(int)
        w = (li >= win[0]) & (li <= win[1])
        if w.sum() == win[1] - win[0] + 1:
            rows.append({"checkpoint_id": r.checkpoint_id, "task": r.task, "epoch": r.epoch, "v": float(pb[col].values[w].mean())})
    return pd.DataFrame(rows)


def phase_matched_compare(E, series, n_boot):
    """Raw (task-scaled) disagreement in the 3 boundaries before the event: drop (failures) vs intentional
    release (successes) -- both are 'object leaves the gripper after a lift'.  AUROC(drop > release)."""
    out = []
    ev = E.split == "hackathon_rollouts_eval"
    for coh, cm in (("eval_all", ev), ("eval_trained_epoch_gt0", ev & (E.epoch > 0))):
        for (ev_d, ev_r), col in [(pair, c) for pair in (("drop_t", "release_t"), ("first_lift", "first_lift"))
                                  for c in ("ov_all", "ov_grip", "ov_pos", "jump_all")]:
            D = pre_window_raw(E, series, ev_d, cm & (E.failure_type == "dropped"), col)
            R = pre_window_raw(E, series, ev_r, cm & E.success, col)
            if not len(D) or not len(R):
                continue
            for task in sorted(set(D.task) | set(R.task)):
                d, r = D[D.task == task], R[R.task == task]
                if len(d) < 3 or len(r) < 3:
                    continue
                x = np.concatenate([d.v.values, r.v.values])
                y = np.concatenate([np.ones(len(d), int), np.zeros(len(r), int)])
                cl = np.concatenate([d.checkpoint_id.values, r.checkpoint_id.values])
                ci, _ = clustered_boot_auroc(x, y, cl, n_boot=n_boot)
                sa, npairs, nck = strat_auroc(x, y, cl)
                out.append({"cohort": coh, "task": task, "col": col, "event_dropped_ep": ev_d, "event_success_ep": ev_r,
                            "n_drop": int(len(d)), "n_release": int(len(r)),
                            "median_drop": float(d.v.median()), "median_release": float(r.v.median()),
                            "auroc_drop_gt_release": auroc_fast(x, y), "ci_lo": ci[0], "ci_hi": ci[1],
                            "within_ckpt_auroc": sa, "within_ckpt_pairs": npairs, "within_ckpt_n_ckpts": nck})
    return out


# ----------------------------------------------------------------------------- VLM timing
def vlm_timing(E, nebius_path, v7_path):
    out = {}
    ev = E[(E.split == "hackathon_rollouts_eval")]
    if Path(nebius_path).is_file():
        rows = [json.loads(l) for l in open(nebius_path)]
        nb = pd.DataFrame([{"checkpoint_id": r["checkpoint_id"], "env_seed": int(r["env_seed"]),
                            "neb_failure_time_s": r.get("cosmos_failure_time_s"), "neb_failure_type": r.get("cosmos_failure_type"),
                            "neb_success": r.get("cosmos_task_success"), "neb_scored": bool(r.get("scored"))} for r in rows])
        J = ev.merge(nb, on=["checkpoint_id", "env_seed"], how="inner")
        D = J[(J.failure_type == "dropped") & (J.drop_t >= 0) & J.neb_failure_time_s.notna()].copy()
        D["sim_drop_s"] = D.drop_t / SIM_HZ
        D["err_s"] = D.neb_failure_time_s.astype(float) - D.sim_drop_s
        ae = D.err_s.abs().values
        rng = np.random.default_rng(0)
        bmed = [np.median(ae[rng.integers(0, len(ae), len(ae))]) for _ in range(2000)] if len(ae) else []
        out["nemotron_v2_drop_timing"] = {
            "n_dropped_with_time": int(len(D)),
            "median_abs_err_s": float(np.median(ae)) if len(ae) else None,
            "median_abs_err_ci": [float(np.percentile(bmed, 2.5)), float(np.percentile(bmed, 97.5))] if len(ae) else None,
            "within_1s": [int((ae <= 1).sum()), int(len(ae)), wilson(int((ae <= 1).sum()), len(ae))],
            "within_2s": [int((ae <= 2).sum()), int(len(ae)), wilson(int((ae <= 2).sum()), len(ae))],
            "spearman_time": spearman(D.neb_failure_time_s.astype(float), D.sim_drop_s) if len(D) >= 3 else None,
            "per_episode": D[["checkpoint_id", "env_seed", "drop_rule", "sim_drop_s", "neb_failure_time_s", "neb_failure_type", "err_s"]].to_dict("records"),
        }
        J["neb_drop_t"] = np.where(J.neb_failure_time_s.notna(), np.round(J.neb_failure_time_s.astype(float) * SIM_HZ), -1).astype(int)
        out["_joined"] = J
    if Path(v7_path).is_file():
        rows = [json.loads(l) for l in open(v7_path)]
        rec = []
        for r in rows:
            tr = r.get("track") or {}
            fr = tr.get("frames") or []
            held = [f.get("p_held_by_gripper") for f in fr]
            times = tr.get("times_s") or []
            t_lost = None
            seen = False
            for t, hh in zip(times, held):
                if hh is True:
                    seen = True
                elif seen and hh is False:
                    t_lost = t
                    break
            rec.append({"checkpoint_id": r["checkpoint_id"], "env_seed": int(r["env_seed"]), "v7_t_lost_s": t_lost,
                        "v7_any_held": seen, "v7_times": list(times)})
        V = ev.merge(pd.DataFrame(rec), on=["checkpoint_id", "env_seed"], how="inner")
        Dv = V[(V.failure_type == "dropped") & (V.drop_t >= 0)]
        out["cosmos_v7_track_drop_timing"] = {
            "n_v7_rows_joined": int(len(V)), "n_dropped": int(len(Dv)),
            "n_dropped_with_held_then_released": int(Dv.v7_t_lost_s.notna().sum()),
            "per_episode": [{"checkpoint_id": a.checkpoint_id, "env_seed": int(a.env_seed), "sim_drop_s": a.drop_t / SIM_HZ,
                             "v7_t_lost_s": (None if a.v7_t_lost_s is None or not np.isfinite(a.v7_t_lost_s) else float(a.v7_t_lost_s)),
                             "first_keyframe_at_or_after_sim_drop_s": next((t for t in a.v7_times if t >= a.drop_t / SIM_HZ), None)}
                            for a in Dv.itertuples()],
            "note": "keyframe spacing ~2.9 s; best possible detection = first keyframe at/after the sim drop",
        }
    return out


def droptimer_eval(E, scratch, nebius_v2_path, n_boot):
    """Part 4: Nemotron time-stamped drop detector (scripts/hackathon/explore/x3_nemotron_drop_timer.py).
    Pre-declared rule (same on dev and eval): predicted drop = left_gripper_before_goal AND NOT task_success.
    Variant chosen on dev by balanced accuracy of drop detection; eval is reported with the frozen variant."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from x3_nemotron_drop_timer import parse as dt_parse
    files = {("droptimer_v1", "dev"): scratch / "nemotron_droptimer_dev.jsonl",
             ("droptimer_v2", "dev"): scratch / "nemotron_droptimer_v2_dev.jsonl",
             ("droptimer_v1", "eval"): scratch / "nemotron_droptimer_eval.jsonl"}
    Ei = E.set_index(["checkpoint_id", "env_seed"])
    out, frames = {}, {}
    for (var, split), f in files.items():
        if not f.is_file():
            continue
        rows = []
        for line in f.read_text().splitlines():
            r = json.loads(line)
            pobj, st = dt_parse(r.get("raw_text", ""))
            key = (r["checkpoint_id"], int(r["env_seed"]))
            if key not in Ei.index:
                continue
            e = Ei.loc[key]
            pred = bool(pobj and pobj["left_gripper_before_goal"] and not pobj["task_success"]) if pobj else None
            rows.append({"checkpoint_id": key[0], "env_seed": key[1], "task": e.task, "epoch": int(e.epoch),
                         "failure_type": e.failure_type, "truth_drop": e.failure_type == "dropped",
                         "drop_t": int(e.drop_t), "parse_status": st, "pred_drop": pred,
                         "pred_success": (bool(pobj["task_success"]) if pobj else None),
                         "pred_time_s": (pobj.get("time_left_gripper_s") if pobj else None),
                         "latency_s": r.get("latency_s")})
        D = pd.DataFrame(rows).drop_duplicates(["checkpoint_id", "env_seed"], keep="last")
        frames[(var, split)] = D
        Ds = D[D.pred_drop.notna()]
        res = {"n_rows": int(len(D)), "n_parsed": int(len(Ds)), "parse_status": D.parse_status.value_counts().to_dict(),
               "median_latency_s": float(D.latency_s.median()) if len(D) else None}
        for task, T in [("all", Ds)] + list(Ds.groupby("task")):
            tp = int((T.truth_drop & T.pred_drop.astype(bool)).sum()); fn = int((T.truth_drop & ~T.pred_drop.astype(bool)).sum())
            fp = int((~T.truth_drop & T.pred_drop.astype(bool)).sum()); tn = int((~T.truth_drop & ~T.pred_drop.astype(bool)).sum())
            sens = tp / (tp + fn) if tp + fn else float("nan"); spec = tn / (tn + fp) if tn + fp else float("nan")
            fp_by = T[~T.truth_drop & T.pred_drop.astype(bool)].failure_type.value_counts().to_dict()
            res[task] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sensitivity": sens, "sens_wilson": wilson(tp, tp + fn),
                         "specificity": spec, "spec_wilson": wilson(tn, tn + fp),
                         "balanced_accuracy": (sens + spec) / 2, "false_positives_by_truth_type": fp_by,
                         "ppv": tp / (tp + fp) if tp + fp else float("nan"), "ppv_wilson": wilson(tp, tp + fp)}
        TP = Ds[Ds.truth_drop & Ds.pred_drop.astype(bool) & Ds.pred_time_s.notna() & (Ds.drop_t >= 0)]
        if len(TP):
            err = TP.pred_time_s.astype(float).values - TP.drop_t.values / SIM_HZ
            ae = np.abs(err)
            res["timing_true_positives"] = {"n": int(len(ae)), "median_abs_err_s": float(np.median(ae)),
                                            "within_1s": [int((ae <= 1).sum()), int(len(ae)), wilson(int((ae <= 1).sum()), len(ae))],
                                            "within_2s": [int((ae <= 2).sum()), int(len(ae)), wilson(int((ae <= 2).sum()), len(ae))],
                                            "signed_err_s": [round(float(x), 2) for x in err]}
        out[f"{var}_{split}"] = res
    # same episodes, existing whole-video Nemotron v2 diagnostic (drop = failure_type 'dropped' or object_dropped & not success)
    De = frames.get(("droptimer_v1", "eval"))
    if De is not None and Path(nebius_v2_path).is_file():
        nb = pd.DataFrame([json.loads(l) for l in open(nebius_v2_path)])
        nb = nb[nb.scored.fillna(False).astype(bool)]
        nb["v2_pred_drop"] = (nb.cosmos_failure_type == "dropped") | (nb.cosmos_object_dropped.fillna(False).astype(bool) & ~nb.cosmos_task_success.astype(bool))
        J = De.merge(nb[["checkpoint_id", "env_seed", "v2_pred_drop"]], on=["checkpoint_id", "env_seed"], how="inner")
        J = J[J.pred_drop.notna()]
        comp = {"n_common": int(len(J))}
        for col in ("pred_drop", "v2_pred_drop"):
            t = J.truth_drop; pz = J[col].astype(bool)
            tp, fn, fp, tn = int((t & pz).sum()), int((t & ~pz).sum()), int((~t & pz).sum()), int((~t & ~pz).sum())
            comp[col] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sens_wilson": wilson(tp, tp + fn), "spec_wilson": wilson(tn, tn + fp),
                         "sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": tn / (tn + fp) if tn + fp else None}
        out["eval_same_episodes_droptimer_vs_nebius_v2"] = comp
    if De is not None:
        # checkpoint level: predicted drop rate vs simulator drop rate (within task)
        g = De[De.pred_drop.notna()].groupby("checkpoint_id")
        C = pd.DataFrame({"task": g.task.first(), "epoch": g.epoch.first(), "n": g.size(),
                          "sim_drop_rate": g.truth_drop.mean(), "pred_drop_rate": g.pred_drop.apply(lambda s: s.astype(bool).mean())}).reset_index()
        cl = {}
        for task, T in C.groupby("task"):
            x, y = T.pred_drop_rate.values.astype(float), T.sim_drop_rate.values.astype(float)
            rho = spearman(x, y)
            rng = np.random.default_rng(0)
            bs = [spearman(x[ix], y[ix]) for ix in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
            bs = [b for b in bs if np.isfinite(b)]
            cl[task] = {"n_checkpoints": int(len(T)), "spearman": rho,
                        "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 50 else None,
                        "mean_abs_diff": float(np.abs(x - y).mean()), "rows": T.round(3).to_dict("records")}
        out["eval_checkpoint_drop_rate"] = cl
    return out, frames


# ----------------------------------------------------------------------------- checkpoint level
def load_checkpoint_records(pool):
    """discover_checkpoints() needs omegaconf (absent in Env B): fall back to parsing logs + run name."""
    from types import SimpleNamespace
    import re
    try:
        return {r.checkpoint_id: r for r in discover_checkpoints(Path(pool))}
    except ImportError:
        pass
    from cosmos_plv.checkpoint_adapter import CKPT_RE, _read_logs
    from cosmos_plv.provenance import repo_relative
    recs = {}
    for run_dir in sorted(p for p in Path(pool).iterdir() if p.is_dir()):
        m = re.match(r"paper_dp_(?P<task>[a-z]+)_abs_ph_seed(?P<seed>\d+)_", run_dir.name)
        if not m or not (run_dir / "checkpoints").is_dir():
            continue
        logs = _read_logs(run_dir)
        for ck in sorted((run_dir / "checkpoints").glob("epoch=*.ckpt")):
            mm = CKPT_RE.search(ck.name)
            if not mm:
                continue
            ep = int(mm.group("epoch"))
            row = logs.get(ep, {})
            cid = f"{m.group('task')}_s{int(m.group('seed'))}_e{ep:04d}"
            recs[cid] = SimpleNamespace(checkpoint_id=cid, run_dir=repo_relative(run_dir), epoch=ep,
                                        train_rollout_success=float(row.get("test/mean_score", float(mm.group("score")))),
                                        val_loss=row.get("val_loss"), train_loss=row.get("train_loss"),
                                        train_action_mse_error=row.get("train_action_mse_error"))
    return recs


def checkpoint_table(E, nebius_path, pool):
    ev = E[E.split == "hackathon_rollouts_eval"]
    g = ev.groupby("checkpoint_id")
    C = g.agg(task=("task", "first"), run=("run", "first"), epoch=("epoch", "first"), k=("success", "size"),
              k_success=("success", "mean"), ov_mean=("sim_replan_overlap_l1", "mean"),
              jump_mean=("sim_boundary_jump_l1", "mean"), ov_grip=("ov_grip_mean", "mean"),
              ov_pre=("ov_all_pre", "mean"), grip_flip_rate=("grip_flip_rate_per100", "mean"),
              exec_smooth=("exec_smoothness", "mean")).reset_index()
    for ft in ("no_grasp", "dropped", "other", "not_placed"):
        C[f"sim_rate_{ft}"] = C.checkpoint_id.map(ev.groupby("checkpoint_id").failure_type.apply(lambda s: (s == ft).mean()))
    C["sim_fail_rate"] = 1 - C.k_success
    nf = (ev[ev.failure_type != "none"].groupby("checkpoint_id").failure_type.apply(lambda s: (s == "dropped").mean()))
    C["sim_dropped_share_of_failures"] = C.checkpoint_id.map(nf)
    if Path(nebius_path).is_file():
        nb = pd.DataFrame([json.loads(l) for l in open(nebius_path)])
        nb = nb[nb.scored.fillna(False).astype(bool)]
        ng = nb.groupby("checkpoint_id")
        C["neb_n"] = C.checkpoint_id.map(ng.size())
        C["neb_fail_rate"] = C.checkpoint_id.map(ng.cosmos_task_success.apply(lambda s: 1 - s.astype(bool).mean()))
        for ft in ("no_grasp", "dropped", "not_placed", "other"):
            C[f"neb_rate_{ft}"] = C.checkpoint_id.map(ng.cosmos_failure_type.apply(lambda s: (s == ft).mean()))
    recs = load_checkpoint_records(pool)
    C["train_rollout_success"] = C.checkpoint_id.map(lambda c: recs[c].train_rollout_success if c in recs else np.nan)
    C["val_loss"] = C.checkpoint_id.map(lambda c: recs[c].val_loss if c in recs else np.nan)
    C["train_action_mse_error"] = C.checkpoint_id.map(lambda c: recs[c].train_action_mse_error if c in recs else np.nan)
    C["train_loss"] = C.checkpoint_id.map(lambda c: recs[c].train_loss if c in recs else np.nan)
    # training-time per-episode outcomes (seeds 100000..100049): split halves for a consistency check
    for c in C.checkpoint_id:
        if c not in recs:
            continue
        rd = ROOT / recs[c].run_dir
        row = None
        for line in (rd / "logs.json.txt").read_text().splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("epoch") == recs[c].epoch and "test/mean_score" in j:
                row = j
        if row:
            a = [row.get(f"test/sim_max_reward_{s}") for s in range(100000, 100025)]
            b = [row.get(f"test/sim_max_reward_{s}") for s in range(100025, 100050)]
            C.loc[C.checkpoint_id == c, "train_success_seedsA"] = np.mean([x for x in a if x is not None]) if any(x is not None for x in a) else np.nan
            C.loc[C.checkpoint_id == c, "train_success_seedsB"] = np.mean([x for x in b if x is not None]) if any(x is not None for x in b) else np.nan
    return C.sort_values(["task", "run", "epoch"]).reset_index(drop=True)


def ckpt_correlations(C, n_boot):
    targets = ["sim_fail_rate", "sim_rate_no_grasp", "sim_rate_dropped", "sim_dropped_share_of_failures"]
    preds = ["epoch", "train_rollout_success", "val_loss", "train_action_mse_error", "ov_mean", "jump_mean",
             "ov_grip", "ov_pre", "grip_flip_rate", "exec_smooth", "neb_fail_rate", "neb_rate_no_grasp", "neb_rate_dropped",
             "neb_rate_not_placed"]
    out = []
    for coh, D in {"all": C, "trained_epoch_gt0": C[C.epoch > 0]}.items():
        for task, T in D.groupby("task"):
            T = T.reset_index(drop=True)
            for tg in targets:
                for p in preds:
                    if p not in T or tg not in T:
                        continue
                    m = np.isfinite(T[p].astype(float)) & np.isfinite(T[tg].astype(float))
                    x, y = T.loc[m, p].values.astype(float), T.loc[m, tg].values.astype(float)
                    if len(x) < 4:
                        continue
                    rho = spearman(x, y)
                    rng = np.random.default_rng(0)
                    bs = []
                    for _ in range(n_boot):
                        ix = rng.integers(0, len(x), len(x))
                        v = spearman(x[ix], y[ix])
                        if np.isfinite(v):
                            bs.append(v)
                    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 50 else [float("nan")] * 2
                    out.append({"cohort": coh, "task": task, "target": tg, "predictor": p, "n_checkpoints": int(len(x)),
                                "spearman": rho, "ci_lo": ci[0], "ci_hi": ci[1]})
    return pd.DataFrame(out)


def make_figure(eta_res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return str(exc)
    panels = [k for k in eta_res if eta_res[k].get("by_col")]
    cols = ["ov_all", "ov_grip", "ov_pos", "jump_all"]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.4), sharey=True)
    for ax, col in zip(axes, cols):
        for name in panels:
            bc = eta_res[name]["by_col"].get(col)
            if not bc:
                continue
            ks = sorted(int(k) for k in bc["lag_mean_lognorm"])
            ys = [bc["lag_mean_lognorm"][k] for k in ks]
            ax.plot([k * 0.4 for k in ks], [np.nan if y is None else y for y in ys], marker="o", ms=3,
                    label=f"{name} (n={eta_res[name]['n_events']})")
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(col)
        ax.set_xlabel("time from event (s, chunk bins of 0.4 s)")
    axes[0].set_ylabel("log disagreement - episode median")
    axes[-1].legend(fontsize=7)
    fig.suptitle("EXPLORATORY: event-triggered replanning disagreement (x3_plv_failure_link)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "data/outputs/hackathon_rollouts_eval"))
    ap.add_argument("--dev", default=str(ROOT / "data/outputs/hackathon_rollouts_dev"))
    ap.add_argument("--pool", default=str(ROOT / "data/outputs/hackathon_pool"))
    ap.add_argument("--nebius", default=str(ROOT / "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl"))
    ap.add_argument("--v7", default=str(ROOT / "data/outputs/hackathon_cosmos/eval_local_v7.jsonl"))
    ap.add_argument("--scratch", default=str(ROOT / f"data/outputs/hackathon_explore/{NAME}"))
    ap.add_argument("--out", default=str(ROOT / f"results/hackathon/explore/{NAME}.json"))
    ap.add_argument("--fig", default=str(ROOT / f"docs/hackathon/explore/{NAME}_eta.png"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    a = ap.parse_args()

    eps = load_episodes([a.eval, a.dev])
    scales = task_scales(eps)
    E, series = episode_table(eps, scales)
    out_dir = Path(a.out).parent
    E.to_csv(out_dir / f"{NAME}_episodes.csv", index=False)
    ev = E[E.split == "hackathon_rollouts_eval"]
    succ_steps = E[E.success].first_success_step
    checks = {"max_abs_diff_vs_replan_metrics_from_plans": float(np.nanmax(E.check_overlap_absdiff)),
              "prefix_steps": PREFIX_STEPS, "min_first_success_step_all_splits": int(succ_steps.min()) if len(succ_steps) else None,
              "prefix_precedes_every_success": bool((succ_steps > PREFIX_STEPS).all()),
              "drop_rules_eval": ev[ev.failure_type == "dropped"].drop_rule.value_counts(dropna=False).to_dict(),
              "dropped_without_event": int(((ev.failure_type == "dropped") & (ev.drop_t < 0)).sum())}
    counts = {s: {t: G.failure_type.value_counts().to_dict() for t, G in S.groupby("task")} for s, S in E.groupby("split")}

    A = auroc_table(E, a.n_boot)
    A.to_csv(out_dir / f"{NAME}_auroc.csv", index=False)
    HL = high_low_table(E)

    # temporal
    cols = ["ov_all", "ov_grip", "ov_pos", "ov_rot", "jump_all", "jump_grip"]
    is_eval = E.split == "hackathon_rollouts_eval"
    is_dev = E.split == "hackathon_rollouts_dev"
    eta_res = {
        "eval_drops": eta(E, series, "drop_t", cols, n_perm=a.n_perm, mask=is_eval & (E.failure_type == "dropped")),
        "dev_drops": eta(E, series, "drop_t", cols, n_perm=a.n_perm, mask=is_dev & (E.failure_type == "dropped")),
        "eval_success_release_control": eta(E, series, "release_t", cols, n_perm=a.n_perm, mask=is_eval & E.success),
        "eval_dropped_first_lift": eta(E, series, "first_lift", cols, n_perm=a.n_perm, mask=is_eval & (E.failure_type == "dropped")),
        "eval_success_first_lift": eta(E, series, "first_lift", cols, n_perm=a.n_perm, mask=is_eval & E.success),
    }
    for task in sorted(ev.task.unique()):
        eta_res[f"eval_drops_{task}"] = eta(E, series, "drop_t", ["ov_all", "ov_grip"], n_perm=a.n_perm,
                                            mask=is_eval & (E.failure_type == "dropped") & (E.task == task))
    phase_matched = phase_matched_compare(E, series, a.n_boot)
    vt = vlm_timing(E, a.nebius, a.v7)
    J = vt.pop("_joined", None)
    if J is not None:
        Jd = J[(J.failure_type == "dropped") & (J.neb_drop_t >= 0)]
        if len(Jd):
            E2 = E.merge(J[["checkpoint_id", "env_seed", "neb_drop_t"]], on=["checkpoint_id", "env_seed"], how="left")
            E2["neb_drop_t"] = E2.neb_drop_t.fillna(-1).astype(int)
            eta_res["eval_drops_nemotron_timed"] = eta(E2, series, "neb_drop_t", cols, n_perm=a.n_perm,
                                                       mask=(E2.split == "hackathon_rollouts_eval") & (E2.failure_type == "dropped"))
    dt_res, dt_frames = droptimer_eval(E, Path(a.scratch), a.nebius, a.n_boot)
    De = dt_frames.get(("droptimer_v1", "eval"))
    if De is not None and len(De):
        De = De[De.pred_drop.fillna(False).astype(bool) & De.pred_time_s.notna()].copy()
        De["dt_drop_t"] = np.round(De.pred_time_s.astype(float) * SIM_HZ).astype(int)
        E3 = E.merge(De[["checkpoint_id", "env_seed", "dt_drop_t"]], on=["checkpoint_id", "env_seed"], how="left")
        E3["dt_drop_t"] = E3.dt_drop_t.fillna(-1).astype(int)
        # what a user WITHOUT simulator state would get: every Nemotron-flagged drop, at Nemotron's time stamp
        eta_res["eval_nemotron_flagged_drops_droptimer_timed"] = eta(E3, series, "dt_drop_t", cols, n_perm=a.n_perm,
                                                                    mask=(E3.split == "hackathon_rollouts_eval"))
        eta_res["eval_true_drops_droptimer_timed"] = eta(E3, series, "dt_drop_t", cols, n_perm=a.n_perm,
                                                         mask=(E3.split == "hackathon_rollouts_eval") & (E3.failure_type == "dropped"))
    fig_err = make_figure({k: v for k, v in eta_res.items() if k in ("eval_drops", "dev_drops", "eval_success_release_control", "eval_nemotron_flagged_drops_droptimer_timed")}, a.fig)

    C = checkpoint_table(E, a.nebius, a.pool)
    C.to_csv(out_dir / f"{NAME}_checkpoints.csv", index=False)
    CC = ckpt_correlations(C, a.n_boot)
    CC.to_csv(out_dir / f"{NAME}_ckpt_correlations.csv", index=False)
    nem_vs_sim = {}
    for task, T in C.groupby("task"):
        d = {}
        for s_col, n_col in (("sim_fail_rate", "neb_fail_rate"), ("sim_rate_no_grasp", "neb_rate_no_grasp"),
                             ("sim_rate_dropped", "neb_rate_dropped")):
            if n_col in T:
                m = T[n_col].notna()
                d[f"{s_col}~{n_col}"] = {"n": int(m.sum()), "spearman": spearman(T.loc[m, s_col], T.loc[m, n_col]),
                                         "mean_abs_diff": float((T.loc[m, s_col] - T.loc[m, n_col]).abs().mean()) if m.any() else None}
        nem_vs_sim[task] = d

    res = {"name": NAME, "label": "EXPLORATORY / post-hoc", "generated_utc": utc_now(), "git_commit": git_commit(),
           "inputs": {"eval": a.eval, "dev": a.dev, "pool": a.pool, "nebius": a.nebius, "v7": a.v7,
                      "baseline_metrics_csv_exists": (ROOT / "results/hackathon/baseline_metrics.csv").is_file()},
           "n_episodes": {s: int(len(S)) for s, S in E.groupby("split")},
           "n_eval_checkpoints": int(ev.checkpoint_id.nunique()), "eval_checkpoints": sorted(ev.checkpoint_id.unique()),
           "failure_counts": counts, "checks": checks,
           "action_scale": {t: v.round(5).tolist() for t, v in scales.items()},
           "auroc": A.to_dict("records"), "high_low": HL, "eta": eta_res, "phase_matched_pre_event": phase_matched, "vlm_timing": vt,
           "checkpoints": C.to_dict("records"), "checkpoint_correlations": CC.to_dict("records"),
           "nemotron_vs_sim_composition": nem_vs_sim, "nemotron_droptimer": dt_res, "figure": a.fig, "figure_error": fig_err,
           "side_outputs": [str(out_dir / f"{NAME}_{x}.csv") for x in ("episodes", "auroc", "checkpoints", "ckpt_correlations")]}
    write_json(a.out, res)
    # console digest
    print(json.dumps({"n_episodes": res["n_episodes"], "checks": checks, "failure_counts": counts}, indent=1, default=str))
    key = A[(A.cohort != "dev_all") & A.feature.isin(["sim_replan_overlap_l1", "sim_boundary_jump_l1", "ov_grip_mean", "ov_all_pre", "grip_flip_rate_per100", "ov_all_liftwin", "n_steps"])]
    pd.set_option("display.width", 220)
    print(key[["cohort", "task", "contrast", "feature", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    for k, v in eta_res.items():
        for col, b in v.get("by_col", {}).items():
            if col in ("ov_all", "ov_grip", "jump_all"):
                print(k, col, "n", b["n_events_with_window"], "pre", round(b["pre_stat_obs"], 3), "null", round(b["null_mean"], 3),
                      "p>=", round(b["p_one_sided_obs_ge_null"], 4), "p<=", round(b["p_one_sided_obs_le_null"], 4))


if __name__ == "__main__":
    main()
