#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION of exploration x3_plv_failure_link (EXPLORATORY / post-hoc).

CPU only, no model or API calls.  Re-uses the explorer's own feature code (x3_plv_failure_link.py) so that the
per-boundary disagreement series are identical, then runs checks the original did not run:

  V0  reproduction: compare a fresh re-run JSON with the published JSON
  V1  headline contrast (can, trained, dropped vs no_grasp): cluster structure, leave-one-checkpoint-out,
      within-checkpoint label permutation, pre-failure windows; dev exact Mann-Whitney p
  V2  temporal: share of the random-time null that lies AFTER the drop (bias), a pre-drop-only null, and a
      lift-offset-matched control (successes of the same checkpoint at the same delay after their first lift)
  V3  Nemotron drop timer: confusion inside failed (400-step) episodes only, outcome-free trivial baselines
      ("every horizon-length clip is a drop"), exact McNemar vs the v2 diagnostic, clip-duration leak,
      checkpoint drop-rate correlation decomposed into true- and false-positive parts
  V4  checkpoint level (R4): disagreement without post-failure tails, partial Spearman given epoch,
      what the "dropped share of failures" variable actually contains
  V5  Cosmos v7 keyframe "held -> not held" transition: rate in successes / no_grasp (specificity)
  V6  gripper flips in a window that precedes every first lift
  V7  drop-timer parser robustness (JSON repair vs field regex)
  V8  video frame / sim step alignment
  V9  can drops: executed gripper-open command near the drop time
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import x3_plv_failure_link as X  # noqa: E402
from x3_nemotron_drop_timer import parse as dt_parse  # noqa: E402
from cosmos_plv.ranking import spearman  # noqa: E402
from cosmos_plv.provenance import utc_now, write_json  # noqa: E402

NAME = "x3_plv_failure_link_verification"
EVAL = "hackathon_rollouts_eval"
DEV = "hackathon_rollouts_dev"


def wilson(k, n):
    return X.wilson(k, n)


def exact_mw_p_le(pos, neg):
    """Exact one-sided P(AUROC <= observed) under random labelling (enumerates all splits; small n only)."""
    allv = np.concatenate([pos, neg])
    n1 = len(pos)
    obs = X.auroc(pos, neg)
    cnt = tot = 0
    for comb in itertools.combinations(range(len(allv)), n1):
        m = np.zeros(len(allv), bool)
        m[list(comb)] = True
        a = X.auroc(allv[m], allv[~m])
        cnt += a <= obs + 1e-12
        tot += 1
    return obs, cnt / tot, tot


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def partial_spearman(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ckpt(fn, n, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        v = fn(ix)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 50 else [float("nan")] * 2


def win_mean(pb, col, te, lags=(-3, -1)):
    li = np.floor((pb.b.values - int(te)) / 8).astype(int)
    w = (li >= lags[0]) & (li <= lags[1])
    if w.sum() != lags[1] - lags[0] + 1:
        return float("nan")
    return float(pb[col].values[w].mean())


# ----------------------------------------------------------------------------- V0
def v0_reproduction(orig, rerun):
    o, r = json.loads(Path(orig).read_text()), json.loads(Path(rerun).read_text())
    A, B = pd.DataFrame(o["auroc"]), pd.DataFrame(r["auroc"])
    m = A.merge(B, on=["cohort", "task", "contrast", "feature"], suffixes=("_o", "_r"))
    diffs = {c: float(np.nanmax((m[c + "_o"] - m[c + "_r"]).abs().values))
             for c in ("auroc", "ci_lo", "ci_hi", "within_ckpt_auroc", "within_ckpt_ci_lo", "within_ckpt_ci_hi")}
    CO, CR = pd.DataFrame(o["checkpoint_correlations"]), pd.DataFrame(r["checkpoint_correlations"])
    mm = CO.merge(CR, on=["cohort", "task", "target", "predictor"], suffixes=("_o", "_r"))
    diffs["ckpt_spearman"] = float(np.nanmax((mm.spearman_o - mm.spearman_r).abs()))
    dto, dtr = o["nemotron_droptimer"]["droptimer_v1_eval"]["all"], r["nemotron_droptimer"]["droptimer_v1_eval"]["all"]
    return {"orig_generated_utc": o["generated_utc"], "rerun_generated_utc": r["generated_utc"],
            "n_episodes_orig": o["n_episodes"], "n_episodes_rerun": r["n_episodes"],
            "same_eval_checkpoints": o["eval_checkpoints"] == r["eval_checkpoints"],
            "n_auroc_rows": [len(A), len(B), len(m)], "max_abs_diff": diffs,
            "droptimer_eval_confusion_orig": {k: dto[k] for k in ("tp", "fn", "fp", "tn")},
            "droptimer_eval_confusion_rerun": {k: dtr[k] for k in ("tp", "fn", "fp", "tn")}}


# ----------------------------------------------------------------------------- V1
def v1_headline(E, n_boot, n_perm):
    out = {}
    T = E[(E.split == EVAL) & (E.epoch > 0) & (E.task == "can") & E.failure_type.isin(["dropped", "no_grasp"])].copy()
    T["y"] = (T.failure_type == "dropped").astype(int)
    out["n_by_type"] = T.failure_type.value_counts().to_dict()
    out["checkpoints_by_type"] = {ft: G.checkpoint_id.value_counts().to_dict() for ft, G in T.groupby("failure_type")}
    out["episode_length_by_type"] = T.groupby("failure_type").n_steps.agg(["min", "max"]).to_dict("index")
    feats = ["ov_grip_mean", "sim_replan_overlap_l1", "ov_pos_mean", "ov_grip_pre", "ov_all_pre", "grip_flips_pre"]
    rng = np.random.default_rng(1)
    res = {}
    for f in feats:
        x, y, cl = T[f].values.astype(float), T.y.values, T.checkpoint_id.values
        a = X.auroc(x[y == 1], x[y == 0])
        ci, _ = X.clustered_boot_auroc(x, y, cl, n_boot=n_boot)
        sa, npairs, nck = X.strat_auroc(x, y, cl)
        # leave-one-checkpoint-out pooled AUROC
        loco = {}
        for k in np.unique(cl):
            m = cl != k
            loco[k] = X.auroc(x[m & (y == 1)], x[m & (y == 0)])
        # within-checkpoint label permutation (only checkpoints containing both types carry information)
        perm = []
        for _ in range(n_perm):
            yp = y.copy()
            for k in np.unique(cl):
                ix = np.nonzero(cl == k)[0]
                yp[ix] = rng.permutation(y[ix])
            perm.append(X.strat_auroc(x, yp, cl)[0])
        perm = np.asarray(perm)
        res[f] = {"auroc_pooled": a, "cluster_ci": ci, "within_ckpt_auroc": sa, "within_ckpt_pairs": npairs,
                  "within_ckpt_n_ckpts": nck,
                  "within_ckpt_perm_p_le": float((1 + (perm <= sa + 1e-12).sum()) / (1 + n_perm)),
                  "loco_min": float(np.nanmin(list(loco.values()))), "loco_max": float(np.nanmax(list(loco.values()))),
                  "loco": {k: float(v) for k, v in loco.items()}}
    out["features"] = res
    # dev replication (exact one-sided p, 3 vs 9 -> 220 labellings)
    D = E[(E.split == DEV) & (E.task == "can") & E.failure_type.isin(["dropped", "no_grasp"])]
    dv = {}
    for f in ("ov_grip_mean", "ov_grip_pre", "grip_flips_pre"):
        pos, neg = D[D.failure_type == "dropped"][f].values.astype(float), D[D.failure_type == "no_grasp"][f].values.astype(float)
        a, p, tot = exact_mw_p_le(pos, neg)
        dv[f] = {"auroc": a, "exact_p_auroc_le_obs": p, "n_labellings": tot, "n_dropped": len(pos), "n_no_grasp": len(neg)}
    out["dev"] = {"checkpoints_by_type": {ft: G.checkpoint_id.value_counts().to_dict() for ft, G in D.groupby("failure_type")},
                  "features": dv}
    return out


# ----------------------------------------------------------------------------- V2
def v2_temporal(E, series, n_perm):
    out = {}
    ev = E[(E.split == EVAL)]
    drops = ev[(ev.failure_type == "dropped") & (ev.drop_t >= 0)]
    # (a) how much of the original random-time null lies after the drop
    frac_after = []
    for r in drops.itertuples():
        pb = series[(r.checkpoint_id, r.env_seed)]
        lo, hi = int(pb.b.min()) + 24, int(pb.b.max()) + 8
        rng_te = np.arange(lo, max(lo, hi) + 1)
        frac_after.append(float((rng_te > r.drop_t).mean()))
    out["null_share_after_drop"] = {"mean": float(np.mean(frac_after)), "min": float(np.min(frac_after)),
                                    "n": len(frac_after)}
    # (b) pre-drop-only null: random te' <= drop_t (window fully before the drop), raw values, within episode
    rng = np.random.default_rng(2)
    pre_only = {}
    for task in ("all", "can", "square"):
        Dt = drops if task == "all" else drops[drops.task == task]
        for col in ("ov_all", "ov_grip", "ov_pos"):
            obs, draws, used = [], [], 0
            for r in Dt.itertuples():
                pb = series[(r.checkpoint_id, r.env_seed)]
                v = np.log(pb[col].values + 1e-6)
                pbl = pb.assign(**{col: v})
                o = win_mean(pbl, col, r.drop_t)
                cands = [te for te in range(int(pb.b.min()) + 24, int(r.drop_t) + 1)]
                vals = np.array([win_mean(pbl, col, te) for te in cands])
                vals = vals[np.isfinite(vals)]
                if not np.isfinite(o) or len(vals) < 2:
                    continue
                used += 1
                obs.append(o - vals.mean())
                draws.append(vals[rng.integers(0, len(vals), n_perm)] - vals.mean())
            if not obs:
                continue
            obs_m = float(np.mean(obs))
            null = np.mean(np.vstack(draws), axis=0)
            pre_only[f"{task}:{col}"] = {"n_events": used, "obs_minus_episode_pre_mean": obs_m,
                                         "null_mean": float(null.mean()),
                                         "p_one_sided_obs_ge_null": float((1 + (null >= obs_m).sum()) / (1 + n_perm))}
    out["pre_drop_only_null"] = pre_only
    # (c) lift-offset-matched control: success j of the SAME checkpoint, evaluated at first_lift_j + (drop_t - first_lift_i)
    lm = {}
    tr = ev[ev.epoch > 0]
    for task in ("can", "square"):
        Dt = tr[(tr.task == task) & (tr.failure_type == "dropped") & (tr.drop_t >= 0) & (tr.first_lift >= 0)]
        St = tr[(tr.task == task) & tr.success & (tr.first_lift >= 0)]
        for lags, lname in (((-3, -1), "pre_lags_-3..-1"), ((0, 0), "at_drop_lag_0"), ((1, 3), "post_lags_+1..+3")):
            for col in ("ov_all", "ov_grip", "ov_pos"):
                conc, pairs, per_ck = 0.0, 0, {}
                n_drop_used = 0
                for r in Dt.itertuples():
                    d_off = int(r.drop_t - r.first_lift)
                    vi = win_mean(series[(r.checkpoint_id, r.env_seed)], col, r.drop_t, lags)
                    if not np.isfinite(vi):
                        continue
                    n_drop_used += 1
                    for s in St[St.checkpoint_id == r.checkpoint_id].itertuples():
                        te = int(s.first_lift + d_off)
                        if s.release_t >= 0 and te + 8 * lags[1] + 8 > s.release_t:  # control must still be holding
                            continue
                        vj = win_mean(series[(s.checkpoint_id, s.env_seed)], col, te, lags)
                        if not np.isfinite(vj):
                            continue
                        c = float(vi > vj) + 0.5 * float(vi == vj)
                        conc += c
                        pairs += 1
                        a, b = per_ck.get(r.checkpoint_id, (0.0, 0))
                        per_ck[r.checkpoint_id] = (a + c, b + 1)
                keys = list(per_ck)
                if pairs:
                    cc = np.array([per_ck[k][0] for k in keys]); pp = np.array([per_ck[k][1] for k in keys])
                    rb = np.random.default_rng(3)
                    pick = rb.integers(0, len(keys), (2000, len(keys)))
                    bv = cc[pick].sum(1) / np.maximum(pp[pick].sum(1), 1)
                    ci = [float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))] if len(keys) >= 3 else [float("nan")] * 2
                else:
                    ci = [float("nan")] * 2
                lm[f"{task}:{lname}:{col}"] = {"n_drops": n_drop_used, "n_pairs": pairs, "n_ckpts": len(keys),
                                               "within_ckpt_auroc_drop_gt_matched_success": conc / pairs if pairs else float("nan"),
                                               "ckpt_boot_ci": ci}
    out["lift_offset_matched_control"] = lm
    out["can_drop_offsets_steps"] = sorted((tr[(tr.task == "can") & (tr.failure_type == "dropped")].drop_t
                                            - tr[(tr.task == "can") & (tr.failure_type == "dropped")].first_lift).astype(int).tolist())
    return out


# ----------------------------------------------------------------------------- V3
def confusion(t, p):
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    tp, fn, fp, tn = int((t & p).sum()), int((t & ~p).sum()), int((~t & p).sum()), int((~t & ~p).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sensitivity": sens, "sens_wilson": wilson(tp, tp + fn),
            "specificity": spec, "spec_wilson": wilson(tn, tn + fp), "balanced_accuracy": (sens + spec) / 2,
            "ppv": tp / (tp + fp) if tp + fp else float("nan"), "ppv_wilson": wilson(tp, tp + fp)}


def v3_droptimer(E, scratch, nebius_v2, n_boot):
    out = {}
    Ei = E.set_index(["checkpoint_id", "env_seed"])
    rows = []
    for line in (scratch / "nemotron_droptimer_eval.jsonl").read_text().splitlines():
        r = json.loads(line)
        pobj, st = dt_parse(r.get("raw_text", ""))
        key = (r["checkpoint_id"], int(r["env_seed"]))
        e = Ei.loc[key]
        rows.append({"checkpoint_id": key[0], "env_seed": key[1], "task": e.task, "epoch": int(e.epoch),
                     "failure_type": e.failure_type, "truth_drop": e.failure_type == "dropped",
                     "horizon_clip": int(e.n_steps) >= 400, "dur": r["video_duration_s"], "parse": st,
                     "pred_drop": bool(pobj["left_gripper_before_goal"] and not pobj["task_success"]),
                     "pred_success": bool(pobj["task_success"]), "pred_left": bool(pobj["left_gripper_before_goal"])})
    D = pd.DataFrame(rows).drop_duplicates(["checkpoint_id", "env_seed"], keep="last")
    out["n"] = int(len(D))
    out["all_reproduced"] = confusion(D.truth_drop, D.pred_drop)
    F = D[D.horizon_clip]
    out["failed_episodes_only"] = {"n": int(len(F)), "by_type": F.failure_type.value_counts().to_dict(),
                                   "droptimer": confusion(F.truth_drop, F.pred_drop),
                                   "by_task": {t: confusion(G.truth_drop, G.pred_drop) for t, G in F.groupby("task")}}
    out["successes_only"] = {"n": int((~D.horizon_clip).sum()), "pred_drop_k": int(D[~D.horizon_clip].pred_drop.sum())}
    # outcome-free trivial baseline: a clip that runs to the 400-step horizon (20 s) is called a drop
    out["baseline_horizon_clip_is_drop"] = {"all": confusion(D.truth_drop, D.horizon_clip),
                                            "by_task": {t: confusion(G.truth_drop, G.horizon_clip) for t, G in D.groupby("task")}}
    # duration leak: the prompt states the clip duration; success predictions vs clip length
    out["duration_vs_pred_success"] = {
        "horizon_clips": {"n": int(D.horizon_clip.sum()), "pred_success_k": int(D[D.horizon_clip].pred_success.sum()),
                          "truth_success_k": 0},
        "short_clips": {"n": int((~D.horizon_clip).sum()), "pred_success_k": int(D[~D.horizon_clip].pred_success.sum())},
        "max_success_clip_s": float(D[~D.horizon_clip].dur.max()), "min_failure_clip_s": float(D[D.horizon_clip].dur.min()),
        "pred_left_on_successes_k": int(D[~D.horizon_clip].pred_left.sum())}
    # paired comparison with v2 on common episodes
    nb = pd.DataFrame([json.loads(l) for l in open(nebius_v2)])
    nb = nb[nb.scored.fillna(False).astype(bool)].copy()
    nb["v2_pred_drop"] = (nb.cosmos_failure_type == "dropped") | (nb.cosmos_object_dropped.fillna(False).astype(bool) & ~nb.cosmos_task_success.astype(bool))
    J = D.merge(nb[["checkpoint_id", "env_seed", "v2_pred_drop"]], on=["checkpoint_id", "env_seed"], how="inner")
    comp = {"n_common": int(len(J))}
    for name, S in (("all", J), ("failed_episodes_only", J[J.horizon_clip])):
        c = {"n": int(len(S)), "droptimer": confusion(S.truth_drop, S.pred_drop), "v2": confusion(S.truth_drop, S.v2_pred_drop),
             "horizon_baseline": confusion(S.truth_drop, S.horizon_clip)}
        dd = S[S.truth_drop]
        b, cc = int((dd.pred_drop & ~dd.v2_pred_drop).sum()), int((~dd.pred_drop & dd.v2_pred_drop).sum())
        c["mcnemar_on_drops"] = {"droptimer_only": b, "v2_only": cc, "exact_two_sided_p": exact_mcnemar(b, cc)}
        nn = S[~S.truth_drop]
        b2, c2 = int((nn.pred_drop & ~nn.v2_pred_drop).sum()), int((~nn.pred_drop & nn.v2_pred_drop).sum())
        c["mcnemar_on_non_drops_false_alarms"] = {"droptimer_only_fp": b2, "v2_only_fp": c2, "exact_two_sided_p": exact_mcnemar(b2, c2)}
        comp[name] = c
    out["paired_vs_v2"] = comp
    # checkpoint level: decompose the predicted drop rate into TP and FP parts; compare with the failure-rate baseline
    g = D.groupby("checkpoint_id")
    C = pd.DataFrame({"task": g.task.first(), "epoch": g.epoch.first(), "sim_drop": g.truth_drop.mean(),
                      "pred_drop": g.pred_drop.mean(),
                      "tp_rate": g.apply(lambda s: (s.truth_drop & s.pred_drop).mean()),
                      "fp_rate": g.apply(lambda s: (~s.truth_drop & s.pred_drop).mean()),
                      "fail_rate": g.horizon_clip.mean(),
                      "sim_no_grasp": g.failure_type.apply(lambda s: (s == "no_grasp").mean())}).reset_index()
    ck = {}
    for task, T in C.groupby("task"):
        T = T.reset_index(drop=True)
        d = {"n_checkpoints": int(len(T)), "rows": T.round(3).to_dict("records")}
        for p in ("pred_drop", "fail_rate", "tp_rate", "fp_rate", "epoch"):
            x, y = T[p].values.astype(float), T.sim_drop.values.astype(float)
            d[f"spearman_{p}_vs_sim_drop"] = {"rho": spearman(x, y),
                                              "ci": boot_ckpt(lambda ix: spearman(x[ix], y[ix]), len(x), n_boot)}
        x, y = T.pred_drop.values.astype(float), T.sim_no_grasp.values.astype(float)
        d["spearman_pred_drop_vs_sim_no_grasp"] = spearman(x, y)
        d["n_ckpt_pred_drop_zero"] = int((T.pred_drop == 0).sum())
        d["n_ckpt_sim_drop_zero"] = int((T.sim_drop == 0).sum())
        ck[task] = d
    out["checkpoint_level"] = ck
    return out


# ----------------------------------------------------------------------------- V4
def v4_checkpoint(E, n_boot):
    ev = E[(E.split == EVAL) & (E.epoch > 0) & (E.task == "can")]
    g = ev.groupby("checkpoint_id")
    C = pd.DataFrame({"epoch": g.epoch.first(), "fail_rate": 1 - g.success.mean(),
                      "ov_mean": g.sim_replan_overlap_l1.mean(), "ov_pre80": g.ov_all_pre.mean(),
                      "ov_grip_pre80": g.ov_grip_pre.mean(),
                      "ov_success_eps_only": g.apply(lambda s: s[s.success].sim_replan_overlap_l1.mean()),
                      "ov_prelift": g.ov_all_prelift.mean(),
                      "n_fail": g.success.apply(lambda s: int((~s).sum())),
                      "n_drop": g.failure_type.apply(lambda s: int((s == "dropped").sum())),
                      "n_no_grasp": g.failure_type.apply(lambda s: int((s == "no_grasp").sum()))}).reset_index()
    C["dropped_share"] = np.where(C.n_fail > 0, C.n_drop / C.n_fail.replace(0, np.nan), np.nan)
    out = {"rows": C.round(4).to_dict("records")}
    res = {}
    for p in ("ov_mean", "ov_pre80", "ov_grip_pre80", "ov_success_eps_only", "ov_prelift", "epoch"):
        x, y = C[p].values.astype(float), C.fail_rate.values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        res[f"{p}~fail_rate"] = {"n": int(m.sum()), "rho": spearman(x, y), "ci": boot_ckpt(lambda ix: spearman(x[ix], y[ix]), len(x), n_boot)}
    for p in ("ov_mean", "ov_pre80", "ov_success_eps_only"):
        x, y, z = C[p].values.astype(float), C.fail_rate.values.astype(float), C.epoch.values.astype(float)
        m = np.isfinite(x)
        res[f"partial_{p}~fail_rate|epoch"] = {"n": int(m.sum()), "rho": partial_spearman(x[m], y[m], z[m]),
                                               "ci": boot_ckpt(lambda ix: partial_spearman(x[m][ix], y[m][ix], z[m][ix]), int(m.sum()), n_boot)}
    S = C[C.n_fail > 0].reset_index(drop=True)
    x, y, z = S.ov_mean.values, S.dropped_share.values, S.epoch.values
    res["ov_mean~dropped_share"] = {"n": int(len(S)), "rho": spearman(x, y), "ci": boot_ckpt(lambda ix: spearman(x[ix], y[ix]), len(x), n_boot)}
    res["partial_ov_mean~dropped_share|epoch"] = {"rho": partial_spearman(x, y, z),
                                                  "ci": boot_ckpt(lambda ix: partial_spearman(x[ix], y[ix], z[ix]), len(x), n_boot)}
    res["dropped_share_composition"] = {"n_ckpts_with_failures": int(len(S)),
                                        "n_ckpts_share_1_from_single_failure": int(((S.dropped_share == 1) & (S.n_fail == 1)).sum()),
                                        "n_ckpts_with_any_no_grasp": int((S.n_no_grasp > 0).sum()),
                                        "ckpts_with_no_grasp": S[S.n_no_grasp > 0].checkpoint_id.tolist()}
    out["correlations"] = res
    return out


# ----------------------------------------------------------------------------- V5
def v5_v7(E, v7_path):
    rows = [json.loads(l) for l in open(v7_path)]
    rec = []
    for r in rows:
        tr = r.get("track") or {}
        held = [f.get("p_held_by_gripper") for f in (tr.get("frames") or [])]
        seen, trans = False, False
        for hh in held:
            if hh is True:
                seen = True
            elif seen and hh is False:
                trans = True
                break
        rec.append({"checkpoint_id": r["checkpoint_id"], "env_seed": int(r["env_seed"]), "v7_transition": trans, "v7_any_held": seen})
    V = E[E.split == EVAL].merge(pd.DataFrame(rec), on=["checkpoint_id", "env_seed"], how="inner")
    out = {"n_joined": int(len(V)), "n_checkpoints": int(V.checkpoint_id.nunique())}
    for coh, S in (("all", V), ("trained_epoch_gt0", V[V.epoch > 0])):
        d = {}
        for (task, ft), G in S.groupby(["task", "failure_type"]):
            k = int(G.v7_transition.sum())
            d[f"{task}:{ft}"] = {"n": int(len(G)), "held_then_not_held_k": k, "rate": k / len(G), "wilson": wilson(k, len(G))}
        out[coh] = d
    return out


# ----------------------------------------------------------------------------- V6
def v6_early_flips(E, eps_by_key, n_boot, window=40):
    ev = E[(E.split == EVAL) & (E.epoch > 0)].copy()
    ev["flips_w"] = [X.grip_flips(eps_by_key[(r.checkpoint_id, r.env_seed)]["action"][:window, 6]) for r in ev.itertuples()]
    out = {"window_steps": window,
           "n_first_lift_before_window": int(((ev.first_lift >= 0) & (ev.first_lift < window)).sum()),
           "min_first_lift_trained_eval": int(ev[ev.first_lift >= 0].first_lift.min())}
    for task, T in ev.groupby("task"):
        x, y, cl = T.flips_w.values.astype(float), (~T.success).values.astype(int), T.checkpoint_id.values
        a = X.auroc(x[y == 1], x[y == 0])
        ci, _ = X.clustered_boot_auroc(x, y, cl, n_boot=n_boot)
        sa, npairs, nck = X.strat_auroc(x, y, cl)
        x80 = T.grip_flips_pre.values.astype(float)
        out[task] = {"failure_vs_success_auroc": a, "ci": ci, "within_ckpt": sa, "within_ckpt_ci": X.clustered_boot_strat(x, y, cl, n_boot),
                     "flip_counts_success": T[T.success].flips_w.value_counts().sort_index().to_dict(),
                     "flip_counts_failure": T[~T.success].flips_w.value_counts().sort_index().to_dict(),
                     "reference_first80_auroc": X.auroc(x80[y == 1], x80[y == 0])}
    return out


# ----------------------------------------------------------------------------- V7
def v7_parser(scratch):
    out = {}
    for fn in ("nemotron_droptimer_eval.jsonl", "nemotron_droptimer_dev.jsonl", "nemotron_droptimer_v2_dev.jsonl"):
        n = agree = repaired = 0
        multi = 0
        for line in (scratch / fn).read_text().splitlines():
            r = json.loads(line)
            raw = r.get("raw_text", "")
            p1, _ = dt_parse(raw)
            fixed = re.sub(r'("explanation"\s*:\s*)(?!")(.*?)(\s*\n?\s*\})', lambda m: m.group(1) + json.dumps(m.group(2).strip()) + m.group(3), raw, flags=re.S)
            try:
                s = fixed[fixed.find("{"): fixed.rfind("}") + 1]
                p2 = json.loads(s)
                repaired += 1
            except Exception:
                p2 = None
            multi += raw.count('"left_gripper_before_goal"') > 1
            n += 1
            if p1 and p2:
                same = all(p1.get(k) == p2.get(k) for k in ("left_gripper_before_goal", "task_success", "object_lifted_in_gripper")) and \
                    (p1.get("time_left_gripper_s") == p2.get("time_left_gripper_s") or
                     (p1.get("time_left_gripper_s") is not None and p2.get("time_left_gripper_s") is not None and
                      abs(float(p1["time_left_gripper_s"]) - float(p2["time_left_gripper_s"])) < 1e-9))
                agree += same
        out[fn] = {"n": n, "json_repair_ok": repaired, "agree_with_field_regex": agree, "rows_with_repeated_key": int(multi)}
    return out


# ----------------------------------------------------------------------------- V8
def v8_alignment(eps):
    """Frame k = sim step 2k for steps 0..n_steps-1  =>  video_frames == ceil(n_steps / 2)."""
    bad, n = 0, 0
    rel = {}
    for e in eps:
        m = e["meta"]
        vf, ns = int(m["video_frames"]), int(m["n_steps"])
        n += 1
        key = vf - (ns + 1) // 2
        rel[key] = rel.get(key, 0) + 1
        bad += key != 0
    return {"n_episodes": n, "video_frames_minus_ceil(n_steps/2)": {str(k): v for k, v in rel.items()},
            "n_mismatch": bad, "video_fps_values": sorted({int(e["meta"]["video_fps"]) for e in eps}),
            "droptimer_time_mapping": "4-fps frame k (t=0.25k s) <- source frame round(2.5k) <- sim step 2*round(2.5k); "
                                      "max |t - sim_time| = 0.05 s"}


# ----------------------------------------------------------------------------- V9
def v9_can_drop_gripper(E, eps_by_key):
    ev = E[(E.split == EVAL) & (E.epoch > 0)]
    out = {}
    for task in ("can", "square"):
        Dt = ev[(ev.task == task) & (ev.failure_type == "dropped")]
        rows = []
        for r in Dt.itertuples():
            a = eps_by_key[(r.checkpoint_id, r.env_seed)]["action"][:, 6]
            lo, hi = max(0, int(r.drop_t) - 8), min(len(a), int(r.drop_t) + 2)
            opened = bool((a[lo:hi] < 0).any())
            closed_before = bool((a[max(0, int(r.first_lift) - 4): int(r.first_lift) + 1] > 0).any()) if r.first_lift >= 0 else None
            rows.append({"checkpoint_id": r.checkpoint_id, "env_seed": int(r.env_seed), "drop_t": int(r.drop_t),
                         "first_lift": int(r.first_lift), "open_cmd_in_[t-8,t+1]": opened, "closed_at_lift": closed_before})
        k = sum(x["open_cmd_in_[t-8,t+1]"] for x in rows)
        out[task] = {"n_drops": len(rows), "open_command_near_drop_k": int(k), "wilson": wilson(int(k), len(rows)), "rows": rows}
    return out


def v10_v2(E, nebius_v2, orig):
    nb = pd.DataFrame([json.loads(l) for l in open(nebius_v2)])
    nb = nb[nb.scored.fillna(False).astype(bool)]
    J = E[E.split == EVAL].merge(nb[["checkpoint_id", "env_seed", "cosmos_failure_type", "cosmos_object_dropped", "cosmos_task_success"]],
                                 on=["checkpoint_id", "env_seed"])
    out = {"n_joined": int(len(J)),
           "can_pred_failure_type_counts": J[J.task == "can"].cosmos_failure_type.value_counts(dropna=False).to_dict(),
           "can_true_drops_pred_type": J[(J.task == "can") & (J.failure_type == "dropped")].cosmos_failure_type.value_counts(dropna=False).to_dict(),
           "square_true_drops_pred_type": J[(J.task == "square") & (J.failure_type == "dropped")].cosmos_failure_type.value_counts(dropna=False).to_dict()}
    C = pd.DataFrame(json.loads(Path(orig).read_text())["checkpoints"])
    C = C[(C.task == "can") & C.neb_rate_no_grasp.notna()]
    for coh, T in (("all_incl_epoch0", C), ("trained_epoch_gt0", C[C.epoch > 0])):
        out[f"neb_no_grasp~sim_no_grasp_{coh}"] = {"n": int(len(T)), "rho": spearman(T.neb_rate_no_grasp.values, T.sim_rate_no_grasp.values)}
    return out


def v11_dev_selection(E_snap, E_cur, scratch):
    """Does the dev choice droptimer_v1 > droptimer_v2 (balanced accuracy) survive the upstream label refresh?"""
    out = {}
    for lab, EE in (("snapshot", E_snap), ("current", E_cur)):
        Ei = EE.set_index(["checkpoint_id", "env_seed"])
        d = {}
        for var, fn in (("droptimer_v1", "nemotron_droptimer_dev.jsonl"), ("droptimer_v2", "nemotron_droptimer_v2_dev.jsonl")):
            t, p, hz = [], [], []
            for line in (scratch / fn).read_text().splitlines():
                r = json.loads(line)
                pobj, _ = dt_parse(r.get("raw_text", ""))
                e = Ei.loc[(r["checkpoint_id"], int(r["env_seed"]))]
                t.append(e.failure_type == "dropped")
                p.append(bool(pobj and pobj["left_gripper_before_goal"] and not pobj["task_success"]))
                hz.append(int(e.n_steps) >= 400)
            c = confusion(t, p)
            t, p, hz = np.array(t), np.array(p), np.array(hz)
            cf = confusion(t[hz], p[hz])
            d[var] = {"tp": c["tp"], "fn": c["fn"], "fp": c["fp"], "tn": c["tn"], "balanced_accuracy": c["balanced_accuracy"],
                      "failed_only_balanced_accuracy": cf["balanced_accuracy"], "n_drops": int(t.sum())}
        d["chosen_by_ba"] = max(("droptimer_v1", "droptimer_v2"), key=lambda v: d[v]["balanced_accuracy"])
        out[lab] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=str(ROOT / "results/hackathon/explore/x3_plv_failure_link.json"))
    ap.add_argument("--rerun", default=str(ROOT / f"data/outputs/hackathon_explore/{NAME}/rerun.json"))
    ap.add_argument("--scratch_x3", default=str(ROOT / "data/outputs/hackathon_explore/x3_plv_failure_link"))
    ap.add_argument("--nebius", default=str(ROOT / "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl"))
    ap.add_argument("--v7", default=str(ROOT / "data/outputs/hackathon_cosmos/eval_local_v7.jsonl"))
    ap.add_argument("--out", default=str(ROOT / f"results/hackathon/explore/{NAME}.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--labels", choices=["snapshot", "current"], default="snapshot",
                    help="snapshot = truth labels as analysed by the explorer (its episodes CSV); current = summary.json now")
    a = ap.parse_args()

    res = {"name": NAME, "label": "ADVERSARIAL VERIFICATION, EXPLORATORY / post-hoc", "generated_utc": utc_now(),
           "verifies": "x3_plv_failure_link"}
    res["V0_reproduction"] = v0_reproduction(a.orig, a.rerun) if Path(a.rerun).is_file() else "rerun json missing"
    eps = X.load_episodes([ROOT / "data/outputs/hackathon_rollouts_eval", ROOT / "data/outputs/hackathon_rollouts_dev"])
    # pin to the snapshot the explorer analysed (eval checkpoints listed in its JSON); new data reported separately
    snap = set(json.loads(Path(a.orig).read_text())["eval_checkpoints"])
    scales = X.task_scales([e for e in eps if e["split"] == DEV or e["meta"]["checkpoint_id"] in snap])
    E_all, series = X.episode_table(eps, scales)
    eps_by_key = {(e["meta"]["checkpoint_id"], int(e["meta"]["env_seed"])): e["d"] for e in eps}
    E = E_all[(E_all.split == DEV) | E_all.checkpoint_id.isin(snap)].reset_index(drop=True)
    new_ck = sorted(set(E_all[E_all.split == EVAL].checkpoint_id) - snap)
    res["snapshot"] = {"n_eval_checkpoints_snapshot": len(snap), "new_eval_checkpoints_since": new_ck,
                       "n_episodes_snapshot": E.groupby("split").size().to_dict(),
                       "n_episodes_now": E_all.groupby("split").size().to_dict(),
                       "note": "task-pooled action scale computed on the snapshot's eval episodes (outcome-free), "
                               "as in the explorer's run; new checkpoints are featurised with the same scale"}
    res["n_episodes"] = E.groupby("split").size().to_dict()
    oc = pd.read_csv(ROOT / "results/hackathon/explore/x3_plv_failure_link_episodes.csv")
    # truth labels were refreshed upstream (scripts/hackathon/refresh_truth.py + failure_annotator.py) after the
    # explorer ran: verify against the labels it analysed, report the revised labels as a sensitivity check
    lab_cols = ["failure_type", "success", "drop_t", "drop_rule", "release_t"]
    E_cur = E.copy()
    cmp = E_cur.merge(oc[["checkpoint_id", "env_seed"] + lab_cols], on=["checkpoint_id", "env_seed"], suffixes=("", "_snap"))
    changed = cmp[(cmp.failure_type != cmp.failure_type_snap) | (cmp.success != cmp.success_snap)]
    res["label_revision"] = {"n_changed": int(len(changed)),
                             "changed": changed[["checkpoint_id", "env_seed", "failure_type_snap", "failure_type",
                                                 "success_snap", "success"]].to_dict("records")}
    if a.labels == "snapshot":
        E = E.drop(columns=lab_cols).merge(oc[["checkpoint_id", "env_seed"] + lab_cols], on=["checkpoint_id", "env_seed"], how="left")
        for c in ("drop_t", "release_t"):
            E[c] = E[c].astype(int)
        E["success"] = E.success.astype(bool)
    res["labels_used"] = a.labels
    res["V11_dev_variant_selection"] = v11_dev_selection(E, E_cur, Path(a.scratch_x3))
    if a.labels == "snapshot" and len(changed):
        res["label_revision"]["V3_droptimer_with_current_labels"] = {
            k: v for k, v in v3_droptimer(E_cur, Path(a.scratch_x3), a.nebius, a.n_boot).items()
            if k in ("n", "all_reproduced", "failed_episodes_only", "baseline_horizon_clip_is_drop")}
    mj = E.merge(oc, on=["checkpoint_id", "env_seed"], suffixes=("", "_o"))
    res["snapshot"]["feature_equality_vs_explorer_csv"] = {
        "n_matched": int(len(mj)),
        **{c: float(np.nanmax((mj[c] - mj[c + "_o"]).abs())) for c in
           ("sim_replan_overlap_l1", "ov_grip_mean", "ov_all_pre", "ov_all_prelift", "drop_t", "first_lift", "release_t")}}
    res["V10_v2_baseline_claims"] = v10_v2(E, a.nebius, a.orig)
    res["V1_headline_dropped_vs_no_grasp"] = v1_headline(E, a.n_boot, a.n_perm)
    if new_ck:
        res["V1_with_new_checkpoints"] = v1_headline(E_all, a.n_boot, a.n_perm)
        res["V4_with_new_checkpoints"] = v4_checkpoint(E_all, a.n_boot)
    res["V2_temporal"] = v2_temporal(E, series, a.n_perm)
    res["V3_droptimer"] = v3_droptimer(E, Path(a.scratch_x3), a.nebius, a.n_boot)
    res["V4_checkpoint_level"] = v4_checkpoint(E, a.n_boot)
    res["V5_cosmos_v7_transition_specificity"] = v5_v7(E, a.v7)
    res["V6_early_gripper_flips"] = v6_early_flips(E, eps_by_key, a.n_boot)
    res["V7_parser"] = v7_parser(Path(a.scratch_x3))
    res["V8_alignment"] = v8_alignment(eps)
    res["V9_drop_gripper_open"] = v9_can_drop_gripper(E, eps_by_key)
    write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("V4_checkpoint_level", "V9_drop_gripper_open")}, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
