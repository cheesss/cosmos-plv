#!/usr/bin/env python3
"""Adversarial verification of x2_early_stop (EXPLORATORY, post-hoc). CPU only, no VLM calls.

1. Reproduce x2_early_stop.json on the frozen data snapshot it used (20 eval checkpoints,
   the 9 v7 checkpoints), by monkeypatching the loaders of x2_early_stop_analyze, and diff.
2. Extra checks: cluster (checkpoint) bootstrap CIs, paired differences vs time-only,
   epoch-0 dependence, Nemotron re-score from saved raw_text, clip length / alignment,
   held-flag definition, false-failure calls, free saving from stopping at first success.

Env B:
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x2_early_stop_verify.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))

import x2_early_stop_analyze as A  # noqa: E402
import x2_early_stop_nebius as N  # noqa: E402

ORIG = json.loads((ROOT / "results/hackathon/explore/x2_early_stop.json").read_text())
SCR = ROOT / "data/outputs/hackathon_explore/x2_verify"
SCR.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results/hackathon/explore/x2_early_stop_verification.json"

ORIG_CKPTS = set(ORIG["part1_state"]["dev_to_eval"]["time_only/maxfp0"]["eval_all"]["ranking"]["per_checkpoint"])
ORIG_V7_CKPTS = set(ORIG["part2_cosmos_v7"]["eval_checkpoints"])

_load_split = A.load_split
_load_v7 = A.load_v7


def frozen_load_split(split):
    eps = _load_split(split)
    if split == "eval":
        eps = [e for e in eps if e["checkpoint_id"] in ORIG_CKPTS]
    return eps


def frozen_load_v7(split, eps):
    rows = _load_v7(split, eps)
    if split == "eval":
        rows = [(e, fr) for e, fr in rows if e["checkpoint_id"] in ORIG_V7_CKPTS]
    return rows


def diff(a, b, path="", out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((path + "/" + str(k), "missing_in_" + ("orig" if k not in a else "rerun")))
            else:
                diff(a[k], b[k], path + "/" + str(k), out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"len {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, float) and isinstance(b, float):
        if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > tol:
            out.append((path, f"{a} vs {b}"))
    elif a != b:
        out.append((path, f"{a} vs {b}"))
    return out


# ------------------------------------------------------------------ bootstrap helpers
def boot_ci(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return [None, None]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def policy_stats(eps, aborts, n_boot=2000, seed=1):
    """Point values + episode bootstrap + checkpoint-cluster bootstrap for saved frac / FA rate."""
    steps = np.array([e["end"] for e in eps], float)
    saved = np.array([(e["end"] - a) if a is not None else 0 for e, a in zip(eps, aborts)], float)
    succ = np.array([e["success"] for e in eps], bool)
    fa = np.array([e["success"] and a is not None for e, a in zip(eps, aborts)], bool)
    ck = np.array([e["checkpoint_id"] for e in eps])
    groups = {c: np.where(ck == c)[0] for c in sorted(set(ck))}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    sf_ep, sf_cl, fa_cl = [], [], []
    n = len(eps)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        sf_ep.append(saved[ix].sum() / steps[ix].sum())
        cix = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        sf_cl.append(saved[cix].sum() / steps[cix].sum())
        fa_cl.append(fa[cix].sum() / max(1, succ[cix].sum()))
    return {"n": n, "n_checkpoints": len(keys), "saved_frac": round(float(saved.sum() / steps.sum()), 4),
            "saved_frac_ci95_episode_boot": boot_ci(sf_ep), "saved_frac_ci95_cluster_boot": boot_ci(sf_cl),
            "false_aborts": int(fa.sum()), "n_successes": int(succ.sum()),
            "false_abort_rate": round(float(fa.sum() / succ.sum()), 4) if succ.sum() else None,
            "false_abort_rate_ci95_wilson": A.wilson(int(fa.sum()), int(succ.sum())),
            "false_abort_rate_ci95_cluster_boot": boot_ci(fa_cl),
            "false_aborts_by_checkpoint": dict(Counter(c for c, x in zip(ck, fa) if x)),
            "true_aborts": int(sum(1 for e, a in zip(eps, aborts) if (not e["success"]) and a is not None)),
            "n_failures": int((~succ).sum())}


def paired_diff(eps, ab1, ab2, n_boot=2000, seed=2):
    """saved_frac(ab1) - saved_frac(ab2) on the same episodes; episode and cluster bootstrap."""
    steps = np.array([e["end"] for e in eps], float)
    s1 = np.array([(e["end"] - a) if a is not None else 0 for e, a in zip(eps, ab1)], float)
    s2 = np.array([(e["end"] - a) if a is not None else 0 for e, a in zip(eps, ab2)], float)
    fa1 = sum(1 for e, a in zip(eps, ab1) if e["success"] and a is not None)
    fa2 = sum(1 for e, a in zip(eps, ab2) if e["success"] and a is not None)
    ck = np.array([e["checkpoint_id"] for e in eps])
    groups = {c: np.where(ck == c)[0] for c in sorted(set(ck))}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    d_ep, d_cl = [], []
    n = len(eps)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        d_ep.append((s1[ix].sum() - s2[ix].sum()) / steps[ix].sum())
        cix = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        d_cl.append((s1[cix].sum() - s2[cix].sum()) / steps[cix].sum())
    return {"n": n, "diff_saved_frac": round(float((s1.sum() - s2.sum()) / steps.sum()), 4),
            "ci95_episode_boot": boot_ci(d_ep), "ci95_cluster_boot": boot_ci(d_cl),
            "false_aborts_1": fa1, "false_aborts_2": fa2}


# ------------------------------------------------------------------ main
def main():
    res = {"label": "verification of x2_early_stop (EXPLORATORY, post-hoc)"}
    # ---- 1. reproduce on the frozen snapshot
    A.load_split = frozen_load_split
    A.load_v7 = frozen_load_v7
    frozen_path = SCR / "rerun_frozen_snapshot.json"
    sys.argv = ["x", "--out", str(frozen_path)]
    A.main()
    rer = json.loads(frozen_path.read_text())
    d = diff(ORIG, rer)
    res["reproduction_frozen_snapshot"] = {"n_differences": len(d), "first_differences": [list(x) for x in d[:40]],
                                           "eval_episodes": rer["data"]["eval_episodes"],
                                           "v7_rows": rer["part2_cosmos_v7"]["n_eval_rows"]}

    dev = A.load_split("dev")
    ev = A.load_split("eval")  # frozen
    for e in ev + dev:
        e.pop("_drop_cache", None)

    # ---- 2. oracle and free saving from stopping at first success (instead of +16 steps)
    tot = sum(e["end"] for e in ev)
    fail_steps = sum(e["end"] for e in ev if not e["success"])
    post = sum(e["end"] - e["fss"] for e in ev if e["success"])
    res["oracle_and_post_success"] = {
        "total_steps": tot, "steps_in_failures": fail_steps, "oracle_frac": round(fail_steps / tot, 4),
        "fraction_of_episodes_failing": round(sum(not e["success"] for e in ev) / len(ev), 4),
        "post_success_steps_recorded": post, "post_success_frac_of_total": round(post / tot, 4),
        "fail_end_values": dict(Counter(e["end"] for e in ev if not e["success"])),
    }

    # ---- 3. rebuild headline Part-1 policies and add cluster CIs / paired diffs / epoch-0 split
    pols = {}
    for fam in ("time_only", "state_combo", "proprio_combo"):
        ab = [None] * len(ev)
        for task in ("can", "square"):
            D = [e for e in dev if e["task"] == task]
            _, rule = A.select(D, fam, 0)
            for i, e in enumerate(ev):
                if e["task"] == task:
                    ab[i] = rule(e)
        pols[f"dev2eval/{fam}"] = ab
        folds = (set(range(200000, 200005)), set(range(200005, 200010)))
        ab = [None] * len(ev)
        for task in ("can", "square"):
            for fi in (0, 1):
                fit = [e for e in ev if e["task"] == task and e["env_seed"] in folds[fi]]
                _, rule = A.select(fit, fam, 0)
                for i, e in enumerate(ev):
                    if e["task"] == task and e["env_seed"] in folds[1 - fi]:
                        ab[i] = rule(e)
        pols[f"crossfit/{fam}"] = ab
    p1 = {}
    for k, ab in pols.items():
        st = policy_stats(ev, ab)
        idx0 = [i for i, e in enumerate(ev) if e["epoch"] > 0]
        st["excluding_epoch0"] = policy_stats([ev[i] for i in idx0], [ab[i] for i in idx0])
        p1[k] = st
    res["part1_policies"] = p1
    res["part1_paired_vs_time_only"] = {}
    for sch in ("dev2eval", "crossfit"):
        for fam in ("state_combo", "proprio_combo"):
            a1, a2 = pols[f"{sch}/{fam}"], pols[f"{sch}/time_only"]
            idx0 = [i for i, e in enumerate(ev) if e["epoch"] > 0]
            res["part1_paired_vs_time_only"][f"{sch}/{fam}"] = {
                "all": paired_diff(ev, a1, a2),
                "excluding_epoch0": paired_diff([ev[i] for i in idx0], [a1[i] for i in idx0], [a2[i] for i in idx0])}
    # noclose at 3 s (the dev->eval proprio rule for square used T1=60)
    nc = {}
    for T in (40, 60, 80):
        for task in ("can", "square"):
            E = [e for e in ev if e["task"] == task]
            ab = [A.r_noclose(T)(e) for e in E]
            nc[f"{task}/T{T}"] = {"aborted": sum(a is not None for a in ab),
                                  "false": sum(1 for e, a in zip(E, ab) if e["success"] and a is not None)}
    res["noclose_counts"] = nc

    # slow successes per checkpoint (who pays the time cut)
    slow = defaultdict(list)
    for e in ev:
        if e["success"]:
            slow[e["checkpoint_id"]].append(e["fss"])
    res["slow_successes"] = {c: {"n_succ": len(v), "n_fss_gt_160": sum(x > 160 for x in v),
                                 "n_fss_gt_180": sum(x > 180 for x in v), "n_fss_gt_240": sum(x > 240 for x in v),
                                 "max_fss": max(v)} for c, v in sorted(slow.items())}

    # ---- 4. Part 2 (Cosmos v7): epoch-0 dependence, paired diff, held-flag definition
    rows_ev = A.load_v7("eval", ev)
    rows_dev = A.load_v7("dev", dev)
    grid = [(T1, uh, T2) for T1 in A.FAIL_KF_STEPS + [None] for uh in (False, True) for T2 in A.GRID_T2]

    def make(T1, use_held, T2):
        return A.v_combo(A.v_snapshot(T1, use_held) if T1 else A._v_off, A.v_time(T2))

    Dc = [(e, fr) for e, fr in rows_dev if e["task"] == "can"]
    params, rule = A._sel_v(Dc, make, grid)
    Es = [e for e, _ in rows_ev]
    ab_vis = [rule(e, fr) for e, fr in rows_ev]
    ab_time = [A.r_time(params[2])(e) for e in Es]
    ab_state_kf = [A.v_combo(A.s_snapshot(params[0]) if params[0] else A._v_off, A.v_time(params[2]))(e, fr) for e, fr in rows_ev]
    ab_state_dense = [A.combo(A.r_nolift(params[0]), A.r_time(params[2]))(e) for e in Es]
    i0 = [i for i, e in enumerate(Es) if e["epoch"] > 0]
    sub = lambda L: [L[i] for i in i0]  # noqa: E731
    p2 = {"chosen_can_params": params, "n_eval": len(Es), "n_eval_epoch_gt0": len(i0),
          "n_epoch0_eval_rows": len(Es) - len(i0),
          "vision": policy_stats(Es, ab_vis), "time_only": policy_stats(Es, ab_time),
          "state_same_keyframe": policy_stats(Es, ab_state_kf), "state_dense_nolift_same_T": policy_stats(Es, ab_state_dense),
          "paired_vision_minus_time": paired_diff(Es, ab_vis, ab_time),
          "excluding_epoch0": {"vision": policy_stats(sub(Es), sub(ab_vis)), "time_only": policy_stats(sub(Es), sub(ab_time)),
                               "paired_vision_minus_time": paired_diff(sub(Es), sub(ab_vis), sub(ab_time))},
          "vision_extra_aborts_by_epoch": dict(Counter(f"epoch{e['epoch']}" for e, a1, a2 in zip(Es, ab_vis, ab_time)
                                                       if a1 is not None and (a2 is None or a1 < a2)))}
    # held flag under alternative truths
    alt = {"truth_grasped_contact": Counter(), "truth_grasped_or_lifted": Counter()}
    for e, frames in rows_ev:
        for fr in frames:
            s = min(fr["step"], e["end"])
            g = bool(e["grasped"][s])
            lifted = bool(s >= A.SETTLE and e["h"][s] >= A.LIFT_M)
            alt["truth_grasped_contact"][(g, fr["held"])] += 1
            alt["truth_grasped_or_lifted"][(g or lifted, fr["held"])] += 1
    p2["held_flag_alt_truth"] = {k: {f"truth_{a}_pred_{b}": v for (a, b), v in sorted(c.items())} for k, c in alt.items()}
    # check keyframe step alignment: failure tracks should be at FAIL_KF_STEPS
    fail_steps = Counter(tuple(fr["step"] for fr in frames) for e, frames in rows_ev if not e["success"])
    p2["failure_keyframe_step_patterns"] = {str(k): v for k, v in fail_steps.items()}
    p2["FAIL_KF_STEPS"] = A.FAIL_KF_STEPS
    res["part2_checks"] = p2

    # ---- 5. Part 3 (Nemotron): re-score from raw_text, clip lengths, false calls, baselines
    rows = A.load_neb()
    mism = []
    for r in rows:
        st, obj = N.parse(r["raw_text"])
        if st != r["parse_status"] or (st == "ok" and (obj.get("will_succeed") != r["will_succeed"] or obj.get("confidence") != r["confidence"])):
            mism.append((r["tag"], st, r["parse_status"]))
    # independent regex parse
    rx_mism = 0
    for r in rows:
        m = re.search(r'"will_succeed"\s*:\s*(true|false)', r["raw_text"] or "")
        c = re.search(r'"confidence"\s*:\s*"(low|medium|high)"', r["raw_text"] or "")
        if not m or not c or (m.group(1) == "true") != r["will_succeed"] or c.group(1) != r["confidence"]:
            rx_mism += 1
    # truth fields vs summary.json
    summ = {}
    for split in ("dev", "eval"):
        for p in sorted(A.SPLIT_ROOT[split].glob("*/summary.json")):
            for ep in json.loads(p.read_text())["episodes"]:
                summ[(ep["checkpoint_id"], int(ep["env_seed"]))] = ep
    truth_mism = sum(1 for r in rows if summ[(r["checkpoint_id"], int(r["env_seed"]))]["success"] != r["truth_success"])
    # evidence log
    ev_tags = []
    for line in (ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl").read_text().splitlines():
        try:
            t = json.loads(line).get("tag", "")
        except json.JSONDecodeError:
            continue
        if t.startswith("explore/x2_early_stop/"):
            ev_tags.append(t)
    # clip frame counts and last-frame alignment
    import imageio.v2 as imageio
    clip_counts = Counter()
    align = []
    for i, r in enumerate(rows):
        rd = imageio.get_reader(str(ROOT / r["clip"]))
        fr = [f for f in rd]
        rd.close()
        clip_counts[(r["cut_step"], len(fr))] += 1
        if i % 15 == 0:  # sample: last clip frame vs source frames k-1, k, k+1
            k = r["cut_step"] // 2
            src = imageio.get_reader(str(ROOT / summ[(r["checkpoint_id"], int(r["env_seed"]))]["video"]))
            sf = {}
            for j, f in enumerate(src):
                if j in (k - 1, k, k + 1):
                    sf[j] = f.astype(float)
                if j > k + 1:
                    break
            src.close()
            last = fr[-1].astype(float)
            dd = {j - k: round(float(np.abs(last - f).mean()), 2) for j, f in sf.items()}
            align.append({"tag": r["tag"], "mad_by_offset": dd, "best_offset": min(dd, key=dd.get)})
    p3 = {"n_rows": len(rows), "parse_mismatch_vs_stored": len(mism), "regex_parse_mismatch": rx_mism,
          "truth_mismatch_vs_summary": truth_mism, "evidence_log_x2_calls": len(ev_tags),
          "evidence_log_unique_tags": len(set(ev_tags)),
          "clip_frames_by_cut": {f"cut{a}_frames{b}": v for (a, b), v in sorted(clip_counts.items())},
          "alignment_sample": align,
          "alignment_best_offset_counts": dict(Counter(x["best_offset"] for x in align))}
    # dev A vs B identical binary predictions?
    devA = {(r["checkpoint_id"], r["env_seed"]): r for r in rows if r["split"] == "dev" and r["variant"] == "A_plain"}
    devB = {(r["checkpoint_id"], r["env_seed"]): r for r in rows if r["split"] == "dev" and r["variant"] == "B_evidence"}
    p3["dev_A_vs_B_same_binary_pred"] = sum(1 for k in devA if k in devB and devA[k]["will_succeed"] == devB[k]["will_succeed"])
    p3["dev_n_pairs"] = len(set(devA) & set(devB))
    p3["dev_B_high_conf_fail"] = Counter(f"truth_success={r['truth_success']}" for r in devB.values()
                                         if r["will_succeed"] is False and r["confidence"] == "high")
    p3["dev_checkpoints"] = sorted({r["checkpoint_id"] for r in devB.values()})
    # rebuild policies
    evr = [r for r in rows if r["split"] == "eval" and r["variant"] == "B_evidence"]
    pred = {(r["checkpoint_id"], r["env_seed"], r["cut_step"]): r for r in evr if r["parse_status"] == "ok"}
    cuts = sorted({r["cut_step"] for r in evr})
    called = {(r["checkpoint_id"], r["env_seed"]) for r in evr}
    E = [e for e in ev if (e["checkpoint_id"], e["env_seed"]) in called]
    t2 = {t: ORIG["part1_state"]["dev_to_eval"]["time_only/maxfp0"]["chosen_on_dev"][t]["params"][0] for t in ("can", "square")}

    def nem(conf_ok, T2=None):
        out = []
        for e in E:
            a = None
            for c in cuts:
                r = pred.get((e["checkpoint_id"], e["env_seed"], c))
                if r is not None and A.running(e, c) and not r["will_succeed"] and r["confidence"] in conf_ok:
                    a = c
                    break
            if a is None and T2 is not None:
                a = A.r_time(T2[e["task"]])(e)
            out.append(a)
        return out

    ab_any, ab_hi = nem(A.CONF_ALL), nem(("high",))
    ab_hi_t, ab_t = nem(("high",), t2), [A.r_time(t2[e["task"]])(e) for e in E]
    ab_any_t = nem(A.CONF_ALL, t2)
    i0 = [i for i, e in enumerate(E) if e["epoch"] > 0]
    sub = lambda L: [L[i] for i in i0]  # noqa: E731
    p3["policies"] = {
        "n_episodes": len(E), "n_epoch_gt0": len(i0),
        "any_conf": policy_stats(E, ab_any), "high_conf": policy_stats(E, ab_hi),
        "time_only_devT2": policy_stats(E, ab_t), "high_conf+time_devT2": policy_stats(E, ab_hi_t),
        "any_conf+time_devT2": policy_stats(E, ab_any_t),
        "paired_high_conf+time_minus_time": paired_diff(E, ab_hi_t, ab_t),
        "paired_any_conf+time_minus_time": paired_diff(E, ab_any_t, ab_t),
        "excluding_epoch0": {
            "high_conf": policy_stats(sub(E), sub(ab_hi)), "any_conf": policy_stats(sub(E), sub(ab_any)),
            "time_only_devT2": policy_stats(sub(E), sub(ab_t)),
            "paired_high_conf+time_minus_time": paired_diff(sub(E), sub(ab_hi_t), sub(ab_t)),
            "paired_any_conf+time_minus_time": paired_diff(sub(E), sub(ab_any_t), sub(ab_t))},
    }
    # time-only baseline matched to <=1 false abort, chosen on the SAME eval episodes (optimistic, like the
    # eval-observed high-conf operating point)
    best = None
    for Tc in range(80, 401, 20):
        for Ts in range(80, 401, 20):
            ab = [A.r_time(Tc if e["task"] == "can" else Ts)(e) for e in E]
            fa = sum(1 for e, a in zip(E, ab) if e["success"] and a is not None)
            sv = sum(e["end"] - a for e, a in zip(E, ab) if a is not None)
            if fa <= 1 and (best is None or sv > best[0]):
                best = (sv, Tc, Ts, ab)
    p3["eval_optimised_time_only_max1FA"] = {"T_can": best[1], "T_square": best[2], **policy_stats(E, best[3])}
    # AUROC cluster bootstrap at 4 s
    r4 = [r for r in evr if r["cut_step"] == 80 and r["parse_status"] == "ok"]
    for lab, rr in (("all", r4), ("excl_epoch0", [r for r in r4 if not r["checkpoint_id"].endswith("_e0000")])):
        y = np.array([int(r["truth_success"]) for r in rr])
        s = np.array([r["score_p_success"] for r in rr], float)
        ck = np.array([r["checkpoint_id"] for r in rr])
        groups = {c: np.where(ck == c)[0] for c in sorted(set(ck))}
        keys = list(groups)
        rng = np.random.default_rng(3)
        vals = []
        for _ in range(2000):
            cix = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
            vals.append(A.auroc(s[cix], y[cix]))
        p3[f"auroc_t80_{lab}"] = {"n": len(rr), "auroc": round(A.auroc(s, y), 4), "ci95_cluster_boot": boot_ci(vals)}
    # within-task AUROC at 4 s excluding epoch 0
    for task in ("can", "square"):
        rr = [r for r in r4 if r["task"] == task and not r["checkpoint_id"].endswith("_e0000")]
        y = np.array([int(r["truth_success"]) for r in rr])
        s = np.array([r["score_p_success"] for r in rr], float)
        p3[f"auroc_t80_{task}_excl_epoch0"] = {"n": len(rr), "n_fail": int((y == 0).sum()), "auroc": round(A.auroc(s, y), 4)}
    # false failure calls: grasped by the cut?
    idx = {(e["checkpoint_id"], e["env_seed"]): e for e in ev}
    ff = []
    for r in evr:
        if r["parse_status"] == "ok" and r["truth_success"] and r["will_succeed"] is False:
            e = idx[(r["checkpoint_id"], r["env_seed"])]
            c = r["cut_step"]
            ff.append({"ckpt": r["checkpoint_id"], "seed": r["env_seed"], "cut": c, "conf": r["confidence"],
                       "fss": e["fss"], "grasped_by_cut": bool(e["grasped"][: c + 1].any()),
                       "lifted_by_cut": bool((e["h"][A.SETTLE: c + 1] >= A.LIFT_M).any())})
    p3["false_failure_calls"] = ff
    # epoch-0 composition of failures
    p3["failures_by_epoch0"] = dict(Counter(f"epoch0={e['epoch'] == 0}" for e in E if not e["success"]))
    res["part3_checks"] = p3

    # ---- 6. ranking claims
    rk = {}
    for k, ab in (("dev2eval/time_only", pols["dev2eval/time_only"]), ("crossfit/time_only", pols["crossfit/time_only"]),
                  ("dev2eval/state_combo", pols["dev2eval/state_combo"])):
        r = A.ranking_effect(ev, ab)
        rk[k] = {"per_run": r["per_run"], "square": {c: v for c, v in r["per_checkpoint"].items() if c.startswith("square")}}
    r = A.ranking_effect(E, ab_any)
    rk["nemotron_any"] = {"per_run": r["per_run"], "square": {c: v for c, v in r["per_checkpoint"].items() if c.startswith("square")}}
    res["ranking_checks"] = rk

    # ---- 7. current data (not frozen) quick look
    A.load_split = _load_split
    A.load_v7 = _load_v7
    cur = ROOT / "data/outputs/hackathon_explore/x2_verify/rerun_current.json"
    if cur.exists():
        c = json.loads(cur.read_text())
        res["current_data_rerun"] = {
            "data": c["data"], "v7_rows": c["part2_cosmos_v7"]["n_eval_rows"],
            "dev2eval_time_only": {k: c["part1_state"]["dev_to_eval"]["time_only/maxfp0"]["eval_all"][k] for k in ("saved_frac", "false_aborts", "n_successes")},
            "dev2eval_state": {k: c["part1_state"]["dev_to_eval"]["state_combo/maxfp0"]["eval_all"][k] for k in ("saved_frac", "false_aborts", "n_successes")},
            "crossfit_time_only": {k: c["part1_state"]["crossfit_eval_by_seed"]["time_only/maxfp0"]["eval_all_crossfit"][k] for k in ("saved_frac", "false_aborts", "n_successes")},
            "crossfit_state": {k: c["part1_state"]["crossfit_eval_by_seed"]["state_combo/maxfp0"]["eval_all_crossfit"][k] for k in ("saved_frac", "false_aborts", "n_successes")},
            "cosmos_vision": {k: c["part2_cosmos_v7"]["dev_to_eval"]["vision_snapshot_rule_eval"][k] for k in ("n_episodes", "saved_frac", "false_aborts")},
            "cosmos_time": {k: c["part2_cosmos_v7"]["dev_to_eval"]["time_only_same_T2_eval"][k] for k in ("n_episodes", "saved_frac", "false_aborts")},
            "square_ranking_time_only_dev2eval": c["part1_state"]["dev_to_eval"]["time_only/maxfp0"]["eval_all"]["ranking"]["per_run"].get("square_s1"),
            "square_per_ckpt_time_only_dev2eval": {k: v for k, v in c["part1_state"]["dev_to_eval"]["time_only/maxfp0"]["eval_all"]["ranking"]["per_checkpoint"].items() if k.startswith("square")},
        }
    OUT.write_text(json.dumps(A.strip(res), indent=1, default=str))
    print(json.dumps({"wrote": str(OUT), "repro_diffs": len(d)}))


if __name__ == "__main__":
    main()
