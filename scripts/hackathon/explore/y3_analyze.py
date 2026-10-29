#!/usr/bin/env python3
"""y3_visual_robustness analysis (Environment B or any python3 with numpy).  EXPLORATORY.

Success-detection metrics per (backend, condition) on the fixed 60-episode subset:
accuracy and balanced accuracy with unparsable / failed outputs counted as WRONG
(balanced_acc_all convention of summarize_diagnostics.py), Wilson 95% CI for
accuracy, checkpoint-clustered bootstrap 95% CIs (resample checkpoints, pool their
episodes; 4000 draws, seed 0), and paired differences vs the control on the same
episodes (cameras vs cam_agentview re-render; degradations vs orig; cam_agentview vs
orig) with a paired clustered bootstrap.  No parameter is fitted.

  python scripts/hackathon/explore/y3_analyze.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCR = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness"
OUT = ROOT / "results/hackathon/explore/y3_visual_robustness.json"
ROWS_OUT = ROOT / "results/hackathon/explore/y3_visual_robustness_rows.jsonl"
B = 4000
CAM_CONDS = ["cam_agentview", "cam_frontview", "cam_birdview", "cam_robot0_eye_in_hand"]
DEG_CONDS = ["orig", "down224", "down112", "jpeg20", "blur", "dim"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def bal_acc(correct, truth):
    correct, truth = np.asarray(correct, bool), np.asarray(truth, bool)
    if truth.sum() == 0 or (~truth).sum() == 0:
        return float("nan")
    return 0.5 * (correct[truth].mean() + correct[~truth].mean())


def load(backend):
    p = SCR / f"judge_{backend}.jsonl"
    rows = {}
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("infra_error"):
            # keep the record only if nothing better exists (counted separately)
            rows.setdefault((r["condition"], r["checkpoint_id"], r["env_seed"]), r)
            continue
        rows[(r["condition"], r["checkpoint_id"], r["env_seed"])] = r
    return rows


def correct_of(r):
    if r is None or r.get("infra_error") or r.get("parse_status") not in ("ok", "ok_repaired") or r.get("pred_success") is None:
        return False
    return bool(r["pred_success"]) == bool(r["truth_success"])


def cluster_boot(keys_by_ck, stat_fn, rng):
    cks = sorted(keys_by_ck)
    vals = []
    for _ in range(B):
        draw = rng.integers(0, len(cks), len(cks))
        keys = [k for i in draw for k in keys_by_ck[cks[i]]]
        v = stat_fn(keys)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return [None, None], 0
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)], B - len(vals)


def summarize(rows, cond, episodes, rng):
    keys = [(e["checkpoint_id"], e["env_seed"]) for e in episodes]
    truth = {(e["checkpoint_id"], e["env_seed"]): bool(e["success"]) for e in episodes}
    task = {(e["checkpoint_id"], e["env_seed"]): e["task"] for e in episodes}
    have = [k for k in keys if (cond,) + k in rows]
    if not have:
        return None
    corr = {k: correct_of(rows[(cond,) + k]) for k in have}
    n, kc = len(have), sum(corr.values())
    parsed = sum(1 for k in have if rows[(cond,) + k].get("parse_status") in ("ok", "ok_repaired"))
    infra = sum(1 for k in have if rows[(cond,) + k].get("infra_error"))
    by_ck = defaultdict(list)
    for k in have:
        by_ck[k[0]].append(k)
    acc_ci, _ = cluster_boot(by_ck, lambda ks: float(np.mean([corr[k] for k in ks])), rng)
    ba = bal_acc([corr[k] for k in have], [truth[k] for k in have])
    ba_ci, ba_skip = cluster_boot(by_ck, lambda ks: bal_acc([corr[k] for k in ks], [truth[k] for k in ks]), rng)
    tp = sum(1 for k in have if truth[k] and corr[k])
    tn = sum(1 for k in have if not truth[k] and corr[k])
    npos = sum(truth[k] for k in have)
    per_task = {}
    for t in ("can", "square"):
        ks = [k for k in have if task[k] == t]
        if ks:
            c = sum(corr[k] for k in ks)
            per_task[t] = {"n": len(ks), "acc": round(c / len(ks), 4), "wilson95": wilson(c, len(ks)),
                           "bal_acc": round(bal_acc([corr[k] for k in ks], [truth[k] for k in ks]), 4)}
    preds = Counter()
    for k in have:
        r = rows[(cond,) + k]
        loc = r.get("object_location") if r["backend"] == "local" else r.get("pred_failure_type")
        preds[f"{task[k]}|truth_{'succ' if truth[k] else 'fail'}|{loc}"] += 1
    return {"condition": cond, "n": n, "n_parsed": parsed, "n_infra_error": infra, "n_correct": kc,
            "acc": round(kc / n, 4), "acc_wilson95": wilson(kc, n), "acc_cluster_boot95": acc_ci,
            "bal_acc": round(ba, 4), "bal_acc_cluster_boot95": ba_ci, "bal_acc_boot_undefined_draws": ba_skip,
            "tpr": round(tp / npos, 4) if npos else None, "tpr_wilson95": wilson(tp, npos),
            "tnr": round(tn / (n - npos), 4) if n - npos else None, "tnr_wilson95": wilson(tn, n - npos),
            "n_success_truth": npos, "per_task": per_task,
            "pred_label_counts": dict(sorted(preds.items()))}


def paired(rows, cond, ctrl, episodes, rng):
    truth = {(e["checkpoint_id"], e["env_seed"]): bool(e["success"]) for e in episodes}
    keys = [(e["checkpoint_id"], e["env_seed"]) for e in episodes
            if (cond, e["checkpoint_id"], e["env_seed"]) in rows and (ctrl, e["checkpoint_id"], e["env_seed"]) in rows]
    if not keys:
        return None
    a = {k: correct_of(rows[(cond,) + k]) for k in keys}
    c = {k: correct_of(rows[(ctrl,) + k]) for k in keys}
    same_pred = sum(1 for k in keys if rows[(cond,) + k].get("pred_success") == rows[(ctrl,) + k].get("pred_success"))
    by_ck = defaultdict(list)
    for k in keys:
        by_ck[k[0]].append(k)
    d_acc = float(np.mean([a[k] for k in keys]) - np.mean([c[k] for k in keys]))
    d_ba = bal_acc([a[k] for k in keys], [truth[k] for k in keys]) - bal_acc([c[k] for k in keys], [truth[k] for k in keys])
    acc_ci, _ = cluster_boot(by_ck, lambda ks: float(np.mean([a[k] for k in ks]) - np.mean([c[k] for k in ks])), rng)
    ba_ci, skip = cluster_boot(by_ck, lambda ks: bal_acc([a[k] for k in ks], [truth[k] for k in ks])
                               - bal_acc([c[k] for k in ks], [truth[k] for k in ks]), rng)
    b01 = sum(1 for k in keys if c[k] and not a[k])  # control right, condition wrong
    b10 = sum(1 for k in keys if a[k] and not c[k])
    # exact two-sided McNemar (binomial on discordant pairs), episode-level (ignores clustering)
    nd = b01 + b10
    p = 1.0 if nd == 0 else min(1.0, 2 * sum(math.comb(nd, i) for i in range(0, min(b01, b10) + 1)) / 2 ** nd)
    return {"condition": cond, "control": ctrl, "n_pairs": len(keys), "d_acc": round(d_acc, 4),
            "d_acc_paired_cluster_boot95": acc_ci, "d_bal_acc": round(d_ba, 4),
            "d_bal_acc_paired_cluster_boot95": ba_ci, "ctrl_right_cond_wrong": b01, "cond_right_ctrl_wrong": b10,
            "mcnemar_exact_p_episode_level": round(p, 4), "same_prediction_frac": round(same_pred / len(keys), 4)}


def frozen_reference(episodes):
    """Accuracy of the frozen eval runs on the same 60 episodes + agreement with our orig rerun."""
    out = {}
    keys = {(e["checkpoint_id"], e["env_seed"]) for e in episodes}
    for name, fn in (("local", "eval_local_v6.jsonl"), ("nebius", "eval_nebius_v2_pad201.jsonl")):
        ref = {}
        for line in (ROOT / "data/outputs/hackathon_cosmos" / fn).read_text().splitlines():
            r = json.loads(line)
            k = (r["checkpoint_id"], r["env_seed"])
            if k in keys:
                ref[k] = r
        ours = load(name)
        n = len(ref)
        k_ok = sum(1 for r in ref.values() if r.get("scored") and r.get("success_correct"))
        agree = [ref[k].get("cosmos_task_success") == ours[("orig",) + k].get("pred_success")
                 for k in ref if ("orig",) + k in ours]
        out[name] = {"file": f"data/outputs/hackathon_cosmos/{fn}", "n": n, "acc": round(k_ok / n, 4) if n else None,
                     "acc_wilson95": wilson(k_ok, n), "orig_rerun_same_prediction": round(float(np.mean(agree)), 4) if agree else None,
                     "n_compared": len(agree)}
    return out


def replay_summary():
    p = SCR / "render/replay_log_full.jsonl"
    if not p.exists():
        return None
    rs = [json.loads(l) for l in p.read_text().splitlines()]
    pr = [x for r in rs if r.get("pixel_check") for x in r["pixel_check"]["psnr_raw_vs_orig_mp4"]]
    pe = [r["pixel_check"].get("psnr_reencoded_vs_orig_mp4_last") for r in rs if r.get("pixel_check")]
    pe = [x for x in pe if x is not None]
    chk = SCR / "render/replay_log_check_with_close.jsonl"
    with_close = [json.loads(l) for l in chk.read_text().splitlines()] if chk.exists() else []
    return {
        "n_episodes_replayed": len(rs),
        "max_abs_eef_err_m": max(r["max_abs_eef_err_m"] for r in rs),
        "max_abs_obj_err_m": max(r["max_abs_obj_err_m"] for r in rs),
        "min_grasped_step_agreement": min(r["grasped_step_agreement"] for r in rs),
        "min_success_step_agreement": min(r["success_step_agreement"] for r in rs),
        "success_flag_match": sum(r["success_match"] for r in rs),
        "agentview_psnr_raw_vs_orig_mp4_db": {"min": min(pr), "median": float(np.median(pr)), "max": max(pr)} if pr else None,
        "agentview_psnr_reencoded_last_vs_orig_last_db": {"min": min(pe), "median": float(np.median(pe))} if pe else None,
        "check_run_with_env_close": [{"checkpoint_id": r["checkpoint_id"], "env_seed": r["env_seed"],
                                      "max_abs_obj_err_m": r["max_abs_obj_err_m"],
                                      "psnr_raw_vs_orig_mp4": r["pixel_check"]["psnr_raw_vs_orig_mp4"]} for r in with_close],
        "total_wall_s": round(sum(r["wall_s"] for r in rs), 1),
    }


def main():
    sel = json.loads((ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json").read_text())
    eps = sel["episodes"]
    rng = np.random.default_rng(0)
    res = {"name": "y3_visual_robustness", "exploratory": True,
           "selection": {k: sel[k] for k in ("rule", "strata", "n", "n_checkpoints")},
           "replay_determinism": replay_summary(), "frozen_eval_reference_same_episodes": frozen_reference(eps),
           "bootstrap": {"draws": B, "seed": 0, "cluster": "checkpoint_id"}, "backends": {}}
    all_rows = []
    for backend in ("local", "nebius"):
        rows = load(backend)
        all_rows += list(rows.values())
        conds = [c for c in DEG_CONDS + CAM_CONDS if any(k[0] == c for k in rows)]
        res["backends"][backend] = {
            "judge": "Cosmos3-Edge v6_perception_last (last frame)" if backend == "local"
            else "Nemotron-3.5-Super-VL v2_evidence_first, video padded to 20.1 s",
            "conditions": {c: summarize(rows, c, eps, rng) for c in conds},
            "paired_vs_control": [p for p in (
                [paired(rows, c, "orig", eps, rng) for c in DEG_CONDS[1:] if c in conds]
                + ([paired(rows, "cam_agentview", "orig", eps, rng)] if "cam_agentview" in conds else [])
                + [paired(rows, c, "cam_agentview", eps, rng) for c in CAM_CONDS[1:] if c in conds]) if p],
            "n_calls_logged": len(rows),
        }
    if (ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl").exists():
        res["nebius_calls_y3"] = sum(1 for l in (ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl").read_text().splitlines()
                                     if '"explore/y3_visual_robustness/' in l)
    OUT.write_text(json.dumps(res, indent=1))
    with open(ROWS_OUT, "w") as f:
        for r in all_rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "params"}, sort_keys=True) + "\n")
    for backend, b in res["backends"].items():
        print(f"== {backend}")
        for c, s in b["conditions"].items():
            if s:
                print(f"  {c:24s} n={s['n']:3d} parsed={s['n_parsed']:3d} acc={s['acc']:.3f} {s['acc_wilson95']} "
                      f"boot{s['acc_cluster_boot95']} bal={s['bal_acc']:.3f} {s['bal_acc_cluster_boot95']} "
                      f"tpr={s['tpr']} tnr={s['tnr']}")
        for p in b["paired_vs_control"]:
            print(f"  {p['condition']:24s} vs {p['control']:14s} n={p['n_pairs']} dacc={p['d_acc']:+.3f} "
                  f"{p['d_acc_paired_cluster_boot95']} dbal={p['d_bal_acc']:+.3f} {p['d_bal_acc_paired_cluster_boot95']} "
                  f"disc {p['ctrl_right_cond_wrong']}/{p['cond_right_ctrl_wrong']} same={p['same_prediction_frac']}")
    print(json.dumps(res["replay_determinism"], indent=0)[:1500])
    print(json.dumps(res["frozen_eval_reference_same_episodes"]))


if __name__ == "__main__":
    main()
