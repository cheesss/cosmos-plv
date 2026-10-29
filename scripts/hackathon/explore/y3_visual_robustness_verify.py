#!/usr/bin/env python3
"""Adversarial verification of y3_visual_robustness (EXPLORATORY, post-hoc; CPU only).

Re-scores the SAVED raw judge outputs with independent code (does not import y3_analyze),
and does not overwrite any y3 file.  Checks:
  1. row integrity (one row per condition x episode, no duplicates, truth labels vs
     summary.json and vs npz success trace, video/prompt hashes in the Nebius evidence log);
  2. recompute acc / bal-acc / TPR / TNR / Wilson / paired deltas and compare with
     results/hackathon/explore/y3_visual_robustness.json;
  3. predicted-success-rate per condition (does a far camera inflate VLM-judged success?)
     and the decomposition of the paired change into extra FN vs extra FP;
  4. multiplicity: Bonferroni-adjusted checkpoint-clustered paired bootstrap intervals
     (18 comparisons = 9 per backend x 2) and Holm on the episode-level McNemar p;
  5. leave-one-checkpoint-out range of each camera delta, and per-task deltas;
  6. cluster structure of the 60-episode subset;
  7. specific counts quoted in the report (on_peg, right_bin, not_visible, Nemotron
     eye-in-hand failure types) with the agentview baseline next to them.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_visual_robustness_verify.py
"""
from __future__ import annotations

import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCR = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness"
RES = ROOT / "results/hackathon/explore/y3_visual_robustness.json"
SEL = ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json"
EVID = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
OUT = ROOT / "results/hackathon/explore/y3_visual_robustness_verification.json"
CAMS = ["cam_agentview", "cam_frontview", "cam_birdview", "cam_robot0_eye_in_hand"]
DEGS = ["orig", "down224", "down112", "jpeg20", "blur", "dim"]
COMPARISONS = ([(c, "orig") for c in DEGS[1:]] + [("cam_agentview", "orig")]
               + [(c, "cam_agentview") for c in CAMS[1:]])
B = 20000


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def bal(corr, truth):
    corr, truth = np.asarray(corr, bool), np.asarray(truth, bool)
    if truth.all() or (~truth).all():
        return float("nan")
    return 0.5 * (corr[truth].mean() + corr[~truth].mean())


def main():
    sel = json.loads(SEL.read_text())
    eps = sel["episodes"]
    keys = [(e["checkpoint_id"], e["env_seed"]) for e in eps]
    truth = {(e["checkpoint_id"], e["env_seed"]): bool(e["success"]) for e in eps}
    task = {(e["checkpoint_id"], e["env_seed"]): e["task"] for e in eps}
    ck_of = {k: k[0] for k in keys}
    out = {"name": "y3_visual_robustness_verification", "exploratory": True}

    # ---- 0. truth labels vs current summary.json and npz success trace
    mism_summary, mism_npz = [], []
    for e in eps:
        summ = json.loads((ROOT / "data/outputs/hackathon_rollouts_eval" / e["checkpoint_id"] / "summary.json").read_text())
        cur = [x for x in summ["episodes"] if x["env_seed"] == e["env_seed"]][0]
        if bool(cur["success"]) != bool(e["success"]) or cur["failure_type"] != e["failure_type"]:
            mism_summary.append((e["checkpoint_id"], e["env_seed"]))
        z = np.load(ROOT / e["episode_npz"])
        if bool(np.asarray(z["success"]).any()) != bool(e["success"]):
            mism_npz.append((e["checkpoint_id"], e["env_seed"]))
    eps_per_ck = Counter(k[0] for k in keys)
    out["truth_and_clusters"] = {
        "truth_mismatch_vs_current_summary": mism_summary, "truth_mismatch_vs_npz_success_any": mism_npz,
        "n_success": sum(truth.values()), "n_fail": len(keys) - sum(truth.values()),
        "n_checkpoints": len(eps_per_ck), "episodes_per_checkpoint": dict(sorted(eps_per_ck.items())),
        "cluster_size_hist": dict(sorted(Counter(eps_per_ck.values()).items())),
        "checkpoints_per_task": dict(Counter(k.split("_")[0] for k in eps_per_ck)),
    }

    # ---- 1. row integrity
    rows, integrity = {}, {}
    for be in ("local", "nebius"):
        raw = [json.loads(l) for l in (SCR / f"judge_{be}.jsonl").read_text().splitlines()]
        cnt = Counter((r["condition"], r["checkpoint_id"], r["env_seed"]) for r in raw)
        dup = {"|".join(map(str, k)): v for k, v in cnt.items() if v > 1}
        rows[be] = {(r["condition"], r["checkpoint_id"], r["env_seed"]): r for r in raw if not r.get("infra_error")}
        missing = [(c,) + k for c in DEGS + CAMS for k in keys if (c,) + k not in rows[be]]
        truth_bad = [k for k, r in rows[be].items() if bool(r["truth_success"]) != truth[(k[1], k[2])]]
        integrity[be] = {"n_rows": len(raw), "n_infra_error": sum(1 for r in raw if r.get("infra_error")),
                         "duplicates": dup, "missing": missing, "row_truth_disagrees_with_selection": truth_bad,
                         "parse_status": dict(Counter(r["parse_status"] for r in raw)),
                         "backend_errors": sum(1 for r in raw if r.get("backend_error"))}
    # local: input file per condition must differ from orig except for exact copies
    # nebius: evidence log tags / hashes
    ev = [json.loads(l) for l in EVID.read_text().splitlines() if '"explore/y3_visual_robustness/' in l]
    ev_by_tag = defaultdict(list)
    for r in ev:
        ev_by_tag[r["tag"]].append(r)
    nb = rows["nebius"]
    req_missing = [r["tag"] for r in nb.values() if r["request_id"] not in {x["request_id"] for x in ev_by_tag[r["tag"]]}]
    vid_hash = {}
    for t, lst in ev_by_tag.items():
        _, _, cond, ck, sd = t.split("/")
        vid_hash[(cond, ck, int(sd.replace("seed", "")))] = lst[-1]["params"].get("video_sha256")
    same_hash_as_orig = Counter()
    for (cond, ck, sd), h in vid_hash.items():
        if cond != "orig" and h == vid_hash.get(("orig", ck, sd)):
            same_hash_as_orig[cond] += 1
    prompt_hash = defaultdict(set)
    for t, lst in ev_by_tag.items():
        _, _, cond, ck, sd = t.split("/")
        prompt_hash[(task[(ck, int(sd.replace('seed', '')))], cond)].add(lst[-1]["params"].get("prompt_sha256"))
    prompts_per_task = {tk: len({h for (t2, c), hs in prompt_hash.items() if t2 == tk for h in hs}) for tk in ("can", "square")}
    integrity["nebius_evidence"] = {
        "n_calls_logged_y3": len(ev), "n_unique_tags": len(ev_by_tag),
        "tags_with_more_than_one_call": sum(1 for v in ev_by_tag.values() if len(v) > 1),
        "n_calls_with_error": sum(1 for r in ev if r.get("error")),
        "rows_whose_request_id_not_in_evidence": len(req_missing),
        "videos_identical_to_orig_by_condition": dict(same_hash_as_orig),
        "distinct_prompt_hashes_per_task_across_all_conditions": prompts_per_task,
        "temperatures": dict(Counter(str(r["params"].get("temperature")) for r in ev)),
        "max_tokens": dict(Counter(str(r["params"].get("max_tokens")) for r in ev)),
        "padded_to_s": dict(Counter(str(r["params"].get("video_padded_to_s")) for r in ev)),
    }
    out["integrity"] = integrity

    def corr_of(r):
        return (r is not None and r.get("parse_status") in ("ok", "ok_repaired") and r.get("pred_success") is not None
                and bool(r["pred_success"]) == bool(r["truth_success"]))

    # ---- 2. recompute and compare with the published JSON
    pub = json.loads(RES.read_text())
    recompute, diffs = {}, []
    for be in ("local", "nebius"):
        recompute[be] = {}
        for c in DEGS + CAMS:
            cr = [corr_of(rows[be].get((c,) + k)) for k in keys]
            tr = [truth[k] for k in keys]
            pred_pos = [bool(rows[be][(c,) + k].get("pred_success")) for k in keys]
            tp = sum(1 for x, t in zip(cr, tr) if x and t)
            tn = sum(1 for x, t in zip(cr, tr) if x and not t)
            npos = sum(tr)
            s = {"acc": round(sum(cr) / len(cr), 4), "wilson": wilson(sum(cr), len(cr)), "bal_acc": round(bal(cr, tr), 4),
                 "tpr": round(tp / npos, 4), "tnr": round(tn / (len(tr) - npos), 4),
                 "fn": npos - tp, "fp_or_unparsed_fail": (len(tr) - npos) - tn,
                 "pred_success_rate": round(float(np.mean(pred_pos)), 4),
                 "true_success_rate": round(npos / len(tr), 4)}
            for t in ("can", "square"):
                ks = [k for k in keys if task[k] == t]
                s[f"pred_success_rate_{t}"] = round(float(np.mean([bool(rows[be][(c,) + k].get("pred_success")) for k in ks])), 4)
            recompute[be][c] = s
            p = pub["backends"][be]["conditions"][c]
            for a, b_ in (("acc", "acc"), ("bal_acc", "bal_acc"), ("tpr", "tpr"), ("tnr", "tnr")):
                if abs(p[b_] - s[a]) > 1e-3:
                    diffs.append(f"{be}/{c}/{a}: published {p[b_]} vs recomputed {s[a]}")
            if p["acc_wilson95"] != s["wilson"]:
                diffs.append(f"{be}/{c}/wilson: {p['acc_wilson95']} vs {s['wilson']}")
    out["recomputed_conditions"] = recompute

    # ---- 3/4/5. paired comparisons with Bonferroni / Holm / LOCO / per task
    rng = np.random.default_rng(12345)
    cks = sorted(eps_per_ck)
    by_ck = defaultdict(list)
    for k in keys:
        by_ck[k[0]].append(k)
    draws = [rng.integers(0, len(cks), len(cks)) for _ in range(B)]
    n_comp = 2 * len(COMPARISONS)
    alpha_b = 0.05 / n_comp
    paired = []
    for be in ("local", "nebius"):
        for cond, ctrl in COMPARISONS:
            a = {k: corr_of(rows[be][(cond,) + k]) for k in keys}
            c = {k: corr_of(rows[be][(ctrl,) + k]) for k in keys}
            d = np.mean([a[k] for k in keys]) - np.mean([c[k] for k in keys])
            b01 = sum(1 for k in keys if c[k] and not a[k])
            b10 = sum(1 for k in keys if a[k] and not c[k])
            nd = b01 + b10
            pm = 1.0 if nd == 0 else min(1.0, 2 * sum(math.comb(nd, i) for i in range(min(b01, b10) + 1)) / 2 ** nd)
            # FN/FP decomposition of the change
            extra_fn = sum(1 for k in keys if truth[k] and c[k] and not a[k]) - sum(1 for k in keys if truth[k] and a[k] and not c[k])
            extra_fp = sum(1 for k in keys if not truth[k] and c[k] and not a[k]) - sum(1 for k in keys if not truth[k] and a[k] and not c[k])
            boots = []
            for dr in draws:
                ks = [k for i in dr for k in by_ck[cks[i]]]
                boots.append(np.mean([a[k] for k in ks]) - np.mean([c[k] for k in ks]))
            boots = np.asarray(boots)
            loco = []
            for ck in cks:
                ks = [k for k in keys if k[0] != ck]
                loco.append(np.mean([a[k] for k in ks]) - np.mean([c[k] for k in ks]))
            per_task = {}
            for t in ("can", "square"):
                ks = [k for k in keys if task[k] == t]
                per_task[t] = {"d_acc": round(float(np.mean([a[k] for k in ks]) - np.mean([c[k] for k in ks])), 4),
                               "disc_ctrl_right_cond_wrong": sum(1 for k in ks if c[k] and not a[k]),
                               "disc_cond_right_ctrl_wrong": sum(1 for k in ks if a[k] and not c[k])}
            paired.append({
                "backend": be, "condition": cond, "control": ctrl, "d_acc": round(float(d), 4),
                "ci95_cluster": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                "ci_bonferroni_cluster": [round(float(np.percentile(boots, 100 * alpha_b / 2)), 4),
                                          round(float(np.percentile(boots, 100 * (1 - alpha_b / 2))), 4)],
                "boot_frac_ge_0": round(float(np.mean(boots >= 0)), 5),
                "disc_ctrl_right_cond_wrong": b01, "disc_cond_right_ctrl_wrong": b10, "mcnemar_p": pm,
                "extra_false_negatives": extra_fn, "extra_false_positives_or_unparsed_fail": extra_fp,
                "loco_min": round(float(min(loco)), 4), "loco_max": round(float(max(loco)), 4),
                "per_task": per_task,
            })
            pp = [x for x in pub["backends"][be]["paired_vs_control"] if x["condition"] == cond and x["control"] == ctrl][0]
            if abs(pp["d_acc"] - d) > 1e-3 or pp["ctrl_right_cond_wrong"] != b01 or pp["cond_right_ctrl_wrong"] != b10:
                diffs.append(f"{be}/{cond} vs {ctrl}: published d={pp['d_acc']} {pp['ctrl_right_cond_wrong']}/{pp['cond_right_ctrl_wrong']}"
                             f" vs recomputed {d:.4f} {b01}/{b10}")
    # Holm on McNemar p over all 18 comparisons (the cam_agentview-vs-orig control check included)
    order = sorted(range(len(paired)), key=lambda i: paired[i]["mcnemar_p"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (n_comp - rank) * paired[i]["mcnemar_p"])
        running = max(running, adj)
        paired[i]["mcnemar_p_holm"] = round(running, 4)
    out["paired"] = paired
    out["multiplicity"] = {"n_comparisons": n_comp, "bonferroni_two_sided_alpha": alpha_b, "bootstrap_draws": B,
                           "bootstrap_seed": 12345}
    out["published_vs_recomputed_differences"] = diffs

    # ---- Nemotron run-to-run noise
    keyset = set(keys)
    ref = {}
    for line in (ROOT / "data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = (r["checkpoint_id"], r["env_seed"])
        if k in keyset:
            ref[k] = r
    flips = [k for k in keys if ref[k].get("cosmos_task_success") != rows["nebius"][("orig",) + k].get("pred_success")]
    flips_succ = sum(1 for k in flips if truth[k])
    out["nemotron_noise"] = {
        "orig_vs_frozen_pad201_flips": len(flips), "flips_on_true_success": flips_succ,
        "flips": [list(k) for k in flips],
        "orig_vs_cam_agentview_flips": sum(1 for k in keys if rows["nebius"][("orig",) + k].get("pred_success")
                                           != rows["nebius"][("cam_agentview",) + k].get("pred_success")),
    }

    # ---- 7. report-quoted counts with agentview baseline
    def cnt(be, cond, t, succ, field):
        return dict(Counter(rows[be][(cond,) + k].get(field) for k in keys if task[k] == t and truth[k] == succ))
    q = {}
    for cond in CAMS + ["orig", "jpeg20"]:
        q[f"local/{cond}/square_fail_object_location"] = cnt("local", cond, "square", False, "object_location")
        q[f"local/{cond}/square_succ_object_location"] = cnt("local", cond, "square", True, "object_location")
        q[f"local/{cond}/can_fail_object_location"] = cnt("local", cond, "can", False, "object_location")
        q[f"nebius/{cond}/succ_pred_success"] = dict(Counter(str(rows["nebius"][(cond,) + k].get("pred_success")) for k in keys if truth[k]))
        q[f"nebius/{cond}/succ_pred_failure_type"] = dict(Counter(str(rows["nebius"][(cond,) + k].get("pred_failure_type")) for k in keys if truth[k] and rows["nebius"][(cond,) + k].get("pred_success") is False))
        q[f"nebius/{cond}/fail_judged_success_by_truth_type"] = dict(Counter(f"{task[k]}/{rows['nebius'][(cond,) + k]['truth_failure_type']}" for k in keys if not truth[k] and rows["nebius"][(cond,) + k].get("pred_success") is True))
    out["quoted_counts"] = q
    OUT.write_text(json.dumps(out, indent=1, default=str))

    print("differences vs published:", diffs or "none")
    print(json.dumps(out["truth_and_clusters"], default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'missing'} | {"n_missing": len(v.get('missing', []))} if isinstance(v, dict) and 'missing' in v else v for k, v in integrity.items()}, default=str, indent=0))
    for be in recompute:
        for c, s in recompute[be].items():
            print(be, c, "acc", s["acc"], "pred_succ_rate", s["pred_success_rate"], "true", s["true_success_rate"],
                  "fn", s["fn"], "fp", s["fp_or_unparsed_fail"], "can", s["pred_success_rate_can"], "sq", s["pred_success_rate_square"])
    for p in paired:
        print(f"{p['backend']:6s} {p['condition']:24s} vs {p['control']:13s} d={p['d_acc']:+.3f} ci95={p['ci95_cluster']} "
              f"bonf={p['ci_bonferroni_cluster']} P(>=0)={p['boot_frac_ge_0']} disc={p['disc_ctrl_right_cond_wrong']}/{p['disc_cond_right_ctrl_wrong']} "
              f"p={p['mcnemar_p']:.4f} holm={p['mcnemar_p_holm']} +FN={p['extra_false_negatives']} +FP={p['extra_false_positives_or_unparsed_fail']} "
              f"loco=[{p['loco_min']},{p['loco_max']}] task={p['per_task']}")
    print(json.dumps(out["nemotron_noise"]))
    for k, v in q.items():
        print(k, v)


if __name__ == "__main__":
    main()
