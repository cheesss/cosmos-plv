#!/usr/bin/env python3
"""Verifier for y3_visual_robustness: selection / weighting / alternative clustering (CPU).

  (a) cherry-pick check: frozen-run accuracy per stratum (task x failure_type) on the 60-episode
      subset vs on the whole 250-episode selection universe;
  (b) post-stratified paired deltas: per-stratum delta re-weighted to the universe composition
      (the subset is failure-enriched: 36/60 failures vs 61/250 in the universe);
  (c) alternative clustering: paired bootstrap clustered by env seed (episodes with the same seed
      share the initial scene across checkpoints) instead of by checkpoint;
  (d) Nemotron retest summary merged in (if the retest file exists).
Writes results/hackathon/explore/y3_visual_robustness_verification_strata.json.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCR = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness"
OUT = ROOT / "results/hackathon/explore/y3_visual_robustness_verification_strata.json"
COMP = [("down224", "orig"), ("down112", "orig"), ("jpeg20", "orig"), ("blur", "orig"), ("dim", "orig"),
        ("cam_frontview", "cam_agentview"), ("cam_birdview", "cam_agentview"), ("cam_robot0_eye_in_hand", "cam_agentview")]


def main():
    sel = json.loads((ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json").read_text())
    eps = sel["episodes"]
    keys = [(e["checkpoint_id"], e["env_seed"]) for e in eps]
    stratum = {(e["checkpoint_id"], e["env_seed"]): f"{e['task']}/{e['failure_type']}" for e in eps}
    truth = {(e["checkpoint_id"], e["env_seed"]): bool(e["success"]) for e in eps}
    universe = Counter()
    ustr = {}
    for ck in sel["universe_checkpoints"] if "universe_checkpoints" in sel else []:
        for ep in json.loads((ROOT / "data/outputs/hackathon_rollouts_eval" / ck / "summary.json").read_text())["episodes"]:
            universe[f"{ep['task']}/{ep['failure_type']}"] += 1
            ustr[(ck, ep["env_seed"])] = f"{ep['task']}/{ep['failure_type']}"
    out = {"universe_strata": dict(universe), "subset_strata": dict(Counter(stratum.values()))}

    # (a) frozen accuracy per stratum, subset vs universe
    fro = {}
    for name, fn in (("local", "eval_local_v6.jsonl"), ("nebius", "eval_nebius_v2_pad201.jsonl")):
        acc_u, acc_s = defaultdict(list), defaultdict(list)
        for line in (ROOT / "data/outputs/hackathon_cosmos" / fn).read_text().splitlines():
            r = json.loads(line)
            k = (r["checkpoint_id"], r["env_seed"])
            if k not in ustr:
                continue
            c = bool(r.get("scored") and r.get("success_correct"))
            acc_u[ustr[k]].append(c)
            if k in stratum:
                acc_s[ustr[k]].append(c)
        fro[name] = {s: {"universe_acc": round(float(np.mean(acc_u[s])), 3), "universe_n": len(acc_u[s]),
                         "subset_acc": round(float(np.mean(acc_s[s])), 3) if acc_s[s] else None, "subset_n": len(acc_s[s])}
                     for s in sorted(acc_u)}
    out["frozen_acc_by_stratum"] = fro

    rows = {}
    for be in ("local", "nebius"):
        rows[be] = {}
        for line in (SCR / f"judge_{be}.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[be][(r["condition"], r["checkpoint_id"], r["env_seed"])] = r

    def ok(r):
        return r["parse_status"] in ("ok", "ok_repaired") and r["pred_success"] is not None and \
            bool(r["pred_success"]) == bool(r["truth_success"])

    tot = sum(universe.values())
    w = {s: universe[s] / tot for s in universe}
    rng = np.random.default_rng(7)
    seeds = sorted({k[1] for k in keys})
    by_seed = defaultdict(list)
    for k in keys:
        by_seed[k[1]].append(k)
    res = []
    for be in ("local", "nebius"):
        for cond, ctrl in COMP:
            d = {k: float(ok(rows[be][(cond,) + k])) - float(ok(rows[be][(ctrl,) + k])) for k in keys}
            per = defaultdict(list)
            for k in keys:
                per[stratum[k]].append(d[k])
            ps = sum(w[s] * np.mean(per[s]) for s in per)
            # acc of condition and control re-weighted
            acc_c = sum(w[s] * np.mean([ok(rows[be][(cond,) + k]) for k in keys if stratum[k] == s]) for s in per)
            acc_0 = sum(w[s] * np.mean([ok(rows[be][(ctrl,) + k]) for k in keys if stratum[k] == s]) for s in per)
            # weighted predicted-success rate (what a screening score would see)
            ps_c = sum(w[s] * np.mean([bool(rows[be][(cond,) + k]["pred_success"]) for k in keys if stratum[k] == s]) for s in per)
            ps_0 = sum(w[s] * np.mean([bool(rows[be][(ctrl,) + k]["pred_success"]) for k in keys if stratum[k] == s]) for s in per)
            boots_seed, boots_ps = [], []
            for _ in range(5000):
                dr = rng.integers(0, len(seeds), len(seeds))
                ks = [k for i in dr for k in by_seed[seeds[i]]]
                boots_seed.append(np.mean([d[k] for k in ks]))
                # stratified episode bootstrap for the post-stratified delta
                v = 0.0
                for s in per:
                    arr = np.asarray(per[s])
                    v += w[s] * arr[rng.integers(0, len(arr), len(arr))].mean()
                boots_ps.append(v)
            res.append({"backend": be, "condition": cond, "control": ctrl,
                        "d_acc_subset": round(float(np.mean(list(d.values()))), 4),
                        "d_acc_by_stratum": {s: round(float(np.mean(v)), 3) for s, v in sorted(per.items())},
                        "d_acc_poststratified_to_universe": round(float(ps), 4),
                        "d_acc_poststratified_ci95_stratified_episode_boot": [round(float(np.percentile(boots_ps, 2.5)), 4),
                                                                             round(float(np.percentile(boots_ps, 97.5)), 4)],
                        "acc_poststratified": {"control": round(float(acc_0), 4), "condition": round(float(acc_c), 4)},
                        "pred_success_rate_poststratified": {"control": round(float(ps_0), 4), "condition": round(float(ps_c), 4),
                                                             "true": round(sum(w[s] for s in w if s.endswith('/none')), 4)},
                        "d_acc_ci95_cluster_by_env_seed": [round(float(np.percentile(boots_seed, 2.5)), 4),
                                                           round(float(np.percentile(boots_seed, 97.5)), 4)]})
    out["paired_poststratified_and_seed_cluster"] = res
    rt_path = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness_verification/retest_nebius.jsonl"
    if rt_path.exists():
        rt = [json.loads(l) for l in rt_path.read_text().splitlines()]
        out["nemotron_birdview_retest_rows"] = [{k: r[k] for k in ("condition", "checkpoint_id", "env_seed", "task", "truth_success",
                                                                  "y3_discordance", "y3_pred_success", "pred_success",
                                                                  "pred_failure_type", "parse_status", "backend_error")} for r in rt]
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["universe_strata"]), json.dumps(out["subset_strata"]))
    for be, v in fro.items():
        print(be, json.dumps(v))
    for r in res:
        print(f"{r['backend']:6s} {r['condition']:24s} subset d={r['d_acc_subset']:+.3f} post-strat d={r['d_acc_poststratified_to_universe']:+.3f} "
              f"{r['d_acc_poststratified_ci95_stratified_episode_boot']} acc {r['acc_poststratified']} predsucc {r['pred_success_rate_poststratified']} "
              f"seedclust {r['d_acc_ci95_cluster_by_env_seed']} strata {r['d_acc_by_stratum']}")


if __name__ == "__main__":
    main()
