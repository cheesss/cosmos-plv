#!/usr/bin/env python3
"""Print the markdown tables of the x2_early_stop report from the results JSON (no new numbers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/hackathon/explore/x2_early_stop.json"
r = json.loads(P.read_text())


def ci(x):
    return "-" if not x or x[0] is None else f"[{x[0]:.2f}, {x[1]:.2f}]"


def f(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def pol_row(name, m, extra=""):
    rk = m.get("ranking", {})
    pr = rk.get("per_run", {})
    rho = ", ".join(f"{k}:{f(v['spearman_est_vs_full'], 2)}/{f(v['regret_top1_pessimistic'], 1)}" for k, v in pr.items())
    return (f"| {name} | {extra} | {m['n_episodes']} | {m['true_aborts']}/{m['n_failures']} | {m['false_aborts']}/{m['n_successes']} "
            f"| {f(m['precision'], 2)} {ci(m['precision_ci95'])} | {f(m['false_abort_rate'], 3)} {ci(m['false_abort_rate_ci95'])} "
            f"| {f(m['saved_frac'], 3)} {ci(m['saved_frac_ci95'])} | {m['robot_time_saved_s']:.0f} / {m['robot_time_total_s']:.0f} "
            f"| {rk.get('checkpoints_changed', '-')}/{rk.get('n_checkpoints', '-')}, max {f(rk.get('max_abs_delta_success'), 1)} | {rho} |")


HDR = ("| policy | chosen params | episodes | failures aborted | successes falsely aborted | precision [95% CI] "
       "| false-abort rate [95% CI] | steps saved frac [95% CI] | robot s saved / total | ckpts whose success changed | per-run Spearman(est,full)/top-1 regret |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|")

p1 = r["part1_state"]
print("## data\n", json.dumps(r["data"]))
print("\n## oracle", json.dumps(p1["oracle_abort_all_failures_at_0"]))
print("\n## side check square annotation", json.dumps(r.get("side_check_square_annotation")))
print("\n## nebius call stats", json.dumps(r["part3_nemotron_prefix"].get("call_stats")))
print("\n## Part1 dev->eval\n" + HDR)
for k, v in p1["dev_to_eval"].items():
    params = "; ".join(f"{t}:{c['params']}" for t, c in v["chosen_on_dev"].items())
    print(pol_row(k, v["eval_all"], params))
print("\n## Part1 crossfit by seed\n" + HDR)
for k, v in p1["crossfit_eval_by_seed"].items():
    params = "; ".join(f"{c['task']}/f{c['fit_fold']}:{c['params']}" for c in v["chosen"])
    print(pol_row(k, v["eval_all_crossfit"], params))
print("\n## Part1 curves (eval, fixed rules, no selection)")
print("| task | rule | T (s) | aborted | TP/fail | FP | precision [CI] | recall [CI] | saved frac |\n|---|---|---|---|---|---|---|---|---|")
for key, rows in p1["curves_eval"].items():
    task, rule = key.split("/")
    if rule == "noclose":
        rows = [x for x in rows if x["T_step"] in (40, 60, 80)]
    for x in rows:
        if x["T_step"] in (60, 80, 100, 120, 160, 200, 240, 300):
            print(f"| {task} | {rule} | {x['T_s']:.0f} | {x['n_aborted']} | {x['true_aborts']} | {x['false_aborts']} | "
                  f"{f(x['precision'], 2)} {ci(x['precision_ci95'])} | {f(x['recall'], 2)} {ci(x['recall_ci95'])} | {f(x['saved_frac'], 3)} |")

p2 = r["part2_cosmos_v7"]
print("\n## Part2 v7", json.dumps({k: p2.get(k) for k in ("n_eval_rows", "n_dev_rows", "eval_checkpoints",
                                                          "frame_held_vs_state_lifted_and_grasped",
                                                          "frame_startloc_vs_state_not_lifted_not_succeeded",
                                                          "mean_keyframe_spacing_steps")}))
if "curves_eval_subset" in p2:
    print("| T (s) | staleness succ/fail (steps) | rule | aborted | TP | FP | precision [CI] | recall [CI] | saved |\n|---|---|---|---|---|---|---|---|---|")
    for row in p2["curves_eval_subset"]:
        st = row["mean_staleness_of_latest_keyframe_steps_running"]
        for nm in ("vis_snapshot_loc", "vis_snapshot_loc_and_not_held", "state_snapshot_same_keyframe", "time_only",
                   "LEAKY_vis_nohold_history"):
            x = row[nm]
            print(f"| {row['T_s']:.1f} | {st['successes']}/{st['failures']} | {nm} | {x['n_aborted']} | {x['true_aborts']} | "
                  f"{x['false_aborts']} | {f(x['precision'], 2)} {ci(x['precision_ci95'])} | {f(x['recall'], 2)} {ci(x['recall_ci95'])} | {f(x['saved_frac'], 3)} |")
    d = p2["dev_to_eval"]
    print("\nchosen", json.dumps(d["chosen_on_dev"]))
    print(HDR)
    for k in ("vision_snapshot_rule_eval", "time_only_same_T2_eval", "state_snapshot_same_T1_T2_eval"):
        print(pol_row(k, d[k]))

p3 = r["part3_nemotron_prefix"]
print("\n## Part3 Nemotron; chosen:", p3.get("chosen_variant_on_dev"))
print("| split | variant | cut (s) | subset | n parsed | succ/fail | acc [CI] | bal acc | AUROC [CI] | pred-fail precision [CI] | pred-fail recall [CI] | FN (succ predicted fail) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def nrow(split, v, cut, sub, s):
    print(f"| {split} | {v} | {cut} | {sub} | {s['n_parsed']} | {s['n_success']}/{s['n_failure']} | {f(s['accuracy'], 2)} {ci(s['accuracy_ci95'])} | "
          f"{f(s['balanced_accuracy'], 2)} | {f(s['auroc_ordinal'], 2)} {ci(s['auroc_ci95'])} | {f(s['fail_pred_precision'], 2)} {ci(s['fail_pred_precision_ci95'])} | "
          f"{f(s['fail_pred_recall'], 2)} {ci(s['fail_pred_recall_ci95'])} | {s['fn_pred_failure_but_succeeded']} |")


for v, d in p3.get("dev", {}).items():
    nrow("dev", v, 4, "all", d["all"])
    for t, s in d["by_task"].items():
        nrow("dev", v, 4, t, s)
for c, d in p3.get("eval", {}).items():
    nrow("eval", p3["chosen_variant_on_dev"], d["cut_s"], "all", d["all"])
    for t, s in d["by_task"].items():
        nrow("eval", p3["chosen_variant_on_dev"], d["cut_s"], t, s)
    nrow("eval", p3["chosen_variant_on_dev"], d["cut_s"], "excl. epoch 0", d["excluding_epoch0"])
    print("pred-failure rate by true failure type:", json.dumps(d["by_failure_type_pred_failure_rate"]))
if "eval_policies" in p3:
    ep = p3["eval_policies"]
    print("\npolicies on", ep["n_episodes"], "episodes, cuts", ep["cuts_steps"])
    print(HDR)
    for k, m in ep.items():
        if isinstance(m, dict):
            print(pol_row(k, m))

if "eval_policies_excluding_epoch0" in p3:
    ep = p3["eval_policies_excluding_epoch0"]
    print("\npolicies excluding epoch-0 checkpoints on", ep["n_episodes"], "episodes")
    print(HDR)
    for k, m in ep.items():
        if isinstance(m, dict):
            print(pol_row(k, m))
    print("high-confidence failure predictions:", json.dumps(p3.get("high_conf_failure_predictions")))
