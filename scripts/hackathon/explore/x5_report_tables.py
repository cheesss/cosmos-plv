#!/usr/bin/env python3
"""Print markdown tables from results/hackathon/explore/x5_cosmos_modes.json (for the report)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
res = json.loads((ROOT / "results/hackathon/explore/x5_cosmos_modes.json").read_text())


def ci(c):
    return f"[{c[0]}, {c[1]}]" if c and c[0] is not None else "n/a"


def row(name, m, ft=True):
    s = (f"| {name} | {m['episodes']} | {m['unparsed']} | {m['success_acc_all']} {ci(m['success_acc_all_ci95'])} "
         f"| {m['balanced_acc_all']} | {m['n_true_failure'] - m['tn']}/{m['n_true_failure']} {ci(m['fp_rate_ci95'])} "
         f"| {m['n_true_success'] - m['tp']}/{m['n_true_success']} ")
    if ft and m.get("failure_type_acc_on_failures_all") is not None:
        s += (f"| {m['failure_type_acc_on_failures_all']} {ci(m['failure_type_acc_ci95'])} "
              f"| {m['failure_type_acc_square_other_as_no_grasp']} | {m['lifted_acc_scored']} ")
    else:
        s += "| - | - | - "
    s += f"| {m['mean_latency_s_per_episode']} |"
    return s


HDR = ("| variant | episodes | unparsed | success acc (all) [95% Wilson] | balanced acc | false success on failures "
       "[CI] | missed successes | failure-type acc on failures [CI] | ft acc (square other=no_grasp) | lifted acc "
       "| latency s/episode |\n|---|---|---|---|---|---|---|---|---|---|---|")

for split in ("dev", "eval"):
    rep = res[split]
    if not rep:
        continue
    print(f"\n### {split}: each variant vs v6 on the same episodes\n")
    print(HDR)
    for v, e in rep.items():
        if "vs_v6_same_episodes" in e:
            x = e["vs_v6_same_episodes"]
            print(row(f"{v} (n={x['n']})", x["variant"], v != "d_reason_last"))
            print(row(f"  v6 same eps", x["v6"], False))
            p = x["paired_success_diff_variant_minus_v6"]
            print(f"|  paired success diff {v} - v6 | {p['n']} | | {p['diff']} {ci(p['diff_ci95_boot'])} boot "
                  f"| | {p['a_right_b_wrong']} fixed / {p['a_wrong_b_right']} broken | | | | | |")
    print(f"\n### {split}: failure-type variants vs v7 on the same episodes\n")
    print(HDR)
    for v, e in rep.items():
        if "vs_v7_same_episodes" in e:
            x = e["vs_v7_same_episodes"]
            print(row(f"{v} (n={x['n']})", x["variant"]))
            print(row("  v7 same eps", x["v7"]))
            p = x["paired_ftype_diff_variant_minus_v7"]
            print(f"|  paired failure-type diff {v} - v7 | {p.get('n')} failures | | {p.get('diff')} "
                  f"{ci(p.get('diff_ci95_boot'))} boot | | {p.get('a_right_b_wrong')} fixed / "
                  f"{p.get('a_wrong_b_right')} broken | | | | | |")
    print(f"\n### {split}: failure-type confusion (truth->pred) on own episodes\n")
    for v, e in rep.items():
        c = e["own_episodes"].get("failure_confusion")
        if c:
            print(f"- {v}: " + ", ".join(f"{k} {n}" for k, n in c.items()))

fl = res.get("frame_level", {})
for split in ("dev", "eval"):
    if not fl.get(split):
        continue
    print(f"\n### {split}: per-keyframe 'held' perception vs simulator (held = contact and >= 2 cm above rest)\n")
    print("| source | frames | held precision [CI] | held recall [CI] | false-held rate on not-held frames |")
    print("|---|---|---|---|---|")
    for k, d in fl[split].items():
        if "held_tp" in d:
            print(f"| {k} | {d['frames']} | {d['held_precision']} {ci(d['held_precision_ci95'])} | "
                  f"{d['held_recall']} {ci(d['held_recall_ci95'])} | {d['held_fp_rate_on_not_held']} |")
        else:
            print(f"| {k} | {d['pairs']} pairs | up precision {d['up_precision']} | up recall {d['up_recall']} "
                  f"| tp {d['up_tp']} fp {d['up_fp']} fn {d['up_fn']} |")

print("\n### per-variant call stats\n")
print("| variant | calls | unparsed | mean latency/call s | mean prompt tokens | mean completion tokens |")
print("|---|---|---|---|---|---|")
for v, d in res["per_variant_call_stats"].items():
    print(f"| {v} | {d['calls']} | {d['unparsed']} | {d['mean_latency_per_call_s']} | {d['mean_prompt_tokens']} "
          f"| {d['mean_completion_tokens']} |")
sel = res.get("selection_in_run") or {}
print("\nselection in run:", json.dumps({k: sel.get(k) for k in ("success_choice", "ftype_choice", "n_dev_episodes")}))
print("dev scores used for selection:", json.dumps(sel.get("dev_scores"), indent=1))
