#!/usr/bin/env python3
"""Markdown tables for the y3_visual_robustness report, generated from the results JSON
(so that no number is typed by hand).

  python scripts/hackathon/explore/y3_tables.py > data/outputs/hackathon_explore/y3_visual_robustness/tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
res = json.loads((ROOT / "results/hackathon/explore/y3_visual_robustness.json").read_text())


def ci(x):
    return "n/a" if not x or x[0] is None else f"[{x[0]:.2f}, {x[1]:.2f}]"


def f(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


for backend, b in res["backends"].items():
    print(f"\n#### {backend}: {b['judge']}\n")
    print("| condition | n | parsed | acc | Wilson 95% | cluster-boot 95% | bal. acc | cluster-boot 95% | TPR (succ) | TNR (fail) | acc can | acc square |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c, s in b["conditions"].items():
        if not s:
            continue
        pt = s["per_task"]
        print(f"| {c} | {s['n']} | {s['n_parsed']} | {f(s['acc'])} | {ci(s['acc_wilson95'])} | {ci(s['acc_cluster_boot95'])} | "
              f"{f(s['bal_acc'])} | {ci(s['bal_acc_cluster_boot95'])} | {f(s['tpr'],2)} {ci(s['tpr_wilson95'])} | "
              f"{f(s['tnr'],2)} {ci(s['tnr_wilson95'])} | {f(pt.get('can',{}).get('acc'),2)} | {f(pt.get('square',{}).get('acc'),2)} |")
    print("\nPaired differences (condition minus control, same episodes; paired bootstrap clustered by checkpoint):\n")
    print("| condition | control | pairs | d acc | 95% CI | d bal. acc | 95% CI | ctrl right / cond wrong | cond right / ctrl wrong | McNemar p (episode-level) | same prediction |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in b["paired_vs_control"]:
        print(f"| {p['condition']} | {p['control']} | {p['n_pairs']} | {p['d_acc']:+.3f} | {ci(p['d_acc_paired_cluster_boot95'])} | "
              f"{'n/a' if p['d_bal_acc'] != p['d_bal_acc'] else format(p['d_bal_acc'], '+.3f')} | {ci(p['d_bal_acc_paired_cluster_boot95'])} | "
              f"{p['ctrl_right_cond_wrong']} | {p['cond_right_ctrl_wrong']} | {p['mcnemar_exact_p_episode_level']:.3f} | {p['same_prediction_frac']:.2f} |")

print("\n#### Predicted labels by condition (task | truth | prediction: count)\n")
for backend, b in res["backends"].items():
    for c, s in b["conditions"].items():
        if s:
            print(f"- {backend} / {c}: " + ", ".join(f"{k}: {v}" for k, v in s["pred_label_counts"].items()))
