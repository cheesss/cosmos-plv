#!/usr/bin/env python3
"""EXPLORATORY: render markdown tables from results/hackathon/explore/x1_adaptive_budget.json
(used to build docs/hackathon/explore/x1_adaptive_budget.md without hand-copying numbers)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/hackathon/explore/x1_adaptive_budget.json"
d = json.loads(path.read_text())
MAIN = d["protocol"]["main_mults"]
rows = d["primary"]


def cell(j, st, m, key="mean_regret", rs=rows):
    x = [r for r in rs if r["judge"] == j and r["strategy"] == st and r["budget_mult"] == m]
    if not x:
        return "-"
    x = x[0]
    return f"{x[key]:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"


def thr(j, st, key, table="thresholds_primary"):
    x = [t for t in d[table] if t["judge"] == j and t["strategy"] == st]
    if not x or x[0][key] is None:
        return "not reached (>25N)"
    m = x[0][key]
    return f"{m}N ({100 * m / 50:.1f}%)"


print("## data context\n")
print("| run | N | held-out best | #ties at best | #within 0.04 | full-pool (25N) uniform regret = noise floor | paper budget (50N) |")
print("|---|---|---|---|---|---|---|")
for c in d["context"]:
    print(f"| {c['run']} | {c['n_checkpoints']} | {c['heldout_best']:.2f} | {c['n_heldout_best_ties']} | "
          f"{c['n_within_0.04_of_best']} | {c['full_pool_uniform_regret_noise_floor']:.3f} | {c['paper_budget_rollouts']} |")

print("\n## cheap priors (outcome-free) vs held-out success\n")
print("| run | prior | Spearman vs held-out [95% CI] | top-30% epochs | top-30% floor regret | prior-only pick (B=0) regret |")
print("|---|---|---|---|---|---|")
for c in d["context"]:
    for p, v in c["priors"].items():
        print(f"| {c['run'][9:]} | {p} | {v['spearman_vs_heldout']:.2f} [{v['spearman_ci95'][0]:.2f}, {v['spearman_ci95'][1]:.2f}] | "
              f"{v['top30_epochs']} | {v['top30_floor_regret']:.2f} | {v['prior_only_regret']:.2f} |")

print("\n## judge FP split by failure type, and dropped-failure share in eval rollouts\n")
for j, dd in d["judge_counts"].items():
    print(f"- {j}: {dd['by_failure_type']}")
print(f"- dropped share (eval rollouts): {d['dropped_share_eval_rollouts']}")
print("\n## judge error rates (recomputed)\n")
print("| judge | task | FP / true fails | FPR [Wilson95] | FN / true succ | FNR [Wilson95] | 1-FPR-FNR | approx budget inflation 1/(1-FPR-FNR)^2 |")
print("|---|---|---|---|---|---|---|---|")
for j, dd in d["judge_counts"].items():
    for t, c in dd.items():
        if t == "by_failure_type":
            continue
        an = d["analytic_noise_inflation"][j][t]
        print(f"| {j} | {t} | {c['fp']}/{c['n_fail']} | {c['fpr']:.3f} [{c['fpr_wilson95'][0]:.3f}, {c['fpr_wilson95'][1]:.3f}] | "
              f"{c['fn']}/{c['n_succ']} | {c['fnr']:.3f} [{c['fnr_wilson95'][0]:.3f}, {c['fnr_wilson95'][1]:.3f}] | "
              f"{an['signal_factor']:.3f} | {an['approx_budget_inflation']:.2f} |")

STRATS = ["uniform", "successive_halving", "ucb1", "thompson", "top_two_thompson",
          "late_epoch_top30+uniform", "late_epoch_top30+successive_halving", "late_epoch_top30+top_two_thompson",
          "train_action_mse_top30+uniform", "train_action_mse_top30+successive_halving", "train_action_mse_top30+top_two_thompson",
          "val_loss_top30+uniform", "val_loss_top30+successive_halving", "val_loss_top30+top_two_thompson",
          "random_top30+uniform", "random_top30+successive_halving", "random_top30+top_two_thompson"]
for j in ["perfect", "nemotron_nebius_v2", "cosmos3_edge_local_v6", "cascade_local_then_nebius",
          "nemotron_nebius_v2_pessimistic", "cosmos3_edge_local_v6_pessimistic",
          "nemotron_nebius_v2_ftype", "cosmos3_edge_local_v6_ftype"]:
    print(f"\n## mean regret over runs [95% bootstrap CI over orderings], judge = {j}\n")
    print("| strategy | " + " | ".join(f"{m}N ({100 * m / 50:.0f}%)" for m in MAIN) + " |")
    print("|---|" + "---|" * len(MAIN))
    for st in STRATS:
        print(f"| {st} | " + " | ".join(cell(j, st, m) for m in MAIN) + " |")

print("\n## budget needed (mean over runs; grid 0.25..25N; % = share of paper budget 50N)\n")
print("| judge | strategy | mean regret <= 0.02 | CI upper <= 0.02 | mean regret <= 0.05 | excess over full-pool uniform <= 0.02 | re-splits: mean regret <= 0.02 | re-splits: excess over mean floor <= 0.02 |")
print("|---|---|---|---|---|---|---|---|")
for j in ["perfect", "nemotron_nebius_v2", "cosmos3_edge_local_v6", "cascade_local_then_nebius",
          "nemotron_nebius_v2_ftype", "cosmos3_edge_local_v6_ftype"]:
    for st in STRATS:
        xs = [t for t in d["random_splits"]["thresholds"] if t["judge"] == j and t["strategy"] == st]
        rs = "-" if not xs else ("not reached (>25N)" if xs[0]["mult_mean_regret_le_thr"] is None else f"{xs[0]['mult_mean_regret_le_thr']}N")
        rx = "-" if not xs else ("not reached (>25N)" if xs[0].get("mult_excess_vs_mean_full_pool_floor_le_thr") is None
                                 else f"{xs[0]['mult_excess_vs_mean_full_pool_floor_le_thr']}N")
        print(f"| {j} | {st} | {thr(j, st, 'mult_mean_regret_le_thr')} | {thr(j, st, 'mult_ci_upper_le_thr')} | "
              f"{thr(j, st, 'mult_mean_regret_le_0.05')} | {thr(j, st, 'mult_excess_vs_full_pool_le_thr')} | {rs} | {rx} |")

print("\n## per-run mean regret, judge = perfect\n")
KEY = ["uniform", "successive_halving", "top_two_thompson", "late_epoch_top30+top_two_thompson",
       "train_action_mse_top30+top_two_thompson", "val_loss_top30+top_two_thompson"]
runs = [c["run"] for c in d["context"]]
print("| strategy | B | " + " | ".join(r[9:] for r in runs) + " |")
print("|---|---|" + "---|" * len(runs))
for st in KEY:
    for m in MAIN:
        x = [r for r in rows if r["judge"] == "perfect" and r["strategy"] == st and r["budget_mult"] == m][0]
        print(f"| {st} | {m}N | " + " | ".join(f"{p['mean_regret']:.3f} (P[reg=0]={p['p_regret0']:.2f})" for p in x["per_run"]) + " |")

print("\n## robustness: random 25/25 re-splits (mean over splits [bootstrap CI over splits]; split spread 2.5-97.5%)\n")
rsr = d["random_splits"]["rows"]
print(f"n_splits={d['random_splits']['n_splits']}, reps/split={d['random_splits']['reps_per_split']}, "
      f"mean full-pool noise floor over splits = {sum(d['random_splits']['full_pool_noise_floor_per_split']) / len(d['random_splits']['full_pool_noise_floor_per_split']):.3f}\n")
print("| judge | strategy | " + " | ".join(f"{m}N" for m in MAIN) + " |")
print("|---|---|" + "---|" * len(MAIN))
for j in ["perfect", "nemotron_nebius_v2", "cosmos3_edge_local_v6", "nemotron_nebius_v2_ftype", "cosmos3_edge_local_v6_ftype"]:
    for st in KEY + ["thompson", "random_top30+top_two_thompson"]:
        cs = []
        for m in MAIN:
            x = [r for r in rsr if r["judge"] == j and r["strategy"] == st and r["budget_mult"] == m]
            cs.append("-" if not x else f"{x[0]['mean_regret_over_splits']:.3f} [{x[0]['boot_ci95_over_splits'][0]:.3f}, {x[0]['boot_ci95_over_splits'][1]:.3f}] ({x[0]['split_spread_2.5_97.5'][0]:.3f}-{x[0]['split_spread_2.5_97.5'][1]:.3f})")
        print(f"| {j} | {st} | " + " | ".join(cs) + " |")

print("\n## sensitivity: independent (unpaired) episode orders vs paired, judge = perfect / nebius\n")
print("| judge | strategy | " + " | ".join(f"{m}N paired" + " / indep" for m in MAIN) + " |")
print("|---|---|" + "---|" * len(MAIN))
for j in ["perfect", "nemotron_nebius_v2"]:
    for st in ["uniform", "successive_halving", "top_two_thompson", "late_epoch_top30+top_two_thompson"]:
        cs = []
        for m in MAIN:
            a = [r for r in rows if r["judge"] == j and r["strategy"] == st and r["budget_mult"] == m][0]["mean_regret"]
            b = [r for r in d["independent_order"] if r["judge"] == j and r["strategy"] == st and r["budget_mult"] == m][0]["mean_regret"]
            cs.append(f"{a:.3f} / {b:.3f}")
        print(f"| {j} | {st} | " + " | ".join(cs) + " |")
