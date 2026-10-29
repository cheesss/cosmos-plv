#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- print the markdown tables of y2_plateau_noise_ceiling from its JSON.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y2_plateau_noise_ceiling_tables.py > <out.md>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
J = json.loads((ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling.json").read_text()
               if len(sys.argv) < 2 else Path(sys.argv[1]).read_text())

SUBS = ["all", "late_median", "plateau_posthoc"]
ORDER_ALL = ["epoch", "val_loss", "train_loss", "train_action_mse", "val_action_l1", "plv1_smoothness",
             "support_drift", "dynamics_uncertainty", "teacher_forced_latent_mse", "open_loop_smoothness",
             "boundary_jump_h16", "replan_overlap_h16", "train_rollouts_otherhalf_K10",
             "train_rollouts_otherhalf_K25", "_random"]
ORDER_EVAL = ["epoch", "val_loss", "val_action_l1", "plv1_smoothness", "open_loop_smoothness",
              "replan_overlap_h16", "sim_replan_overlap_K10", "sim_boundary_jump_K10",
              "executed_smoothness_K10", "eval_sim_success_K10", "vlm_cosmos_local_v6_K10",
              "vlm_cosmos_local_v7_K10", "vlm_nemotron_nebius_v2_K10", "vlm_nemotron_nebius_v2_pad201_K10",
              "train_rollouts_otherhalf_K10", "train_rollouts_otherhalf_K25", "_random"]


def f(x, d=2):
    return "--" if x is None else f"{x:+.{d}f}" if d == 2 else f"{x:.{d}f}"


def ci(c, d=2):
    return "--" if not c else f"[{c['lo']:+.{d}f}, {c['hi']:+.{d}f}]"


def ci3(c):
    return "--" if not c else f"[{c['lo']:.3f}, {c['hi']:.3f}]"


def noise_table():
    print("| subset | run | n | S50 min-max | split-half rho (fixed) | split-half rho (random splits, mean [2.5,97.5]) "
          "| SB rel50 | ceiling vs 25 | ceiling vs 50 | ANOVA rel25 / rel50 | ANOVA F p | perm. heterog. p "
          "| epoch-trend rho(S50) (p) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in [s + suf for s in SUBS for suf in ("", "__eval_ckpts")]:
        nc = J["noise_ceiling"][key]
        for run, r in nc["per_run"].items():
            print(f"| {key} | {run} | {r['n_checkpoints']} | {r['success50_min']:.2f}-{r['success50_max']:.2f} | "
                  f"{f(r['fixed_split_spearman_A_B'])} | {f(r['random_split_spearman_mean'])} "
                  f"[{f(r['random_split_spearman_p2_5'])}, {f(r['random_split_spearman_p97_5'])}] | "
                  f"{f(r['spearman_brown_reliability_50'])} | {f(r['ceiling_vs_25_heldout'])} | {f(r['ceiling_vs_50'])} | "
                  f"{f(r['anova_reliability_25'])} / {f(r['anova_reliability_50'])} | {r['anova_F_p']:.3f} | "
                  f"{r['perm_test_heterogeneity_p']:.3f} | {f(r['perm_test_epoch_trend_spearman'])} "
                  f"({'--' if r['perm_test_epoch_trend_p_onesided'] is None else format(r['perm_test_epoch_trend_p_onesided'], '.3f')}) |")
        m = nc["mean_over_runs"]
        print(f"| **{key}** | **mean** | {m['n_checkpoints']} | | {f(m['fixed_split_spearman_A_B'])} | "
              f"**{f(m['random_split_spearman_mean'])}** [{f(m['random_split_spearman_p2_5'])}, "
              f"{f(m['random_split_spearman_p97_5'])}] | {f(m['spearman_brown_reliability_50_mean_of_runs'])} | "
              f"**{f(m['ceiling_vs_25_heldout_mean_of_runs'])}** | **{f(m['ceiling_vs_50_mean_of_runs'])}** | "
              f"{f(m['anova_reliability_25_mean_of_runs'])} / {f(m['anova_reliability_50_mean_of_runs'])} | | | |")


def signal_table(key, order):
    sd = J["signals"][key]
    nc = J["noise_ceiling"][key]["mean_over_runs"]
    print(f"\n**{key}** (n = {nc['n_checkpoints']} checkpoints; noise ceiling vs held-out 25 = "
          f"{f(nc['ceiling_vs_25_heldout_mean_of_runs'])}; constant-target runs skipped: "
          f"{sd['_target_constant_runs'] or 'none'})\n")
    print("| signal | Spearman vs B | hier. CI | run-level CI | mean over random splits | Regret@30 vs B | hier. CI "
          "| Spearman vs S50 |")
    print("|---|---|---|---|---|---|---|---|")
    for sig in order:
        if sig not in sd:
            continue
        d = sd[sig]
        sp, rg = d["spearman"], d["regret30"]
        print(f"| {sig} | {f(sp['point'])} | {ci(sp['ci_hier_block_x_seed'])} | {ci(sp['ci_run_level_exact'])} | "
              f"{f(sp.get('random_splits_mean'))} | {f(rg['point'], 3)} | {ci3(rg['ci_hier_block_x_seed'])} | "
              f"{f(d.get('spearman_vs_full50'))} |")


def incr_table(key, order):
    sd = J["signals"][key]
    print(f"\n**{key}**\n")
    print("| signal | partial Spearman given epoch | hier. CI | block CI | run-level CI | "
          "d Spearman rankavg(epoch,signal) - epoch | hier. CI | P(d>0) | d Regret@30 rankavg - epoch | hier. CI |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for sig in order:
        if sig not in sd or sig in ("epoch", "_random"):
            continue
        d = sd[sig]
        p, ds, dr = d["partial_given_epoch"], d["d_rankavg_spearman_vs_epoch"], d["d_rankavg_regret30_vs_epoch"]
        pg = ds["ci_hier_block_x_seed"]
        print(f"| {sig} | {f(p['point'])} | {ci(p['ci_hier_block_x_seed'])} | {ci(p['ci_block_only'])} | "
              f"{ci(p['ci_run_level_exact'])} | {f(ds['point'])} | {ci(pg)} | "
              f"{'--' if not pg else format(pg['p_gt0'], '.2f')} | {f(dr['point'], 3)} | "
              f"{ci3(dr['ci_hier_block_x_seed'])} |")


if __name__ == "__main__":
    print("## Noise ceiling\n")
    noise_table()
    print("\n## Signals vs held-out half B\n")
    for s in SUBS:
        signal_table(s, ORDER_ALL)
    for s in SUBS:
        signal_table(s + "__eval_ckpts", ORDER_EVAL)
    print("\n## Incremental value beyond epoch\n")
    for s in SUBS:
        incr_table(s, ORDER_ALL)
    for s in SUBS:
        incr_table(s + "__eval_ckpts", ORDER_EVAL)
