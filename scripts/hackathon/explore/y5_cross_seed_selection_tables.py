#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- render markdown tables from y5_cross_seed_selection.json.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y5_cross_seed_selection_tables.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAME = "y5_cross_seed_selection"
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / f"results/hackathon/explore/{NAME}.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / f"results/hackathon/explore/{NAME}_tables.txt"
D = json.loads(IN.read_text())

LABEL = {
    "random": "random order (reference)", "epoch": "epoch (ties random)",
    "val_action_l1": "Action L1", "open_loop_smoothness": "open-loop smoothness",
    "plv1_smoothness": "PLV-1 smoothness", "replan_overlap_h16": "replan overlap H=16",
    "boundary_jump_h16": "boundary jump H=16", "val_loss": "validation loss", "train_loss": "train loss",
    "support_drift": "support drift", "dynamics_uncertainty": "dynamics uncertainty",
    "teacher_forced_latent_mse": "teacher-forced latent MSE",
    "k10_sim": "K=10 eval, instrumented", "vlm_local_v6": "K=10 eval, Cosmos3-Edge v6",
    "vlm_local_v7": "K=10 eval, Cosmos3-Edge v7", "vlm_nebius_v2": "K=10 eval, Nemotron v2",
    "vlm_nebius_v2_pad201": "K=10 eval, Nemotron v2 pad201",
    "k25_rollouts_fit_half": "K=25 train rollouts (fit half)",
    "fit_half_best_cheap": "cheap metric chosen on fit half",
    "fit_half_best_any": "any outcome-free score chosen on fit half",
    "rankavg_epoch+k25_fit_half": "rank-avg(epoch, K=25 fit half)",
}
MAIN = ["val_action_l1", "open_loop_smoothness", "replan_overlap_h16", "plv1_smoothness", "boundary_jump_h16"]
CHEAP_ALL = MAIN + ["val_loss", "train_loss", "support_drift", "dynamics_uncertainty", "teacher_forced_latent_mse"]


def lab(m):
    if m in LABEL:
        return LABEL[m]
    if m.startswith("rankavg_epoch+"):
        return f"rank-avg(epoch, {lab(m[len('rankavg_epoch+'):])})"
    if m.startswith("rankavg_l1+"):
        return f"rank-avg(Action L1, {lab(m[len('rankavg_l1+'):])})"
    if m.startswith("lex_epoch>"):
        return f"epoch, ties -> {lab(m[len('lex_epoch>'):])}"
    if m.startswith("lex_") and m.endswith(">l1"):
        return f"{lab(m[4:-3])}, ties -> Action L1"
    return m


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def ci(r, k, d=3):
    lo, hi = r[f"{k}_ci95"]
    return f"{f(r[k], d)} [{f(lo, d)}, {f(hi, d)}]"


def dci(r, k, d=3):
    if f"{k}_minus_epoch" not in r:
        return "—"
    lo, hi = r[f"{k}_minus_epoch_ci95"]
    return f"{r[f'{k}_minus_epoch']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def sel_table(pool, target, methods):
    res = D["results"][pool][target]
    n = res["epoch"]["n"]
    out = [f"Pool `{pool}` (N={n}), target `{target}`. Point estimate [95% block x episode bootstrap CI].",
           "",
           "| score | Spearman | ΔSpearman vs epoch | Regret@10% | Regret@30% | top-1 regret | Δtop-1 vs epoch |",
           "|---|---|---|---|---|---|---|"]
    for m in methods:
        if m not in res:
            continue
        r = res[m]
        out.append(f"| {lab(m)} | {ci(r, 'spearman')} | {dci(r, 'spearman')} | {ci(r, 'regret@10')} | "
                   f"{ci(r, 'regret@30')} | {ci(r, 'top1_regret')} | {dci(r, 'top1_regret')} |")
    return "\n".join(out)


def pw_table(key, methods):
    pw = D["pairwise"][key]
    out = [f"`{key}`: {pw['n_pairs_passing_filter']} pairs pass the filter (epochs {pw['epochs']}); "
           f"{pw['n_target_ties_excluded']} excluded because the scoring half ties; scored pairs = {pw['n_pairs']}; "
           f"seed2 better in {f(pw['frac_seed2_better'], 2)} of them.",
           "",
           "| picker | scored pairs | method ties | correct | accuracy (ties=0.5) | 95% CP CI (excl. ties) | binom p vs 0.5 | picks seed2 |",
           "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        r = pw["per_method"].get(m)
        if r is None:
            continue
        lo, hi = r["cp95_excl_ties"]
        out.append(f"| {lab(m)} | {r['n']} | {r['n_method_ties(scored_0.5)']} | {r['n_correct']} | "
                   f"{f(r['accuracy_ties_half'], 2)} | [{f(lo, 2)}, {f(hi, 2)}] | {f(r['binom_p_vs_0.5'], 3)} | "
                   f"{f(r['frac_pick_seed2'], 2)} |")
    return "\n".join(out)


def main():
    parts = []
    k10 = ["k10_sim", "vlm_nebius_v2", "vlm_nebius_v2_pad201", "vlm_local_v6", "vlm_local_v7"]
    base = ["random", "epoch"] + CHEAP_ALL + [f"rankavg_epoch+{m}" for m in MAIN + ["teacher_forced_latent_mse"]] + \
           [f"lex_epoch>{m}" for m in ["val_action_l1", "open_loop_smoothness", "replan_overlap_h16",
                                       "teacher_forced_latent_mse"]]
    parts.append("### T1 common10\n\n" + sel_table("common10", "S50", base))
    short = ["random", "epoch"] + MAIN + ["val_loss", "teacher_forced_latent_mse",
                                          "rankavg_epoch+val_action_l1", "rankavg_epoch+open_loop_smoothness",
                                          "lex_epoch>val_action_l1"]
    parts.append("### T2 all\n\n" + sel_table("all", "S50", short))
    parts.append("### T3 late_common10\n\n" + sel_table("late_common10", "S50", short))
    ev = ["random", "epoch"] + MAIN[:4] + ["teacher_forced_latent_mse", "rankavg_epoch+val_action_l1"] + k10 + \
         [f"rankavg_epoch+{k}" for k in ("k10_sim", "vlm_nebius_v2", "vlm_local_v6")] + \
         [f"rankavg_l1+{k}" for k in ("k10_sim", "vlm_nebius_v2", "vlm_local_v6")] + \
         [f"lex_{k}>l1" for k in ("k10_sim", "vlm_nebius_v2", "vlm_local_v6")]
    parts.append("### T4 eval20\n\n" + sel_table("eval20", "S50", ev))
    ho = ["random", "epoch", "val_action_l1", "open_loop_smoothness", "rankavg_epoch+val_action_l1",
          "k25_rollouts_fit_half", "fit_half_best_cheap", "fit_half_best_any", "rankavg_epoch+k25_fit_half"]
    for pool in ("common10", "late_common10", "eval20"):
        extra = ["k10_sim", "vlm_nebius_v2"] if pool == "eval20" else []
        for t in ("HB", "HA"):
            parts.append(f"### T5 held-out {pool} {t}\n\n" + sel_table(pool, t, ho + extra))
            sp = D["fit_half_selection_point"][pool][t]
            fr = D["fit_half_selection_bootstrap_freq"][pool][t]
            top = {k: sorted(v.items(), key=lambda kv: -kv[1])[:3] for k, v in fr.items()}
            parts.append(f"selected on the fit half (point): {sp}; bootstrap top-3: {top}")
    rel = D["split_half_reliability"]
    parts.append("### T6 split-half reliability of the target\n\n| pool | Spearman(HA, HB) | Spearman-Brown (50 eps) |\n|---|---|---|\n" +
                 "\n".join(f"| {p} | {f(v['spearman_halfA_halfB'])} | {f(v['spearman_brown_full50'])} |" for p, v in rel.items()))
    pwm = CHEAP_ALL + ["epoch_method"] + k10
    for key in ("primary_S50_thr0.06", "any_difference_S50", "filter_HA_score_HB", "filter_HB_score_HA"):
        parts.append(f"### T7 pairwise {key}\n\n" + pw_table(key, pwm))
    pw = D["pairwise"]
    parts.append("### T8 pairwise controls\n\n" + json.dumps({
        "target_split_half_sign_agreement_primary_pairs": pw["target_split_half_sign_agreement_primary_pairs"],
        "k25": {k: v for k, v in pw.items() if k.startswith("k25_")},
        "constant_pick_primary": pw["constant_pick_controls_primary"],
        "constant_pick_any_difference": pw["constant_pick_controls_any_difference"],
        "circular_shift_any_difference": {m: pw["circular_shift_null_any_difference"][m] for m in CHEAP_ALL + k10},
        "holm_any_difference": pw["holm_adjusted_p_any_difference_cheap"],
        "heldout_selected_pairwise_metric": pw["heldout_selected_pairwise_metric"],
    }, indent=1))
    pdc = pw["pair_difference_spearman"]
    parts.append("### T9 within-pair difference correlation (all 20 non-trivial common epochs; K=10 rows: 10 eval epochs)\n\n"
                 "| picker | pairs | Spearman(Δcost, ΔS50) [episode-boot CI] | Spearman(Δcost, ΔHB) |\n|---|---|---|---|\n" +
                 "\n".join(f"| {lab(m)} | {v['n_pairs']} | {f(v['spearman_dcost_vs_dS50'])} [{f(v['ci95_episode_boot'][0])}, "
                           f"{f(v['ci95_episode_boot'][1])}] | {f(v['spearman_dcost_vs_dHB'])} |"
                           for m, v in pdc.items() if m in CHEAP_ALL + k10))
    rows = D["pair_table"]
    lines = ["| epoch | S50 seed1 | S50 seed2 | ΔS50 | ΔHA | ΔHB | K10 sim s1/s2 | Nemotron v2 s1/s2 | Cosmos v6 s1/s2 | Action L1 picks | TF-latent picks |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        k = (f"{r['k10_sim_seed1']:.1f}/{r['k10_sim_seed2']:.1f}" if "k10_sim_seed1" in r else "—")
        n = (f"{r['vlm_nebius_v2_seed1']:.1f}/{r['vlm_nebius_v2_seed2']:.1f}" if "vlm_nebius_v2_seed1" in r else "—")
        c = (f"{r['vlm_local_v6_seed1']:.1f}/{r['vlm_local_v6_seed2']:.1f}" if "vlm_local_v6_seed1" in r else "—")
        pk = {0.0: "s1", 1.0: "s2", 0.5: "tie"}
        lines.append(f"| {r['epoch']} | {r['S50_seed1']:.2f} | {r['S50_seed2']:.2f} | {r['dS50(s2-s1)']:+.2f} | {r['dHA']:+.2f} | "
                     f"{r['dHB']:+.2f} | {k} | {n} | {c} | {pk[r['val_action_l1_pick_seed2']]} | {pk[r['teacher_forced_latent_mse_pick_seed2']]} |")
    parts.append("### T10 epoch-matched pairs\n\n" + "\n".join(lines))
    parts.append("### T11 seed-level late\n\n" + json.dumps(D["seed_level_late"], indent=1))
    parts.append("### T12 tie check\n\n" + json.dumps(D["tie_handling_check_common10_S50_epoch"], indent=1))
    txt = "\n\n".join(parts) + "\n"
    OUT.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
