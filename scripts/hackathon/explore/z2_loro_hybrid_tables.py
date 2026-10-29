#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- render tables for z2_loro_hybrid from its JSON.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/z2_loro_hybrid_tables.py [results.json] [out.txt]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/hackathon/explore/z2_loro_hybrid.json"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/hackathon/explore/z2_loro_hybrid_tables.txt"
R = json.loads(src.read_text())
RUNS = ["can_seed1", "can_seed2", "square_seed1"]
L = []


def f(x, d=3):
    return f"{x:+.{d}f}" if x == x else "nan"


def ci(c, d=3):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


for pool, P in R["pools"].items():
    M = P["methods"]
    L.append(f"\n## pool {pool}  (N={P['n_checkpoints']}, per run {P['per_run_n']}, block len {P['block_len_ckpts']})\n")
    L.append("| method | kind | held-out rho (HB) | per-run rho (can1, can2, sq) | d rho vs epoch | run-level CI of d | block-boot CI of d | P(d>0) | runs better/worse | Regret@30 | top-1 regret | d top-1, block CI | swap rho (fit HB, eval HA) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(M, key=lambda n: -M[n]["heldout_HB"]["spearman"])
    for n in order:
        d = M[n]
        rl = d["run_level"]["spearman"]
        bb = d["block_boot"]["spearman"]
        t1 = d["block_boot"]["top1_regret"]
        pr = ", ".join(f"{d['per_run'][h]['spearman']:.3f}" for h in RUNS)
        L.append(f"| {n} | {d['kind']} | {d['heldout_HB']['spearman']:.3f} {ci(bb['ci'])} | {pr} | {f(rl['delta_vs_epoch'])} | "
                 f"{ci(rl['delta_ci'])} | {ci(bb['delta_vs_epoch_ci'])} | {bb['p_delta_gt0']:.2f} | "
                 f"{rl['n_runs_better_than_epoch']}/{rl['n_runs_worse_than_epoch']} | "
                 f"{d['heldout_HB']['regret@30']:.3f} | {d['heldout_HB']['top1_regret']:.3f} | "
                 f"{f(d['run_level']['top1_regret']['delta_vs_epoch'])} {ci(t1['delta_vs_epoch_ci'])} | "
                 f"{d['swap_fit_HB_eval_HA']['spearman']:.3f} |")
    # weights
    L.append(f"\n### {pool}: learned weights (normalised to unit length; nested lambda) and stability\n")
    WS = P["weight_stability"]
    WB = P.get("weight_bootstrap", {})
    for n, ws in WS.items():
        feats = ws["feats"]
        L.append(f"**{n}**  lambda chosen {ws['lam_chosen']}; mean cross-fold cosine (nested) {ws['w']['mean_cosine']:.3f}, "
                 f"(fixed lambda) {ws['w_fixed_lam']['mean_cosine']:.3f}\n")
        L.append("| feature | " + " | ".join(f"w {h}" for h in RUNS) + " | sign agrees in 3 folds | boot frac same sign (can1, can2, sq) |")
        L.append("|---|" + "---|" * (len(RUNS) + 2))
        for ft in feats:
            ws_ = " | ".join(f"{ws['w']['per_fold_normalised'][h][ft]:+.3f}" for h in RUNS)
            bs = ", ".join(f"{WB[n][h]['frac_same_sign_as_point'][ft]:.2f}" for h in RUNS) if n in WB else "-"
            L.append(f"| {ft} | {ws_} | {ws['w']['sign_agree_all_folds'][ft]} | {bs} |")
        L.append("")
    # selection / sensitivity
    fd = P["fold_details"]
    L.append(f"\n### {pool}: select_best_on_train choices\n")
    for h in RUNS:
        L.append(f"* held-out {h}: chose `{fd[h]['select_best_on_train']['chosen']}` (train rho {fd[h]['select_best_on_train']['train_rho']:.3f})")
    L.append(f"\n### {pool}: post-hoc lambda sensitivity (mean held-out rho over folds at each FIXED lambda; optimistic upper bound)\n")
    lams = None
    rows = []
    for n, d in fd[RUNS[0]].items():
        if "heldout_by_lam" not in d:
            continue
        lams = list(d["heldout_by_lam"].keys())
        vals = [sum(fd[h][n]["heldout_by_lam"][l]["spearman"] for h in RUNS) / 3 for l in lams]
        rows.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {max(vals):.3f} |")
    L.append("| method | " + " | ".join(f"lam={l}" for l in lams) + " | best |")
    L.append("|---|" + "---|" * (len(lams) + 1))
    L.extend(rows)
    L.append(f"\n(epoch: {M['epoch']['heldout_HB']['spearman']:.3f})")

L.append("\n## noise ceiling: within-run Spearman(HA success, HB success)\n")
for pool, d in R["noise_ceiling_HA_vs_HB"].items():
    L.append(f"* {pool}: " + ", ".join(f"{h} {v:.3f}" for h, v in d.items()))
dst.write_text("\n".join(L) + "\n")
print(f"wrote {dst}")
