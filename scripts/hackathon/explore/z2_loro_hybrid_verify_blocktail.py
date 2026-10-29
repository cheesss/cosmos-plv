#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- z2 verification: where does the block-bootstrap left tail come from?

Re-runs z2's own block bootstrap (imports z2_loro_hybrid.py; same RNG scheme) for a
subset of methods and records, per replicate and fold, whether the resampled
training runs / held-out run contain their earliest checkpoints (epoch <= 10% of
max) and how many distinct checkpoints survived.  CPU only.
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z2_loro_hybrid_verify_blocktail.py
"""
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import z2_loro_hybrid as Z  # noqa: E402

ROOT = Z.ROOT
ck, Y, E, V, has = Z.load_data()
runs = ck.run_id.values
rows = np.ones(len(ck), bool)
S_all = Z.method_specs("all93")
specs = {k: S_all[k] for k in ["epoch", "pairlogit_rank[all]", "ridge_rank[all]", "ridge_rank[small]", "RA(epoch,val_action_l1)"]}
NR = Z._features(ck, runs, rows, E, V)
base_idx = {r: np.where(runs == r)[0] for r in Z.RUNS}
Lb = {"can_seed1": 8, "can_seed2": 4, "square_seed1": 4}
seed = 20261029
out = []
for b in range(400):
    rng = np.random.default_rng([seed, b])           # identical stream to z2 _boot_one
    ia = rng.choice(Z.HA, size=len(Z.HA), replace=True)
    ib = rng.choice(Z.HB, size=len(Z.HB), replace=True)
    rng.integers(0, 10, size=10)
    yA, yB = Y[:, ia].mean(1), Y[:, ib].mean(1)
    idx = {r: base_idx[r][Z.block_resample(len(base_idx[r]), Lb[r], rng)] for r in Z.RUNS}
    early = {r: bool((ck.epoch.values[idx[r]] <= 0.1 * ck.epoch.values[base_idx[r]].max()).any()) for r in Z.RUNS}
    zero_succ = {r: bool((yA[idx[r]] == 0).any()) for r in Z.RUNS}
    for h in Z.RUNS:
        res, _ = Z.run_fold(specs, NR, runs, yA, yB, idx, h)
        tr = [r for r in Z.RUNS if r != h]
        out.append({"b": b, "hold": h, "train_both_have_early": all(early[r] for r in tr),
                    "train_any_missing_early": not all(early[r] for r in tr),
                    "heldout_has_early": early[h],
                    "n_distinct_heldout": int(len(np.unique(idx[h]))),
                    **{f"d_{k}": res[k]["spearman"] - res["epoch"]["spearman"] for k in specs if k != "epoch"},
                    "epoch_rho": res["epoch"]["spearman"]})
import pandas as pd
df = pd.DataFrame(out)
summ = {}
for k in [c for c in df.columns if c.startswith("d_")]:
    g = {}
    for lab, m in [("train_both_have_early", df.train_both_have_early), ("train_any_missing_early", df.train_any_missing_early),
                   ("heldout_missing_early", ~df.heldout_has_early)]:
        sub = df[m][k].dropna()
        g[lab] = {"n": int(len(sub)), "mean": float(sub.mean()), "q05": float(sub.quantile(.05)),
                  "frac_lt_-0.1": float((sub < -0.1).mean())}
    allv = df[k].dropna()
    g["all_folds"] = {"n": int(len(allv)), "mean": float(allv.mean()), "q05": float(allv.quantile(.05)),
                      "frac_lt_-0.1": float((allv < -0.1).mean())}
    summ[k] = g
summ["frac_folds_train_missing_early"] = float(df.train_any_missing_early.mean())
summ["frac_folds_heldout_missing_early"] = float((~df.heldout_has_early).mean())
summ["epoch_rho_mean_heldout_has_early"] = float(df[df.heldout_has_early].epoch_rho.mean())
summ["epoch_rho_mean_heldout_missing_early"] = float(df[~df.heldout_has_early].epoch_rho.mean())
p = ROOT / "results/hackathon/explore/z2_loro_hybrid_verification_blocktail.json"
p.write_text(json.dumps({"label": "EXPLORATORY / post-hoc; z2 block bootstrap re-run (400 replicates, all93), per-fold", "summary": summ}, indent=1))
print(json.dumps(summ, indent=1))
