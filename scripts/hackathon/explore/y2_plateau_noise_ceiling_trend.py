#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- y2_plateau_noise_ceiling, supplementary trend check.

How large is the systematic (epoch) trend of success inside each subset compared with
the per-checkpoint rollout noise?  Per run and subset: OLS slope of success vs epoch
(percentage points per 100 epochs) for S50, half A and half B, with a circular
block-bootstrap CI (checkpoints) and a seed-bootstrap CI (episodes); Spearman of
epoch with half A and with half B separately; per-checkpoint binomial noise SD.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y2_plateau_noise_ceiling_trend.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import y2_plateau_noise_ceiling as y2  # noqa: E402

OUT = y2.ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling_trend.json"


def slope(ep, s):
    return float(np.polyfit(ep, s, 1)[0] * 100 * 100)  # pp per 100 epochs


def trend_anova(Y, ep, rng, n_sim=4000):
    """Two-way (checkpoint x shared seed) ANOVA with the checkpoint factor split into a linear
    epoch trend (1 df) and checkpoint-specific deviations (n-2 df).  MSe = interaction.
    Returns F tests, the reliability of the deviations for a 25-rollout half (= ceiling^2 for
    the partial Spearman of ANY metric given epoch), and a Gaussian simulation of the expected
    Spearman of a perfect trend follower (~ epoch) vs a fresh 25-rollout half."""
    from scipy.stats import f as fdist
    n, k = Y.shape
    rm, cm, g = Y.mean(1), Y.mean(0), Y.mean()
    resid = Y - rm[:, None] - cm[None, :] + g
    mse = float((resid ** 2).sum()) / ((n - 1) * (k - 1))
    fit = np.polyval(np.polyfit(ep, rm, 1), ep)
    ss_trend = k * float(((fit - g) ** 2).sum())
    ss_dev = k * float(((rm - fit) ** 2).sum())
    ms_dev = ss_dev / (n - 2)
    dfe = (n - 1) * (k - 1)
    var_dev_true = (ms_dev - mse) / k
    noise25 = mse / 25
    rel_dev25 = var_dev_true / (var_dev_true + noise25) if var_dev_true > 0 else 0.0
    sims_t, sims_all = [], []
    for _ in range(n_sim):
        b = fit + rng.normal(0, np.sqrt(noise25), size=n)
        sims_t.append(y2.spear(fit, b))
        if var_dev_true > 0:
            truth = fit + rng.normal(0, np.sqrt(var_dev_true), size=n)
            sims_all.append(y2.spear(truth, truth + rng.normal(0, np.sqrt(noise25), size=n)))
    return {"MSe_interaction": mse, "F_trend": ss_trend / mse if mse > 0 else None,
            "F_trend_p": float(fdist.sf(ss_trend / mse, 1, dfe)) if mse > 0 else None,
            "F_dev": ms_dev / mse if mse > 0 else None,
            "F_dev_p": float(fdist.sf(ms_dev / mse, n - 2, dfe)) if mse > 0 else None,
            "var_dev_true_pp2": var_dev_true * 1e4, "noise_sd_25_shared_seed_pp": float(np.sqrt(noise25) * 100),
            "trend_sd_pp": float(np.std(fit) * 100),
            "rel_dev_25": rel_dev25, "ceiling_partial_given_epoch_vs_25": float(np.sqrt(max(rel_dev25, 0))),
            "sim_spearman_trend_follower_vs_25": float(np.nanmean(sims_t)),
            "sim_spearman_oracle_trend_plus_dev_vs_25": float(np.nanmean(sims_all)) if sims_all else None}


def main(n_boot=2000, seed=7):
    rng = np.random.default_rng(seed)
    runs, _ = y2.load()
    out = {"name": "y2_plateau_noise_ceiling_trend", "exploratory": True, "post_hoc": True,
           "n_boot": n_boot, "seed": seed, "subsets": {}}
    for subset in y2.SUBSETS:
        rows = y2.subset_rows(runs, subset, False)
        res = {}
        for run_id, r in rows.items():
            run = runs[run_id]
            Y, ep = run["Y"][r], run["epoch"][r].astype(float)
            S, A, B = Y.mean(1), Y[:, y2.A_IDX].mean(1), Y[:, y2.B_IDX].mean(1)
            bl, bs = [], []
            for _ in range(n_boot):
                ii = y2.block_resample(len(r), rng, y2.block_len(len(r)))
                bl.append(slope(ep[ii], S[ii]) if np.ptp(ep[ii]) > 0 else np.nan)
                jj = rng.integers(0, 50, size=50)
                bs.append(slope(ep, Y[:, jj].mean(1)))
            res[run_id] = {
                "n": int(len(r)), "epoch_range": [int(ep.min()), int(ep.max())],
                "slope_S50_pp_per_100ep": slope(ep, S),
                "slope_ci_block": [float(np.nanpercentile(bl, 2.5)), float(np.nanpercentile(bl, 97.5))],
                "slope_ci_seed": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "slope_A_pp_per_100ep": slope(ep, A), "slope_B_pp_per_100ep": slope(ep, B),
                "implied_trend_span_pp": slope(ep, S) * (ep.max() - ep.min()) / 100,
                "spearman_epoch_A": y2.spear(ep, A), "spearman_epoch_B": y2.spear(ep, B),
                "spearman_epoch_S50": y2.spear(ep, S),
                "residual_sd_S50_pp": float(np.std(S - np.polyval(np.polyfit(ep, S, 1), ep), ddof=2) * 100),
                "binomial_noise_sd_S50_pp": float(np.sqrt(np.mean(S * (1 - S)) / 50) * 100),
                "binomial_noise_sd_25_pp": float(np.sqrt(np.mean(S * (1 - S)) / 25) * 100),
                "mean_S50": float(S.mean()), "max_S50": float(S.max()),
                "trend_anova": trend_anova(Y, ep, rng),
            }
        out["subsets"][subset] = res
    OUT.write_text(json.dumps(y2.clean(out), indent=1))
    for k, v in out["subsets"].items():
        for run_id, r in v.items():
            print(f"{k:16s} {run_id:13s} n={r['n']:2d} slope={r['slope_S50_pp_per_100ep']:+.2f}pp/100ep "
                  f"block{np.round(r['slope_ci_block'], 2)} seed{np.round(r['slope_ci_seed'], 2)} "
                  f"span={r['implied_trend_span_pp']:+.1f}pp resid_sd={r['residual_sd_S50_pp']:.1f} "
                  f"binom_sd50={r['binomial_noise_sd_S50_pp']:.1f} rho(ep,A)={r['spearman_epoch_A']:+.2f} "
                  f"rho(ep,B)={r['spearman_epoch_B']:+.2f}")
            t = r["trend_anova"]
            print(f"{'':31s} F_trend p={t['F_trend_p']:.4f} F_dev={t['F_dev']:.2f} p={t['F_dev_p']:.3f} "
                  f"rel_dev25={t['rel_dev_25']:+.3f} ceil_partial={t['ceiling_partial_given_epoch_vs_25']:.2f} "
                  f"trend_sd={t['trend_sd_pp']:.1f}pp noise25={t['noise_sd_25_shared_seed_pp']:.1f}pp "
                  f"sim_rho_trend={t['sim_spearman_trend_follower_vs_25']:+.2f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
