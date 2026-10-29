#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- supplementary checks for the y2_plateau_noise_ceiling verification.

  E1 lower-tail p-values of the y2 ANOVA F statistics (under-dispersion = noise model rejected)
  E2 Monte-Carlo check of y2.exp_regret (random tie-break, exact expectation)
  E3 per-run partial Spearman of teacher_forced_latent_mse / val_action_l1 given epoch, and the
     quadratic-epoch-rank version with a circular block bootstrap (block length 3 vs 8)
  E4 block-length sensitivity of the y2 block CI for epoch vs B on the plateau

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y2_plateau_noise_ceiling_verify_extra.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import y2_plateau_noise_ceiling as y2  # noqa: E402

OUT = y2.ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling_verification_extra.json"


def quad_partial(x, y, z):
    rx, ry, rz = y2._rk(x), y2._rk(y), y2._rk(z)
    Z = np.column_stack([np.ones_like(rz), rz, rz ** 2])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    den = math.sqrt((ex * ex).sum() * (ey * ey).sum())
    return float((ex * ey).sum() / den) if den > 0 else np.nan


def main():
    rng = np.random.default_rng(4242)
    runs, _ = y2.load()
    TJ = json.loads((y2.ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling_trend.json").read_text())
    J = json.loads((y2.ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling.json").read_text())
    out = {"name": "y2_plateau_noise_ceiling_verification_extra", "exploratory": True}

    # E1
    e1 = {}
    for subset in ("plateau_posthoc", "late_median"):
        rows = y2.subset_rows(runs, subset, False)
        for run_id, r in rows.items():
            n, k = len(r), 50
            t = TJ["subsets"][subset][run_id]["trend_anova"]
            Fd = t["F_dev"]
            a = y2.anova_icc(runs[run_id]["Y"][r])
            Fr = None
            Y = runs[run_id]["Y"][r]
            rm, cm, g = Y.mean(1), Y.mean(0), Y.mean()
            msr = k * float(((rm - g) ** 2).sum()) / (n - 1)
            mse = float(((Y - rm[:, None] - cm[None, :] + g) ** 2).sum()) / ((n - 1) * (k - 1))
            Fr = msr / mse
            e1[f"{subset}/{run_id}"] = {
                "F_rows": Fr, "F_rows_upper_p_y2": a["F_p"],
                "F_rows_lower_tail_p": float(fdist.cdf(Fr, n - 1, (n - 1) * (k - 1))),
                "F_dev": Fd, "F_dev_upper_p_y2": t["F_dev_p"],
                "F_dev_lower_tail_p": float(fdist.cdf(Fd, n - 2, (n - 1) * (k - 1))),
                "perm_heterogeneity_lower_tail_p_approx":
                    1 - J["noise_ceiling"][subset]["per_run"][run_id]["perm_test_heterogeneity_p"]}
    out["E1_underdispersion_lower_tail"] = e1

    # E2
    e2 = []
    for _ in range(20):
        n = int(rng.integers(8, 30))
        score = rng.integers(0, 4, size=n).astype(float)
        target = rng.integers(20, 26, size=n) / 25
        ex = y2.exp_regret(score, target)
        k = max(1, int(math.ceil(0.3 * n)))
        mc = []
        for _ in range(4000):
            o = np.lexsort((rng.random(n), -score))
            mc.append(target.max() - target[o[:k]].max())
        e2.append({"n": n, "exact": ex, "mc": float(np.mean(mc)), "abs_diff": abs(ex - float(np.mean(mc)))})
    out["E2_exp_regret_mc_check"] = {"max_abs_diff": max(d["abs_diff"] for d in e2), "cases": e2}

    # E3
    e3 = {}
    for subset in ("all", "plateau_posthoc"):
        rows = y2.subset_rows(runs, subset, False)
        res = {}
        for sig in ("teacher_forced_latent_mse", "val_action_l1"):
            per_run = {}
            for run_id, r in rows.items():
                run = runs[run_id]
                ep = run["epoch"][r].astype(float)
                x = -run["fixed"][sig][r]
                S, B = run["Y"][r].mean(1), run["Y"][r][:, y2.B_IDX].mean(1)
                per_run[run_id] = {"lin_vs_B": y2.partial_spear(x, B, ep), "lin_vs_S50": y2.partial_spear(x, S, ep),
                                   "quad_vs_B": quad_partial(x, B, ep), "quad_vs_S50": quad_partial(x, S, ep)}
            boot = {}
            for blen in (None, 8):
                reps = []
                for _ in range(1000):
                    vals = []
                    for run_id, r in rows.items():
                        run = runs[run_id]
                        b = y2.block_len(len(r)) if blen is None else blen
                        ii = r[y2.block_resample(len(r), rng, b)]
                        ep = run["epoch"][ii].astype(float)
                        vals.append(quad_partial(-run["fixed"][sig][ii], run["Y"][ii].mean(1), ep))
                    reps.append(np.nanmean(vals))
                boot["y2_blocklen" if blen is None else f"blocklen{blen}"] = [
                    float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]
            res[sig] = {"per_run": per_run,
                        "mean_quad_vs_S50": float(np.nanmean([v["quad_vs_S50"] for v in per_run.values()])),
                        "block_ci_quad_vs_S50": boot}
        e3[subset] = res
    out["E3_partial_per_run_and_quadratic"] = e3

    # E4
    rows = y2.subset_rows(runs, "plateau_posthoc", False)
    e4 = {}
    for blen in (None, 8, 12):
        reps = []
        for _ in range(1000):
            vals = []
            for run_id, r in rows.items():
                run = runs[run_id]
                b = y2.block_len(len(r)) if blen is None else blen
                ii = r[y2.block_resample(len(r), rng, b)]
                vals.append(y2.spear(run["epoch"][ii], run["Y"][ii][:, y2.B_IDX].mean(1)))
            reps.append(np.nanmean(vals))
        e4["y2_blocklen" if blen is None else f"blocklen{blen}"] = [float(np.percentile(reps, 2.5)),
                                                                     float(np.percentile(reps, 97.5))]
    out["E4_epoch_vs_B_plateau_block_ci_by_length"] = e4
    OUT.write_text(json.dumps(y2.clean(out), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
