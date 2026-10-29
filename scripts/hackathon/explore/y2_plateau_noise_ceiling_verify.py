#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- adversarial verification of y2_plateau_noise_ceiling.

Re-uses the y2 loader and statistics (imported, not modified) and adds checks the original
analysis did not run:

  V1  per-run full-pool epoch-vs-S50 Spearman against the per-run split-half "ceiling"
  V2  checkpoint-sampling CI of the split-half reliability / ceiling (circular block bootstrap
      over checkpoints x random 25/25 splits) -- the y2 bracket is only the across-split range
  V3  within-seed temporal persistence of the checkpoint x seed interaction (lag-1 residual
      autocorrelation and failure persistence) vs the y2 permutation null (checkpoints
      exchangeable within a seed column)
  V4  seed-cluster bootstrap of the plateau epoch trend (seeds resampled; keeps persistence)
  V5  F-distribution upper bound of the checkpoint-specific deviation variance -> upper bound of
      the "ceiling of any partial Spearman beyond epoch"
  V6  Monte-Carlo check of the ceiling estimator and of the power to detect deviations
      (monotone truth from an outcome-free-of-B isotonic fit, i.i.d. Bernoulli rollouts)
  V7  plateau-cut sensitivity (outcome-free re-cuts)
  V8  top-1 regret (pick one checkpoint) vs random held-out halves
  V9  multiplicity / fixed-split luck of every partial-given-epoch value in the y2 JSON
  V10 direct VLM-vs-instrumented agreement on the eval checkpoints
  V11 curvature check of full-pool partials: partial given a monotone epoch curve fitted on
      half A only (outcome-free w.r.t. the evaluated half B), over random splits

Outputs only under results/hackathon/explore/ and data/outputs/hackathon_explore/.
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y2_plateau_noise_ceiling_verify.py
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

ROOT = y2.ROOT
OUT = ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling_verification.json"
Y2J = ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling.json"
SEED = 9029


def iso_increasing(x, y):
    """Pool-adjacent-violators isotonic (non-decreasing) fit of y on sorted x."""
    order = np.argsort(x, kind="stable")
    yy = np.asarray(y, float)[order]
    vals, wts, cnt = [], [], []
    for v in yy:
        vals.append(v); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w; cnt[-2] += cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
    fit = np.concatenate([[v] * c for v, c in zip(vals, cnt)])
    out = np.empty_like(fit)
    out[order] = fit
    return out


def resid2(Y):
    return Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()


def lag1(R):
    a, b = R[:-1], R[1:]
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def persistence(Y):
    """P(fail at c+1 | fail at c, same seed) and the marginal failure rate of those rows."""
    F = 1 - Y
    prev, nxt = F[:-1], F[1:]
    n_prev = prev.sum()
    return (float((prev * nxt).sum() / n_prev) if n_prev > 0 else np.nan, float(nxt.mean()))


def split_mean(Ys, rng, n_split):
    vals = []
    for _ in range(n_split):
        p = rng.permutation(50)
        v = [y2.spear(Y[:, p[:25]].mean(1), Y[:, p[25:]].mean(1)) for Y in Ys]
        vals.append(np.nanmean(v))
    return float(np.nanmean(vals))


def sb(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


def main():
    rng = np.random.default_rng(SEED)
    runs, checks = y2.load()
    J = json.loads(Y2J.read_text())
    out = {"name": "y2_plateau_noise_ceiling_verification", "exploratory": True, "post_hoc": True,
           "seed": SEED, "loader_checks": checks}

    def rows(subset):
        return y2.subset_rows(runs, subset, False)

    # ---------------------------------------------------------------- V1
    v1 = {}
    for run_id, run in runs.items():
        S = run["Y"].mean(1)
        nc = J["noise_ceiling"]["all"]["per_run"][run_id]
        v1[run_id] = {"epoch_vs_S50": y2.spear(run["epoch"], S), "ceiling_vs_50": nc["ceiling_vs_50"],
                      "epoch_vs_B_fixed": y2.spear(run["epoch"], run["Y"][:, y2.B_IDX].mean(1)),
                      "ceiling_vs_25": nc["ceiling_vs_25_heldout"]}
    out["V1_full_pool_epoch_vs_ceiling_per_run"] = v1

    # ---------------------------------------------------------------- V2
    v2 = {}
    for subset in y2.SUBSETS:
        rr = rows(subset)
        Ys = {k: runs[k]["Y"][r] for k, r in rr.items()}
        reps_r, reps_c50, reps_c25 = [], [], []
        for _ in range(400):
            Yb = []
            for k, Y in Ys.items():
                idx = y2.block_resample(len(Y), rng, y2.block_len(len(Y)))
                Yb.append(Y[idx])
            per = []
            for Y in Yb:
                per.append(split_mean([Y], rng, 10))
            per = np.array(per)
            reps_r.append(np.nanmean(per))
            reps_c25.append(np.nanmean([math.sqrt(max(p, 0)) for p in per]))
            reps_c50.append(np.nanmean([math.sqrt(max(sb(p), 0)) if np.isfinite(sb(p)) else 0.0 for p in per]))
        q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
        m = J["noise_ceiling"][subset]["mean_over_runs"]
        v2[subset] = {"y2_split_half_mean": m["random_split_spearman_mean"],
                      "y2_bracket_across_splits": [m["random_split_spearman_p2_5"], m["random_split_spearman_p97_5"]],
                      "block_boot_ci_split_half_mean": q(reps_r),
                      "y2_ceiling_vs_50": m["ceiling_vs_50_mean_of_runs"], "block_boot_ci_ceiling_vs_50": q(reps_c50),
                      "y2_ceiling_vs_25": m["ceiling_vs_25_heldout_mean_of_runs"], "block_boot_ci_ceiling_vs_25": q(reps_c25)}
    out["V2_checkpoint_sampling_ci_of_ceiling"] = v2

    # ---------------------------------------------------------------- V3
    v3 = {}
    for subset in ("plateau_posthoc", "late_median", "all"):
        res = {}
        for run_id, r in rows(subset).items():
            Y = runs[run_id]["Y"][r]
            a0 = lag1(resid2(Y))
            p0, m0 = persistence(Y)
            null_a, null_p = [], []
            for _ in range(2000):
                Yp = np.take_along_axis(Y, np.argsort(rng.random(Y.shape), axis=0), axis=0)
                null_a.append(lag1(resid2(Yp)))
                null_p.append(persistence(Yp)[0])
            null_a, null_p = np.array(null_a), np.array(null_p)
            res[run_id] = {"lag1_resid_autocorr": a0, "null_mean": float(np.nanmean(null_a)),
                           "null_p97_5": float(np.nanpercentile(null_a, 97.5)),
                           "p_onesided": float((np.sum(null_a >= a0 - 1e-12) + 1) / (len(null_a) + 1)),
                           "P_fail_next_given_fail": p0, "P_fail_marginal": m0,
                           "persistence_null_mean": float(np.nanmean(null_p)),
                           "persistence_p_onesided": float((np.sum(null_p >= p0 - 1e-12) + 1) / (len(null_p) + 1))}
        v3[subset] = res
    out["V3_within_seed_persistence"] = v3

    # ---------------------------------------------------------------- V4
    v4 = {}
    for subset in ("plateau_posthoc", "late_median"):
        res = {}
        for run_id, r in rows(subset).items():
            Y, ep = runs[run_id]["Y"][r], runs[run_id]["epoch"][r].astype(float)
            t0 = y2.spear(ep, Y.mean(1))
            bs = []
            for _ in range(4000):
                jj = rng.integers(0, 50, size=50)
                bs.append(y2.spear(ep, Y[:, jj].mean(1)))
            bs = np.array(bs, float)
            res[run_id] = {"spearman_epoch_S50": t0,
                           "seed_boot_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                           "seed_boot_frac_le_0": float(np.nanmean(bs <= 0)),
                           "y2_perm_p_onesided": J["noise_ceiling"][subset]["per_run"][run_id][
                               "perm_test_epoch_trend_p_onesided"]}
        v4[subset] = res
    out["V4_seed_cluster_trend"] = v4

    # ---------------------------------------------------------------- V5
    TJ = json.loads((ROOT / "results/hackathon/explore/y2_plateau_noise_ceiling_trend.json").read_text())
    v5 = {}
    for subset in ("plateau_posthoc", "late_median"):
        res = {}
        for run_id, r in rows(subset).items():
            Y = runs[run_id]["Y"][r]
            n, k = Y.shape
            t = TJ["subsets"][subset][run_id]["trend_anova"]
            mse, Fobs = t["MSe_interaction"], t["F_dev"]
            dfe = (n - 1) * (k - 1)
            out_b = {}
            for lvl in (0.95, 0.80):
                fl = fdist.ppf(1 - lvl, n - 2, dfe)  # one-sided upper bound
                vd_hi = max(0.0, (Fobs / fl - 1.0)) * mse / k
                rel_hi = vd_hi / (vd_hi + mse / 25) if vd_hi > 0 else 0.0
                out_b[f"upper{int(lvl * 100)}"] = {"dev_sd_pp": math.sqrt(vd_hi) * 100,
                                                    "ceiling_partial_vs_25": math.sqrt(rel_hi)}
            res[run_id] = {"F_dev": Fobs, "y2_ceiling_partial": t["ceiling_partial_given_epoch_vs_25"],
                           "noise_sd_25_pp": t["noise_sd_25_shared_seed_pp"], **out_b}
        res["mean_upper95_ceiling"] = float(np.mean([res[k]["upper95"]["ceiling_partial_vs_25"] for k in res]))
        res["mean_upper80_ceiling"] = float(np.mean([res[k]["upper80"]["ceiling_partial_vs_25"]
                                                     for k in res if k.startswith(("can", "square"))]))
        v5[subset] = res
    out["V5_deviation_ceiling_upper_bounds"] = v5

    # ---------------------------------------------------------------- V6
    # monotone truth per run from the isotonic fit of S50 on epoch (+ tiny tie-break slope so
    # that epoch is exactly the truth ranking), i.i.d. Bernoulli(50) rollouts, optional
    # checkpoint-specific deviations of SD d (probability scale, clipped).  Reported: expected
    # split-half mean, ceiling estimates, the true ceiling E[spear(truth, half/S50)], epoch's
    # E[spear], the y2 F_dev detection rate, and the expected partial of a perfect deviation
    # oracle given epoch.
    from scipy.stats import f as _f
    v6 = {}
    n_sim = 150
    for subset in ("all", "plateau_posthoc"):
        rr = rows(subset)
        for d in (0.0, 0.01, 0.02, 0.03):
            acc = {k: [] for k in ("split", "c50_est", "c25_est", "truth_vs_S50", "truth_vs_half",
                                   "epoch_vs_S50", "epoch_vs_half", "oracle_partial_vs_half", "Fdev_sig")}
            for _ in range(n_sim):
                per = {k: [] for k in acc}
                for run_id, r in rr.items():
                    ep = runs[run_id]["epoch"][r].astype(float)
                    S = runs[run_id]["Y"][r].mean(1)
                    base = np.clip(iso_increasing(ep, S) + 1e-6 * ep, 0, 1)
                    p = np.clip(base + rng.normal(0, d, size=len(ep)), 0, 1) if d > 0 else base
                    Ys = (rng.random((len(ep), 50)) < p[:, None]).astype(float)
                    sm = split_mean([Ys], rng, 5)
                    per["split"].append(sm)
                    per["c25_est"].append(math.sqrt(max(sm, 0)))
                    per["c50_est"].append(math.sqrt(max(sb(sm), 0)) if np.isfinite(sb(sm)) else 0.0)
                    half = Ys[:, 25:].mean(1)
                    per["truth_vs_S50"].append(y2.spear(p, Ys.mean(1)))
                    per["truth_vs_half"].append(y2.spear(p, half))
                    per["epoch_vs_S50"].append(y2.spear(ep, Ys.mean(1)))
                    per["epoch_vs_half"].append(y2.spear(ep, half))
                    per["oracle_partial_vs_half"].append(y2.partial_spear(p, half, ep) if d > 0 else 0.0)
                    if subset != "all":
                        n, k = Ys.shape
                        rm, g = Ys.mean(1), Ys.mean()
                        mse = float((resid2(Ys) ** 2).sum()) / ((n - 1) * (k - 1))
                        fit = np.polyval(np.polyfit(ep, rm, 1), ep)
                        msd = k * float(((rm - fit) ** 2).sum()) / (n - 2)
                        per["Fdev_sig"].append(float(_f.sf(msd / mse, n - 2, (n - 1) * (k - 1)) < 0.05)
                                               if mse > 0 else np.nan)
                for kk in acc:
                    if per[kk]:
                        acc[kk].append(np.nanmean(per[kk]))
            v6[f"{subset}__dev_sd_{int(d * 100)}pp"] = {k: (float(np.nanmean(v)) if v else None) for k, v in acc.items()}
    out["V6_ceiling_estimator_simulation"] = v6

    # ---------------------------------------------------------------- V7
    v7 = {}
    for can_cut, sq_cut in ((40, 60), (60, 90), (80, 120), (100, 150), (140, 210)):
        rr = {k: np.nonzero(run["epoch"] >= (sq_cut if k.startswith("square") else can_cut))[0]
              for k, run in runs.items()}
        sp_half, sp_ep, part_l1, part_tf = [], [], [], []
        for _ in range(300):
            p = rng.permutation(50)
            a, b = p[:25], p[25:]
            h, e_, pl, pt = [], [], [], []
            for k, r in rr.items():
                run = runs[k]
                Y, ep = run["Y"][r], run["epoch"][r].astype(float)
                A_, B_ = Y[:, a].mean(1), Y[:, b].mean(1)
                h.append(y2.spear(A_, B_)); e_.append(y2.spear(ep, B_))
                pl.append(y2.partial_spear(-run["fixed"]["val_action_l1"][r], B_, ep))
                pt.append(y2.partial_spear(-run["fixed"]["teacher_forced_latent_mse"][r], B_, ep))
            sp_half.append(np.nanmean(h)); sp_ep.append(np.nanmean(e_))
            part_l1.append(np.nanmean(pl)); part_tf.append(np.nanmean(pt))
        v7[f"can>={can_cut},square>={sq_cut}"] = {
            "n": int(sum(len(r) for r in rr.values())),
            "split_half_mean": float(np.nanmean(sp_half)), "epoch_vs_half_mean": float(np.nanmean(sp_ep)),
            "partial_val_action_l1_mean": float(np.nanmean(part_l1)),
            "partial_val_action_l1_range": [float(np.nanpercentile(part_l1, 2.5)), float(np.nanpercentile(part_l1, 97.5))],
            "partial_tf_latent_mse_mean": float(np.nanmean(part_tf)),
            "partial_tf_latent_mse_range": [float(np.nanpercentile(part_tf, 2.5)), float(np.nanpercentile(part_tf, 97.5))]}
    out["V7_plateau_cut_sensitivity"] = v7

    # ---------------------------------------------------------------- V8
    v8 = {}
    for subset in ("plateau_posthoc", "late_median"):
        rr = rows(subset)
        acc = {k: [] for k in ("epoch", "random", "val_action_l1", "otherhalf_K25", "last_checkpoint_true_gap_vs_best_half")}
        for _ in range(400):
            p = rng.permutation(50)
            a, b = p[:25], p[25:]
            per = {k: [] for k in acc}
            for k, r in rr.items():
                run = runs[k]
                Y, ep = run["Y"][r], run["epoch"][r].astype(float)
                B_ = Y[:, b].mean(1)
                per["epoch"].append(y2.exp_regret(ep, B_, q=1e-9))
                per["random"].append(y2.exp_regret(np.zeros(len(ep)), B_, q=1e-9))
                per["val_action_l1"].append(y2.exp_regret(-run["fixed"]["val_action_l1"][r], B_, q=1e-9))
                per["otherhalf_K25"].append(y2.exp_regret(Y[:, a].mean(1), B_, q=1e-9))
            for kk in ("epoch", "random", "val_action_l1", "otherhalf_K25"):
                acc[kk].append(np.mean(per[kk]))
        res = {k: {"mean": float(np.mean(v)), "p2_5": float(np.percentile(v, 2.5)), "p97_5": float(np.percentile(v, 97.5))}
               for k, v in acc.items() if v}
        # trend-implied (not noisy-max) expected loss of a random pick vs the last checkpoint
        tl = []
        for k, r in rr.items():
            ep = runs[k]["epoch"][r].astype(float)
            slope = TJ["subsets"][subset][k]["slope_S50_pp_per_100ep"]
            tl.append(slope * (ep.max() - ep.mean()) / 100)
        res["trend_implied_random_vs_last_pp_per_run"] = dict(zip(rr.keys(), [float(x) for x in tl]))
        v8[subset] = res
    out["V8_top1_regret_random_splits"] = v8

    # ---------------------------------------------------------------- V9
    v9 = {"rows": []}
    for key, sd in J["signals"].items():
        for sig, d in sd.items():
            if sig.startswith("_") or sig == "epoch":
                continue
            pa = d["partial_given_epoch"]
            if pa["point"] is None:
                continue
            blk, spl = pa["ci_block_only"], pa["random_splits"]
            v9["rows"].append({"subset": key, "signal": sig, "point_fixed": pa["point"],
                               "block_excl0": bool(blk and (blk["lo"] > 0 or blk["hi"] < 0)),
                               "run_excl0": bool(pa["ci_run_level_exact"] and (pa["ci_run_level_exact"]["lo"] > 0
                                                                               or pa["ci_run_level_exact"]["hi"] < 0)),
                               "splits_lo": spl["lo"] if spl else None, "splits_hi": spl["hi"] if spl else None,
                               "splits_p_gt0": spl["p_gt0"] if spl else None,
                               "partial_vs_S50": d.get("partial_vs_full50_given_epoch")})
    v9["n_tests"] = len(v9["rows"])
    v9["n_block_excl0"] = int(sum(r["block_excl0"] for r in v9["rows"]))
    v9["n_block_excl0_positive"] = int(sum(r["block_excl0"] and r["point_fixed"] > 0 for r in v9["rows"]))
    out["V9_partial_multiplicity"] = v9

    # ---------------------------------------------------------------- V10
    v10 = {}
    for subset in ("all", "plateau_posthoc"):
        rr = y2.subset_rows(runs, subset, True)
        res = {}
        for jn in y2.JUDGES:
            per_ck, per_ep = [], []
            for k, r in rr.items():
                run = runs[k]
                E = run["E"][r]
                Jm = run["J"][jn][r]
                per_ck.append(y2.spear(Jm.mean(1), E.mean(1)))
                per_ep.append(float(np.mean(Jm == E)))
            res[jn] = {"within_run_spearman_vlmK10_vs_instrK10_mean": float(np.nanmean(per_ck)),
                       "per_run": [None if not np.isfinite(x) else float(x) for x in per_ck],
                       "episode_agreement": float(np.mean(per_ep))}
        v10[subset] = res
    out["V10_vlm_vs_instrumented"] = v10

    # ---------------------------------------------------------------- V11
    v11 = {}
    for subset, eval_only in (("all", False), ("plateau_posthoc", False), ("all", True)):
        rr = y2.subset_rows(runs, subset, eval_only)
        sigs = ["val_action_l1", "teacher_forced_latent_mse", "val_loss", "dynamics_uncertainty"]
        if eval_only:
            sigs += ["vlm_cosmos_local_v6_K10", "vlm_nemotron_nebius_v2_K10", "eval_sim_success_K10"]
        acc = {s: {"lin": [], "isoA": [], "quad": []} for s in sigs}
        for _ in range(300):
            p = rng.permutation(50)
            a, b = p[:25], p[25:]
            per = {s: {"lin": [], "isoA": [], "quad": []} for s in sigs}
            for k, r in rr.items():
                run = runs[k]
                Y, ep = run["Y"][r], run["epoch"][r].astype(float)
                A_, B_ = Y[:, a].mean(1), Y[:, b].mean(1)
                fA = iso_increasing(ep, A_) + 1e-9 * ep  # outcome-free w.r.t. B
                for s in sigs:
                    x = y2.signal_values(run, r, s, a, np.arange(10), None)
                    per[s]["lin"].append(y2.partial_spear(x, B_, ep))
                    per[s]["isoA"].append(y2.partial_spear(x, B_, fA))
                    # partial given rank(epoch) and rank(epoch)^2 (outcome-free)
                    rx, ry, re_ = y2._rk(x), y2._rk(B_), y2._rk(ep)
                    Z = np.column_stack([np.ones_like(re_), re_, re_ ** 2])
                    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
                    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
                    den = math.sqrt((ex * ex).sum() * (ey * ey).sum())
                    per[s]["quad"].append(float((ex * ey).sum() / den) if den > 0 else np.nan)
            for s in sigs:
                for m in ("lin", "isoA", "quad"):
                    acc[s][m].append(np.nanmean(per[s][m]))
        v11[f"{subset}{'__eval_ckpts' if eval_only else ''}"] = {
            s: {m: {"mean": float(np.nanmean(v)), "p2_5": float(np.nanpercentile(v, 2.5)),
                    "p97_5": float(np.nanpercentile(v, 97.5))} for m, v in d.items()} for s, d in acc.items()}
    out["V11_partial_curvature_check"] = v11

    OUT.write_text(json.dumps(y2.clean(out), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
