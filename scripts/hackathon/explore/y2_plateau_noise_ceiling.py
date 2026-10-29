#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- y2_plateau_noise_ceiling.

Question
--------
Near the best checkpoints (late training / plateau), can ANY signal rank
checkpoints, and what ceiling does the rollout noise of the success target set?

(1) Noise ceiling.  Split-half reliability of the 50-rollout training-time
    success (fixed split 100000-24 | 100025-49, plus many random 25/25 splits):
    within-run Spearman between halves, Spearman-Brown corrected reliability
    for 50 rollouts, implied ceiling sqrt(reliability) for any predictor of the
    true success, two-way (checkpoint x shared seed) ANOVA reliability ICC(C,k),
    and seed-stratified permutation tests of between-checkpoint heterogeneity
    and of an epoch trend.
(2) Within subsets (all / epoch >= run median / post-hoc plateau: can >= 60,
    square >= 90 -- the plateau cut was chosen by looking at the learning
    curves, i.e. post hoc), Spearman and Regret@30 against the held-out half
    (seeds 100025-100049) for epoch, every cheap metric, validation loss,
    K=10 eval-rollout instrumented success, K=10 VLM-judged success, K=10
    rollout PLV metrics from Experiment 3, and the OTHER half of the training
    rollouts (25, and random 10-subsets) as the "more rollouts" reference.
(3) Incremental value beyond epoch: partial Spearman controlling for epoch
    (within run), and rank-average(epoch, metric) vs epoch alone.

Uncertainty: (a) hierarchical bootstrap = moving-block bootstrap over
epoch-ordered checkpoints within each run x resampling of rollout seeds
(training seeds within each half, eval seeds 200000-09) jointly across runs;
(b) block-only bootstrap (target fixed); (c) exact run-level bootstrap over the
3 runs (27 equally likely resamples -- coarse, reported because required);
(d) distribution over random 25/25 seed splits (target noise only).

No score is built from the target outcomes: the held-out half B is used only
for evaluation; "more rollouts" references use only half A; no parameters are
fitted (rank-average is unweighted).  Ties in Regret@30 are broken at random
(exact expectation), NOT by checkpoint path as in the paper code, because many
plateau signals are tied (e.g. 25/25 successes).

Run from the repo root:
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y2_plateau_noise_ceiling.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from itertools import product
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
NAME = "y2_plateau_noise_ceiling"
POOL = ROOT / "data/outputs/hackathon_pool"
EVAL = ROOT / "data/outputs/hackathon_rollouts_eval"
COSMOS = ROOT / "data/outputs/hackathon_cosmos"
BASELINE = ROOT / "results/hackathon/baseline_metrics.csv"
HYBRID = ROOT / "results/hackathon/hybrid_screening.csv"
OUT_JSON = ROOT / "results/hackathon/explore" / f"{NAME}.json"
OUT_CSV = ROOT / "results/hackathon/explore" / f"{NAME}_table.csv"

SEEDS = list(range(100000, 100050))
A_IDX = np.arange(0, 25)    # seeds 100000-100024 (reference "more rollouts")
B_IDX = np.arange(25, 50)   # seeds 100025-100049 (held-out target)
EVAL_SEEDS = list(range(200000, 200010))
Q = 0.3

# outcome-free per-checkpoint metrics: name -> +1 (higher better) / -1 (lower better)
# boundary_jump_h10 / replan_overlap_h10 are bit-identical to the h16 columns in
# baseline_metrics.csv, and val_loss_logged == val_loss, so the duplicates are dropped.
FIXED = {
    "epoch": +1, "val_loss": -1, "train_loss": -1, "train_action_mse": -1,
    "val_action_l1": -1, "plv1_smoothness": -1, "support_drift": -1,
    "dynamics_uncertainty": -1, "teacher_forced_latent_mse": -1,
    "open_loop_smoothness": -1, "boundary_jump_h16": -1, "replan_overlap_h16": -1,
}
TRAIN_K = {"train_rollouts_otherhalf_K25": 25, "train_rollouts_otherhalf_K10": 10}
JUDGES = {
    "vlm_cosmos_local_v6_K10": COSMOS / "eval_local_v6.jsonl",
    "vlm_nemotron_nebius_v2_K10": COSMOS / "eval_nebius_v2.jsonl",
    "vlm_nemotron_nebius_v2_pad201_K10": COSMOS / "eval_nebius_v2_pad201.jsonl",
    "vlm_cosmos_local_v7_K10": COSMOS / "eval_local_v7.jsonl",
}
HYB = {"sim_replan_overlap_K10": "sim_replan_overlap", "sim_boundary_jump_K10": "sim_boundary_jump",
       "executed_smoothness_K10": "executed_smoothness"}  # all costs (lower better)
EVAL_SIGNALS = ["eval_sim_success_K10"] + list(JUDGES) + list(HYB)
SUBSETS = {
    "all": lambda run, ep, med: np.ones_like(ep, dtype=bool),
    "late_median": lambda run, ep, med: ep >= med,
    "plateau_posthoc": lambda run, ep, med: ep >= (90 if run.startswith("square") else 60),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None


# --------------------------------------------------------------------------- data
def parse_logs(path: Path):
    by_epoch = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" not in row:
            continue
        by_epoch.setdefault(int(row["epoch"]), {}).update(
            {k: v for k, v in row.items() if isinstance(v, (int, float))})
    return by_epoch


def as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def load():
    bm = pd.read_csv(BASELINE)
    hyb = pd.read_csv(HYBRID).set_index("checkpoint_id") if HYBRID.is_file() else None
    judged = {}
    for name, p in JUDGES.items():
        d = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("scored") and r.get("parse_status") == "ok" and isinstance(r.get("cosmos_task_success"), bool):
                d[(r["checkpoint_id"], int(r["env_seed"]))] = float(r["cosmos_task_success"])
        judged[name] = d
    runs = {}
    checks = {"success_matches_baseline": True, "eval_success_matches_summary": True,
              "imputed_train_action_mse": 0}
    for run_id, g in bm.groupby("run_id", sort=True):
        g = g.sort_values("epoch")
        run_dir = POOL / Path(g["checkpoint_path"].iloc[0]).parts[-3]
        logs = parse_logs(run_dir / "logs.json.txt")
        ids = [str(x) for x in g["checkpoint_id"].tolist()]
        eps = g["epoch"].to_numpy(dtype=int)
        Y = np.array([[1.0 if float(logs[e][f"test/sim_max_reward_{s}"]) > 0.5 else 0.0 for s in SEEDS]
                      for e in eps])
        if not np.allclose(Y.mean(1), g["success"].to_numpy(float)):
            checks["success_matches_baseline"] = False
        fixed = {m: g[m].to_numpy(dtype=float) for m in FIXED if m in g.columns}
        fixed["epoch"] = eps.astype(float)
        tam = np.array([float(logs[e].get("train_action_mse_error", np.nan)) for e in eps])
        ok = np.isfinite(tam)
        checks["imputed_train_action_mse"] += int((~ok).sum())
        if 0 < ok.sum() < len(tam):
            tam = np.interp(eps, eps[ok], tam[ok])
        fixed["train_action_mse"] = tam
        n = len(ids)
        E = np.full((n, 10), np.nan)
        J = {k: np.full((n, 10), np.nan) for k in JUDGES}
        H = {k: np.full(n, np.nan) for k in HYB}
        for i, cid in enumerate(ids):
            f = EVAL / cid / "summary.json"
            if not f.is_file():
                continue
            s = json.loads(f.read_text())
            for ep in s["episodes"]:
                j = int(ep["env_seed"]) - 200000
                E[i, j] = float(as_bool(ep["success"]))
            if abs(np.nanmean(E[i]) - float(s["success_rate"])) > 1e-9:
                checks["eval_success_matches_summary"] = False
            for k, d in judged.items():
                for j, sd in enumerate(EVAL_SEEDS):
                    if (cid, sd) in d:
                        J[k][i, j] = d[(cid, sd)]
            if hyb is not None and cid in hyb.index:
                for k, col in HYB.items():
                    H[k][i] = float(hyb.loc[cid, col])
        is_eval = np.isfinite(E).all(1)
        for k in JUDGES:
            is_eval &= np.isfinite(J[k]).all(1)
        if hyb is not None:
            for k in HYB:
                is_eval &= np.isfinite(H[k])
        runs[run_id] = {"ids": ids, "epoch": eps, "median_epoch": float(np.median(eps)), "Y": Y,
                        "fixed": fixed, "E": E, "J": J, "H": H, "is_eval": is_eval}
    return runs, checks


# --------------------------------------------------------------------------- stats
def _rk(x):
    return rankdata(x, method="average")


def spear(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.ptp(y) == 0:
        return np.nan
    if np.ptp(x) == 0:
        return 0.0  # a constant predictor carries no ranking information (= random ranking)
    rx, ry = _rk(x), _rk(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))


def partial_spear(x, y, z):
    if len(x) < 4 or np.ptp(y) == 0 or np.ptp(z) == 0:
        return np.nan
    if np.ptp(x) == 0:
        return 0.0
    rxy, rxz, ryz = spear(x, y), spear(x, z), spear(y, z)
    den = (1 - rxz ** 2) * (1 - ryz ** 2)
    if not np.isfinite(den) or den <= 1e-10:
        return np.nan
    return float((rxy - rxz * ryz) / math.sqrt(den))


def exp_regret(score, target, q=Q):
    """Regret@q with ties in `score` broken uniformly at random (exact expectation)."""
    score = np.asarray(score, float)
    target = np.asarray(target, float)
    n = len(score)
    k = max(1, int(math.ceil(q * n)))
    best = float(target.max())
    maxf, cnt = -np.inf, 0
    for lev in np.unique(score)[::-1]:
        idx = np.nonzero(score == lev)[0]
        if cnt + len(idx) <= k:
            maxf = max(maxf, float(target[idx].max()))
            cnt += len(idx)
            if cnt == k:
                break
        else:
            m, t = k - cnt, len(idx)
            v = np.sort(target[idx])[::-1]
            tot = math.comb(t, m)
            e = sum(math.comb(t - 1 - j, m - 1) / tot * max(maxf, v[j]) for j in range(t - m + 1))
            return best - e
    return best - maxf


def rank_avg(a, b):
    return _rk(a) + _rk(b)


def block_resample(n, rng, b):
    """Circular moving-block bootstrap over epoch-ordered checkpoints (no edge under-sampling)."""
    if n <= b:
        return rng.integers(0, n, size=n)
    nb = int(math.ceil(n / b))
    starts = rng.integers(0, n, size=nb)
    return (np.concatenate([np.arange(s, s + b) for s in starts])[:n]) % n


def block_len(n):
    return max(2, int(math.ceil(n ** (1.0 / 3.0))))


# --------------------------------------------------------------------------- one replicate
def signal_values(run, r, sig, a_seeds, e_seeds, a_sub):
    if sig in FIXED:
        return FIXED[sig] * run["fixed"][sig][r]
    if sig == "train_rollouts_otherhalf_K25":
        return run["Y"][r][:, a_seeds].mean(1)
    if sig == "train_rollouts_otherhalf_K10":
        return run["Y"][r][:, a_sub].mean(1)
    if sig == "eval_sim_success_K10":
        return run["E"][r][:, e_seeds].mean(1)
    if sig in JUDGES:
        return run["J"][sig][r][:, e_seeds].mean(1)
    if sig in HYB:
        return -run["H"][sig][r]
    raise KeyError(sig)


def run_stats(run, r, sigs, b_seeds, a_seeds, e_seeds, a_subs, target_full=False):
    """Per-run statistics for every signal.  r = row indices (subset, possibly resampled)."""
    Y = run["Y"][r]
    tgt = Y.mean(1) if target_full else Y[:, b_seeds].mean(1)
    ep = run["epoch"][r].astype(float)
    out = {}
    for sig in sigs:
        subs = a_subs if sig == "train_rollouts_otherhalf_K10" else [None]
        acc = []
        for a_sub in subs:
            x = signal_values(run, r, sig, a_seeds, e_seeds, a_sub)
            ra = rank_avg(ep, x)
            acc.append((spear(x, tgt), exp_regret(x, tgt), partial_spear(x, tgt, ep),
                        spear(ra, tgt), exp_regret(ra, tgt)))
        acc = np.array(acc, float)
        with np.errstate(all="ignore"):
            m = np.nanmean(acc, 0) if np.isfinite(acc).any() else acc[0]
        out[sig] = m
    out["_random"] = np.array([0.0 if np.ptp(tgt) > 0 else np.nan,
                               exp_regret(np.zeros(len(tgt)), tgt), np.nan, np.nan, np.nan])
    out["_target_constant"] = bool(np.ptp(tgt) == 0)
    return out


STAT_NAMES = ("spearman", "regret30", "partial_given_epoch", "rankavg_spearman", "rankavg_regret30")


def aggregate(per_run, sigs):
    """Unweighted mean over runs (NaN runs, i.e. constant target, are skipped)."""
    res = {}
    for sig in list(sigs) + ["_random"]:
        arr = np.array([per_run[run][sig] for run in per_run], float)  # runs x 5
        with np.errstate(all="ignore"):
            res[sig] = np.array([np.nanmean(arr[:, j]) if np.isfinite(arr[:, j]).any() else np.nan
                                 for j in range(arr.shape[1])])
    ep = res["epoch"]
    for sig in sigs:
        v = res[sig]
        # deltas vs epoch alone: d_spearman(signal-epoch), d_regret(signal-epoch),
        # d_rankavg_spearman(rankavg-epoch), d_rankavg_regret(rankavg-epoch)
        res[sig] = np.concatenate([v, [v[0] - ep[0], v[1] - ep[1], v[3] - ep[0], v[4] - ep[1]]])
    res["_random"] = np.concatenate([res["_random"], [np.nan] * 4])
    return res


ALL_STAT_NAMES = STAT_NAMES + ("d_spearman_vs_epoch", "d_regret30_vs_epoch",
                               "d_rankavg_spearman_vs_epoch", "d_rankavg_regret30_vs_epoch")


def subset_rows(runs, subset, eval_only):
    rows = {}
    for run_id, run in runs.items():
        m = SUBSETS[subset](run_id, run["epoch"], run["median_epoch"])
        if eval_only:
            m = m & run["is_eval"]
        rows[run_id] = np.nonzero(m)[0]
    return rows


def signals_for(eval_only):
    return list(FIXED) + list(TRAIN_K) + (EVAL_SIGNALS if eval_only else [])


# --------------------------------------------------------------------------- part 1
def noise_ceiling(runs, rows, rng, n_split, n_perm):
    out = {"per_run": {}}
    splits = [rng.permutation(50) for _ in range(n_split)]
    split_mean = []
    per_run_split = {k: [] for k in rows}
    for perm in splits:
        a, b = perm[:25], perm[25:]
        vals = []
        for run_id, r in rows.items():
            Y = runs[run_id]["Y"][r]
            v = spear(Y[:, a].mean(1), Y[:, b].mean(1))
            per_run_split[run_id].append(v)
            vals.append(v)
        with np.errstate(all="ignore"):
            split_mean.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
    for run_id, r in rows.items():
        run = runs[run_id]
        Y = run["Y"][r]
        n = len(r)
        S = Y.mean(1)
        fixed_r = spear(Y[:, A_IDX].mean(1), Y[:, B_IDX].mean(1))
        rs = np.array(per_run_split[run_id], float)
        rbar = float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan
        sb = 2 * rbar / (1 + rbar) if np.isfinite(rbar) and rbar > -1 else np.nan
        # two-way (checkpoint x seed) ANOVA consistency reliability, ICC(C,k): seeds are shared
        # across checkpoints, so seed difficulty is a common fixed effect, not between-checkpoint
        # noise.  (The naive independent-binomial version is kept for reference only; it ignores
        # the shared seeds and is biased low.)
        icc = anova_icc(Y)
        var_obs = float(np.var(S, ddof=1)) if n > 1 else np.nan
        noise = float(np.mean(S * (1 - S)) / (50 - 1))
        rel50_binom = (var_obs - noise) / var_obs if var_obs > 0 else np.nan
        # seed-stratified permutation test: permute checkpoints within each seed column
        def stats(Yp):
            s = Yp.mean(1)
            return float(np.var(s, ddof=1)), spear(run["epoch"][r], s)
        v0, t0 = stats(Y)
        ge_v, ge_t, cnt_t = 0, 0, 0
        for _ in range(n_perm):
            Yp = np.take_along_axis(Y, np.argsort(rng.random(Y.shape), axis=0), axis=0)
            vp, tp = stats(Yp)
            ge_v += vp >= v0 - 1e-12
            if np.isfinite(tp) and np.isfinite(t0):
                cnt_t += 1
                ge_t += tp >= t0 - 1e-12
        out["per_run"][run_id] = {
            "n_checkpoints": int(n), "epochs": [int(e) for e in run["epoch"][r]],
            "success50_mean": float(S.mean()), "success50_min": float(S.min()), "success50_max": float(S.max()),
            "success50_sd": float(np.std(S, ddof=1)) if n > 1 else None,
            "binomial_noise_sd_50": math.sqrt(noise), "binomial_noise_sd_25": math.sqrt(noise * 49 / 24),
            "fixed_split_spearman_A_B": fixed_r,
            "random_split_spearman_mean": rbar,
            "random_split_spearman_p2_5": float(np.nanpercentile(rs, 2.5)) if np.isfinite(rs).any() else None,
            "random_split_spearman_p97_5": float(np.nanpercentile(rs, 97.5)) if np.isfinite(rs).any() else None,
            "random_split_frac_nan_constant_half": float(np.mean(~np.isfinite(rs))),
            "spearman_brown_reliability_50": sb,
            "ceiling_vs_25_heldout": math.sqrt(max(rbar, 0.0)) if np.isfinite(rbar) else None,
            "ceiling_vs_50": math.sqrt(max(sb, 0.0)) if np.isfinite(sb) else None,
            "anova_icc_single": icc["icc1"], "anova_reliability_25": icc["rel25"],
            "anova_reliability_50": icc["rel50"],
            "anova_ceiling_vs_25": math.sqrt(max(icc["rel25"], 0.0)) if np.isfinite(icc["rel25"]) else None,
            "anova_ceiling_vs_50": math.sqrt(max(icc["rel50"], 0.0)) if np.isfinite(icc["rel50"]) else None,
            "anova_F_p": icc["F_p"],
            "naive_binomial_reliability_50_ignores_shared_seeds": rel50_binom,
            "perm_test_heterogeneity_p": (ge_v + 1) / (n_perm + 1),
            "perm_test_epoch_trend_spearman": t0,
            "perm_test_epoch_trend_p_onesided": (ge_t + 1) / (cnt_t + 1) if cnt_t else None,
        }
    pr = out["per_run"]
    sm = np.array(split_mean, float)

    def mean_of(key):
        v = [pr[k][key] for k in pr if pr[k][key] is not None and np.isfinite(pr[k][key])]
        return float(np.mean(v)) if v else None
    out["mean_over_runs"] = {
        "n_checkpoints": int(sum(pr[k]["n_checkpoints"] for k in pr)),
        "fixed_split_spearman_A_B": mean_of("fixed_split_spearman_A_B"),
        "random_split_spearman_mean": float(np.nanmean(sm)),
        "random_split_spearman_p2_5": float(np.nanpercentile(sm, 2.5)),
        "random_split_spearman_p97_5": float(np.nanpercentile(sm, 97.5)),
        "spearman_brown_reliability_50_mean_of_runs": mean_of("spearman_brown_reliability_50"),
        "ceiling_vs_25_heldout_mean_of_runs": mean_of("ceiling_vs_25_heldout"),
        "ceiling_vs_50_mean_of_runs": mean_of("ceiling_vs_50"),
        "anova_reliability_25_mean_of_runs": mean_of("anova_reliability_25"),
        "anova_reliability_50_mean_of_runs": mean_of("anova_reliability_50"),
        "anova_ceiling_vs_25_mean_of_runs": mean_of("anova_ceiling_vs_25"),
        "anova_ceiling_vs_50_mean_of_runs": mean_of("anova_ceiling_vs_50"),
    }
    return out


def anova_icc(Y):
    """Two-way ANOVA without replication (rows = checkpoints, cols = shared seeds).
    ICC(C,1) = (MSr - MSe) / (MSr + (k-1) MSe);  reliability of a k-seed mean = (MSr - MSe)/MSr
    for k = all seeds, Spearman-Brown for other k.  Pearson-scale, Gaussian approximation."""
    from scipy.stats import f as fdist
    n, k = Y.shape
    if n < 3:
        return {"icc1": np.nan, "rel25": np.nan, "rel50": np.nan, "F_p": np.nan}
    g = Y.mean()
    rm, cm = Y.mean(1), Y.mean(0)
    msr = k * float(((rm - g) ** 2).sum()) / (n - 1)
    resid = Y - rm[:, None] - cm[None, :] + g
    mse = float((resid ** 2).sum()) / ((n - 1) * (k - 1))
    if msr <= 0 and mse <= 0:
        return {"icc1": np.nan, "rel25": np.nan, "rel50": np.nan, "F_p": np.nan}
    icc1 = (msr - mse) / (msr + (k - 1) * mse)
    sb = lambda m: m * icc1 / (1 + (m - 1) * icc1)  # noqa: E731
    F = msr / mse if mse > 0 else np.inf
    return {"icc1": float(icc1), "rel25": float(sb(25)), "rel50": float(sb(50)),
            "F_p": float(fdist.sf(F, n - 1, (n - 1) * (k - 1)))}


# --------------------------------------------------------------------------- parts 2+3
def evaluate_subset(runs, rows, sigs, rng, n_boot, n_split, n_asub):
    point_subs = [rng.choice(A_IDX, size=10, replace=False) for _ in range(n_asub)]
    allseeds_e = np.arange(10)
    per_run_point = {k: run_stats(runs[k], r, sigs, B_IDX, A_IDX, allseeds_e, point_subs)
                     for k, r in rows.items()}
    point = aggregate(per_run_point, sigs)
    # outcome-free signals vs the full 50-rollout target (context; not for train-rollout refs)
    of_sigs = [s for s in sigs if s not in TRAIN_K]
    per_run_full = {k: run_stats(runs[k], r, of_sigs, B_IDX, A_IDX, allseeds_e, point_subs, target_full=True)
                    for k, r in rows.items()}
    point_full = aggregate(per_run_full, of_sigs)

    # exact run-level bootstrap (all 3^3 resamples)
    run_ids = list(rows)
    run_boot = []
    for combo in product(run_ids, repeat=len(run_ids)):
        pr = {f"{i}_{k}": per_run_point[k] for i, k in enumerate(combo)}
        run_boot.append(aggregate(pr, sigs))

    def boot(kind):
        reps = []
        for _ in range(n_boot):
            if kind == "hier":
                bs = rng.choice(B_IDX, size=25, replace=True)
                as_ = rng.choice(A_IDX, size=25, replace=True)
                es = rng.integers(0, 10, size=10)
                asub = [as_[rng.choice(25, size=10, replace=False)]]
            else:
                bs, as_, es = B_IDX, A_IDX, allseeds_e
                asub = [point_subs[rng.integers(len(point_subs))]]
            pr = {}
            for k, r in rows.items():
                rr = r[block_resample(len(r), rng, block_len(len(r)))]
                pr[k] = run_stats(runs[k], rr, sigs, bs, as_, es, asub)
            reps.append(aggregate(pr, sigs))
        return reps

    hier = boot("hier")
    blk = boot("block")

    # random 25/25 seed splits (checkpoints fixed): target = random half, reference = other half
    split_reps = []
    for _ in range(n_split):
        perm = rng.permutation(50)
        a, b = perm[:25], perm[25:]
        asub = [a[rng.choice(25, size=10, replace=False)]]
        pr = {k: run_stats(runs[k], r, sigs, b, a, allseeds_e, asub) for k, r in rows.items()}
        split_reps.append(aggregate(pr, sigs))

    def summarize(reps, sig, j):
        v = np.array([x[sig][j] for x in reps], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return None
        return {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                "p_gt0": float(np.mean(v > 0)), "p_lt0": float(np.mean(v < 0)), "n": int(len(v))}

    res = {}
    for sig in list(sigs) + ["_random"]:
        d = {}
        for j, st in enumerate(ALL_STAT_NAMES):
            pv = point[sig][j]
            d[st] = {
                "point": None if not np.isfinite(pv) else float(pv),
                "ci_hier_block_x_seed": summarize(hier, sig, j),
                "ci_block_only": summarize(blk, sig, j),
                "ci_run_level_exact": summarize(run_boot, sig, j),
                "random_splits": summarize(split_reps, sig, j),
            }
            if st in ("spearman", "regret30"):
                vs = np.array([x[sig][j] for x in split_reps], float)
                d[st]["random_splits_mean"] = float(np.nanmean(vs)) if np.isfinite(vs).any() else None
        d["per_run"] = {k: dict(zip(STAT_NAMES, [None if not np.isfinite(x) else float(x)
                                                  for x in per_run_point[k][sig]])) for k in rows}
        if sig in point_full:
            d["spearman_vs_full50"] = None if not np.isfinite(point_full[sig][0]) else float(point_full[sig][0])
            d["regret30_vs_full50"] = None if not np.isfinite(point_full[sig][1]) else float(point_full[sig][1])
            d["partial_vs_full50_given_epoch"] = (None if not np.isfinite(point_full[sig][2])
                                                  else float(point_full[sig][2]))
        res[sig] = d
    res["_target_constant_runs"] = [k for k in rows if per_run_point[k]["_target_constant"]]
    return res


# --------------------------------------------------------------------------- main
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _worker(task):
    i, subset, eval_only, args = task
    rng = np.random.default_rng([args.seed, i])  # independent, reproducible stream per subset
    runs, _ = load()
    key = f"{subset}{'__eval_ckpts' if eval_only else ''}"
    rows = subset_rows(runs, subset, eval_only)
    nc = noise_ceiling(runs, rows, rng, args.splits, args.perm)
    sg = evaluate_subset(runs, rows, signals_for(eval_only), rng, args.boot, args.splits, args.asub)
    return key, nc, sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--asub", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args()
    t0 = time.time()
    runs, checks = load()
    out = {
        "name": NAME, "exploratory": True, "post_hoc": True,
        "label": "EXPLORATORY / post-hoc. Plateau cut (can>=60, square>=90) chosen by looking at learning curves.",
        "command": "PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python "
                   f"scripts/hackathon/explore/{NAME}.py --boot {args.boot} --splits {args.splits} "
                   f"--perm {args.perm} --asub {args.asub} --seed {args.seed} --workers {args.workers}",
        "args": vars(args),
        "inputs": {"baseline_metrics.csv": sha(BASELINE), "hybrid_screening.csv": sha(HYBRID),
                   **{p.name: sha(p) for p in JUDGES.values()}},
        "checks": checks,
        "conventions": {
            "target": "held-out half B = training-time rollouts seeds 100025-100049 (25 episodes)",
            "reference_more_rollouts": "half A = seeds 100000-100024 (K25), random 10-subsets of A (K10)",
            "orientation": "all signals oriented higher=better (costs negated, epoch later=better)",
            "within_run": "statistics computed within run, then unweighted mean over runs; runs whose "
                          "target is constant within the subset are skipped (NaN)",
            "constant_predictor": "Spearman/partial = 0 (no ranking information), regret = random-pick",
            "regret_ties": "Regret@30 ties broken uniformly at random (exact expectation)",
            "rankavg": "rank(epoch) + rank(signal), unweighted, within run",
            "block_bootstrap": "circular moving blocks of ceil(n^(1/3)) (min 2) epoch-consecutive checkpoints within run",
            "hier_bootstrap": "block bootstrap x resampling of seeds (B within B, A within A, eval seeds) "
                              "jointly across runs",
            "run_level": "exact bootstrap over the 3 runs (27 resamples) -- very coarse",
            "duplicates_dropped": "boundary_jump_h10==boundary_jump_h16, replan_overlap_h10==replan_overlap_h16, "
                                  "val_loss_logged==val_loss (bit-identical in baseline_metrics.csv)",
        },
        "runs": {k: {"n": len(v["ids"]), "median_epoch": v["median_epoch"],
                     "n_eval": int(v["is_eval"].sum())} for k, v in runs.items()},
        "noise_ceiling": {}, "signals": {},
    }
    tasks = [(i, subset, eval_only, args) for i, (subset, eval_only) in
             enumerate((s_, e_) for s_ in SUBSETS for e_ in (False, True))]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        for key, nc, sg in ex.map(_worker, tasks):
            print(f"[{time.time() - t0:6.1f}s] done {key}", flush=True)
            out["noise_ceiling"][key] = nc
            out["signals"][key] = sg
    out["runtime_s"] = time.time() - t0
    Path(args.out_json).write_text(json.dumps(clean(out), indent=1))
    # flat table
    recs = []
    for key, sd in out["signals"].items():
        nc = out["noise_ceiling"][key]["mean_over_runs"]
        for sig, d in sd.items():
            if sig.startswith("_target"):
                continue
            rec = {"subset": key, "signal": sig, "n_checkpoints": nc["n_checkpoints"],
                   "ceiling_vs_25": nc["ceiling_vs_25_heldout_mean_of_runs"]}
            for st in ALL_STAT_NAMES:
                rec[st] = d[st]["point"]
                c = d[st]["ci_hier_block_x_seed"]
                rec[st + "_hier_lo"] = c["lo"] if c else None
                rec[st + "_hier_hi"] = c["hi"] if c else None
                c = d[st]["ci_run_level_exact"]
                rec[st + "_run_lo"] = c["lo"] if c else None
                rec[st + "_run_hi"] = c["hi"] if c else None
            rec["spearman_vs_full50"] = d.get("spearman_vs_full50")
            recs.append(rec)
    pd.DataFrame(recs).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_json} and {args.out_csv} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
