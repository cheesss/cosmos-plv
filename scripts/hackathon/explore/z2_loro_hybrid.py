#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- z2_loro_hybrid.

Question
--------
Can ANY learned combination of outcome-free checkpoint signals (epoch, cheap PLV
metrics, losses, and -- on the 33 eval checkpoints -- K=10 VLM-judged success
rates) beat the free "later epoch" baseline when it is properly
cross-validated across runs?

Protocol (no outcome leakage)
-----------------------------
* 93 Diffusion Policy checkpoints in 3 runs (can_seed1 41, can_seed2 21,
  square_seed1 31).  Features come from results/hackathon/baseline_metrics.csv
  (outcome-free) and, for the 33 eval checkpoints, from the K=10 eval rollouts
  (env seeds 200000-200009, disjoint from the target seeds) judged by
  Cosmos3-Edge (eval_local_v6) and Nemotron (eval_nebius_v2, eval_nebius_v2_pad201).
  The privileged instrumented K=10 success is kept as a reference feature only.
* Per-episode training-time outcomes test/sim_max_reward_<seed> (seeds
  100000-100049) from each run's logs.json.txt.  HA = 100000-100024 (FIT labels),
  HB = 100025-100049 (TARGET).  Asserted to reproduce baseline `success`.
* Leave-one-run-out (3 folds): models are fit on the two training runs with HA
  labels only (hyper-parameters chosen by an inner leave-one-run-out over the two
  training runs, again on HA only), then score the held-out run, which is
  evaluated against its HB success.  Feature normalisation is within run and uses
  no outcomes.
* Statistics on the held-out run: within-run Spearman (exact expectation over
  random tie-breaks), Regret@30 = max S - max S in the top ceil(0.3 n),
  top-1 regret.  Averaged over the 3 held-out runs.
* CIs:
    run-level   : exact enumeration of the 27 ordered resamples of the 3 fold
                  results (with 3 runs the 95% interval is [min fold, max fold]).
    block       : joint bootstrap with refitting of (i) moving blocks of
                  checkpoints inside every run (40-epoch blocks), (ii) HA env
                  seeds, (iii) HB env seeds, (iv) eval env seeds; paired
                  differences vs epoch are taken inside each replicate.

Run from the repo root:
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/z2_loro_hybrid.py --n-boot 1000
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import comb, expit  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
NAME = "z2_loro_hybrid"
RUN_DIRS = {
    "can_seed1": "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100",
    "can_seed2": "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed2_e200_s200_r10_inf100",
    "square_seed1": "data/outputs/hackathon_pool/paper_dp_square_abs_ph_seed1_e300_s200_r10_inf100",
}
RUNS = list(RUN_DIRS)
TRAIN_SEEDS = list(range(100000, 100050))
HA = np.arange(0, 25)
HB = np.arange(25, 50)
EVAL_SEEDS = list(range(200000, 200010))
JUDGES = {
    "vlm_cosmos_v6": "data/outputs/hackathon_cosmos/eval_local_v6.jsonl",
    "vlm_nemotron_v2": "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl",
    "vlm_nemotron_pad201": "data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl",
}
# cheap outcome-free metrics (costs: lower = better a priori).  replan/boundary
# H=10 are identical to H=16 in this pool (structural) and val_loss_logged ==
# val_loss, so they are omitted.
CHEAP = [
    "val_action_l1", "open_loop_smoothness", "plv1_smoothness", "replan_overlap_h16",
    "boundary_jump_h16", "val_loss", "train_loss", "support_drift",
    "dynamics_uncertainty", "teacher_forced_latent_mse",
]
SMALL = ["epoch", "val_action_l1", "open_loop_smoothness", "replan_overlap_h16", "val_loss"]
VLM2 = ["vlm_cosmos_v6", "vlm_nemotron_v2"]
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
FIXED_LAM = 0.1
BLOCK_EPOCHS = 40


# ============================================================================ data
def load_train_outcomes(run_dir: Path, epochs):
    by_epoch = {}
    for line in (run_dir / "logs.json.txt").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" not in row or f"test/sim_max_reward_{TRAIN_SEEDS[0]}" not in row:
            continue
        by_epoch.setdefault(int(row["epoch"]), {}).update(row)
    return np.array([[1.0 if float(by_epoch[int(e)][f"test/sim_max_reward_{s}"]) > 0.5 else 0.0
                      for s in TRAIN_SEEDS] for e in epochs])


def load_data():
    base = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv")
    ck = base.sort_values(["run_id", "epoch"]).reset_index(drop=True)
    assert list(ck.run_id.unique()) == sorted(RUNS) and len(ck) == 93
    Y = np.zeros((len(ck), 50))
    for run, rd in RUN_DIRS.items():
        m = (ck.run_id == run).values
        Y[m] = load_train_outcomes(ROOT / rd, ck.epoch.values[m])
    err = np.abs(Y.mean(1) - ck.success.values).max()
    assert err < 1e-9, f"per-episode outcomes do not reproduce success (max err {err})"
    pos = {c: i for i, c in enumerate(ck.checkpoint_id)}
    E = np.full((len(ck), 10), np.nan)
    for c, i in pos.items():
        p = ROOT / "data/outputs/hackathon_rollouts_eval" / c / "summary.json"
        if p.is_file():
            for ep in json.loads(p.read_text())["episodes"]:
                E[i, EVAL_SEEDS.index(int(ep["env_seed"]))] = float(bool(ep["success"]))
    V = {j: np.full((len(ck), 10), np.nan) for j in JUDGES}
    for j, f in JUDGES.items():
        for line in open(ROOT / f):
            r = json.loads(line)
            i, k = pos[r["checkpoint_id"]], EVAL_SEEDS.index(int(r["env_seed"]))
            assert bool(r.get("scored")), (j, r["checkpoint_id"], r["env_seed"])
            assert float(bool(r["truth_success"])) == E[i, k], "VLM truth != eval summary"
            V[j][i, k] = float(bool(r["cosmos_task_success"]))
    has_eval = np.isfinite(E).all(1)
    assert has_eval.sum() == 33
    for j in JUDGES:
        assert np.isfinite(V[j][has_eval]).all()
    return ck, Y, E, V, has_eval


# ============================================================================ stats
def exp_spearman(score, y):
    """E over uniform random tie-breaks of the score of Spearman(score, y)."""
    n = len(score)
    if n < 3:
        return np.nan
    rx = stats.rankdata(score) - (n + 1) / 2
    ry = stats.rankdata(y) - (n + 1) / 2
    dy = (ry * ry).sum()
    if dy == 0:
        return np.nan
    return float((rx * ry).sum() / math.sqrt(n * (n * n - 1) / 12.0 * dy))


def exp_topk_max(score, y, k):
    """E over uniform random tie-breaks of max y among the k best-scored items."""
    order = np.argsort(-score, kind="stable")
    s, v = score[order], y[order]
    n, taken, cur, i = len(s), 0, -np.inf, 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        g = j - i + 1
        if taken + g <= k:
            cur = max(cur, v[i:j + 1].max())
            taken += g
            if taken == k:
                return float(cur)
        else:
            m = k - taken
            vals = np.sort(v[i:j + 1])[::-1]
            t = np.arange(g - m + 1)
            p = comb(g - 1 - t, m - 1) / comb(g, m)
            return float((p * np.maximum(cur, vals[: g - m + 1])).sum())
        i = j + 1
    return float(cur)


def sel_stats(score, y):
    n = len(score)
    best = float(y.max())
    k30 = max(1, int(math.ceil(0.3 * n)))
    return {"spearman": exp_spearman(score, y),
            "regret@30": best - exp_topk_max(score, y, k30),
            "top1_regret": best - exp_topk_max(score, y, 1)}


STAT_KEYS = ["spearman", "regret@30", "top1_regret"]


def rank01(x):
    n = len(x)
    if n < 2:
        return np.zeros(n)
    return (stats.rankdata(x) - 1.0) / (n - 1.0)


# ============================================================================ features
def raw_oriented(ck, Ek, Vk):
    """Outcome-free raw features, oriented so that higher = better a priori.
    K=10 features are NaN outside eval checkpoints."""
    F = {"epoch": ck.epoch.values.astype(float)}
    for m in CHEAP:
        F[m] = -np.log(ck[m].values.astype(float))
    F["k10_sim"] = np.nanmean(Ek, 1) if np.isfinite(Ek).any() else np.full(len(ck), np.nan)
    for j, a in Vk.items():
        F[j] = a.mean(1)
    return F


def normalise(F, runs, rows):
    """Within-run z-score ('z') and rank-to-[0,1] ('rank') over the given rows."""
    out = {"z": {}, "rank": {}}
    for name, x in F.items():
        z = np.full(len(x), np.nan)
        r = np.full(len(x), np.nan)
        for run in RUNS:
            idx = np.where(rows & (runs == run))[0]
            v = x[idx]
            sd = v.std()
            z[idx] = (v - v.mean()) / sd if sd > 0 else 0.0
            r[idx] = rank01(v)
        out["z"][name] = z
        out["rank"][name] = r
    return out


# ============================================================================ models
def within_run_pairs(X, y, runs):
    D, W = [], []
    ur = np.unique(runs)
    for r in ur:
        idx = np.where(runs == r)[0]
        ii, jj = np.triu_indices(len(idx), 1)
        dy = y[idx[ii]] - y[idx[jj]]
        m = dy != 0
        if not m.any():
            continue
        d = (X[idx[ii[m]]] - X[idx[jj[m]]]) * np.sign(dy[m])[:, None]
        w = np.abs(dy[m])
        D.append(d)
        W.append(w / w.sum())
    if not D:
        return np.zeros((0, X.shape[1])), np.zeros(0)
    return np.vstack(D), np.concatenate(W) / len(D)


def fit_pairlogit(X, y, runs, lam):
    """Linear RankNet: L2-penalised logistic loss on within-run pairs (no
    intercept), weights |dy|, each run contributes equally.  Newton steps."""
    D, W = within_run_pairs(X, y, runs)
    w = np.zeros(X.shape[1])
    if len(W) == 0:                        # no informative pair (constant fit labels)
        return w
    for _ in range(50):
        z = D @ w
        p = expit(-z)                      # d/dz log(1+exp(-z)) = -p
        g = -(D.T @ (W * p)) + lam * w
        H = (D * (W * p * (1 - p))[:, None]).T @ D + lam * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-10:
            break
    return w


def fit_ridge_rank(X, y, runs, lam):
    """Ridge of the within-run rank of the fit-half success on within-run
    centred features (each run equally weighted)."""
    A = np.zeros((X.shape[1], X.shape[1]))
    b = np.zeros(X.shape[1])
    ur = np.unique(runs)
    for r in ur:
        idx = np.where(runs == r)[0]
        Xi = X[idx] - X[idx].mean(0)
        t = rank01(y[idx])
        t = t - t.mean()
        A += Xi.T @ Xi / len(idx)
        b += Xi.T @ t / len(idx)
    A /= len(ur)
    b /= len(ur)
    return np.linalg.solve(A + lam * np.eye(len(b)), b)


FITTERS = {"pairlogit": fit_pairlogit, "ridge": fit_ridge_rank}


def nested_fit(kind, X, y, runs, lams=LAMS):
    """Inner leave-one-run-out over the training runs (fit labels only)."""
    tr_runs = np.unique(runs)
    inner = []
    for lam in lams:
        rhos = []
        for hold in tr_runs:
            a, h = runs != hold, runs == hold
            w = FITTERS[kind](X[a], y[a], runs[a], lam)
            rhos.append(exp_spearman(X[h] @ w, y[h]))
        inner.append(np.nanmean(rhos) if np.isfinite(rhos).any() else -np.inf)
    inner = np.array(inner)
    best = max(i for i in range(len(lams)) if inner[i] >= inner.max() - 1e-12)  # ties -> larger lam
    return FITTERS[kind](X, y, runs, lams[best]), lams[best], inner


def mean_within_run_rho(score, y, runs):
    return np.nanmean([exp_spearman(score[runs == r], y[runs == r]) for r in np.unique(runs)])


# ============================================================================ method specs
def method_specs(pool):
    """name -> spec.  kinds: 'free' (fixed score from one feature or a rank-average),
    'learn' (fit on training runs), 'select', 'unit', 'ref'."""
    S = {"epoch": dict(kind="free", feats=["epoch"])}
    for m in CHEAP:
        S[m] = dict(kind="free", feats=[m])
    for m in CHEAP:
        S[f"RA(epoch,{m})"] = dict(kind="free", feats=["epoch", m])
    S["RA(all cheap, no epoch)"] = dict(kind="free", feats=list(CHEAP))
    S["RA(epoch + all cheap)"] = dict(kind="free", feats=["epoch"] + CHEAP)
    all_f = ["epoch"] + CHEAP
    sets = {"all": all_f, "noepoch": list(CHEAP), "small": SMALL}
    if pool == "eval33":
        for j in ["k10_sim"] + list(JUDGES):
            S[j] = dict(kind="free", feats=[j])
            S[f"RA(epoch,{j})"] = dict(kind="free", feats=["epoch", j])
        S["RA(epoch,val_action_l1,vlm_nemotron_v2)"] = dict(kind="free",
                                                           feats=["epoch", "val_action_l1", "vlm_nemotron_v2"])
        sets = {"all": all_f, "small": SMALL, "all+vlm": all_f + VLM2,
                "epoch+vlm": ["epoch"] + VLM2, "small+vlm": SMALL + VLM2}
    for sname, feats in sets.items():
        for norm in ("z", "rank"):
            S[f"pairlogit_{norm}[{sname}]"] = dict(kind="learn", model="pairlogit", norm=norm, feats=feats)
        S[f"ridge_rank[{sname}]"] = dict(kind="learn", model="ridge", norm="rank", feats=feats)
        S[f"unitweight[{sname}]"] = dict(kind="unit", feats=feats)
    cand = [k for k, v in S.items() if v["kind"] == "free"]
    S["select_best_on_train"] = dict(kind="select", cands=cand)
    S["oracle_HA25 (ref, uses held-out run HA)"] = dict(kind="ref")
    S["random (ref)"] = dict(kind="random")
    return S


def free_score(spec, NR, idx):
    """Gradient-free score: mean of within-run ranks of the oriented features."""
    return np.mean([NR["rank"][f][idx] for f in spec["feats"]], axis=0)


def run_fold(specs, NR, runs, yA, yB, idx_by_run, hold, want_weights=False, record_w=False):
    """One leave-one-run-out fold.  idx_by_run: run -> array of row indices (may
    contain duplicates in a bootstrap replicate)."""
    tr = np.concatenate([idx_by_run[r] for r in RUNS if r != hold])
    te = idx_by_run[hold]
    rtr = runs[tr]
    res, extra = {}, {}
    free_cache = {}
    for name, sp in specs.items():
        k = sp["kind"]
        if k == "free":
            sc = free_score(sp, NR, te)
            free_cache[name] = sp
        elif k == "learn":
            Xtr = np.column_stack([NR[sp["norm"]][f][tr] for f in sp["feats"]])
            Xte = np.column_stack([NR[sp["norm"]][f][te] for f in sp["feats"]])
            w, lam, inner = nested_fit(sp["model"], Xtr, yA[tr], rtr)
            sc = Xte @ w
            if record_w:
                extra[name] = (w / max(np.linalg.norm(w), 1e-300)).tolist()
            if want_weights:
                wf = FITTERS[sp["model"]](Xtr, yA[tr], rtr, FIXED_LAM)
                extra[name] = {"lam": lam, "w": w.tolist(), "w_fixed_lam": wf.tolist(),
                               "inner_rho": inner.tolist(), "feats": sp["feats"]}
                # sensitivity: held-out stats at every fixed lambda (post-hoc upper bound)
                extra[name]["heldout_by_lam"] = {
                    str(l_): sel_stats(Xte @ FITTERS[sp["model"]](Xtr, yA[tr], rtr, l_), yB[te])
                    for l_ in LAMS}
        elif k == "unit":
            signs = []
            for f in sp["feats"]:
                rho = mean_within_run_rho(NR["rank"][f][tr], yA[tr], rtr)
                signs.append(1.0 if (not np.isfinite(rho) or rho >= 0) else -1.0)
            sc = np.sum([s * NR["rank"][f][te] for s, f in zip(signs, sp["feats"])], axis=0)
            if want_weights:
                extra[name] = {"signs": signs, "feats": sp["feats"]}
        elif k == "select":
            best, bname = -np.inf, "epoch"     # fallback when every training rho is undefined
            for c in sp["cands"]:
                rho = mean_within_run_rho(free_score(specs[c], NR, tr), yA[tr], rtr)
                if np.isfinite(rho) and rho > best + 1e-12:
                    best, bname = rho, c
            sc = free_score(specs[bname], NR, te)
            if want_weights:
                extra[name] = {"chosen": bname, "train_rho": best}
        elif k == "ref":
            sc = yA[te]
        elif k == "random":
            sc = np.zeros(len(te))
        res[name] = sel_stats(sc, yB[te])
    return res, extra


def aggregate(per_fold):
    """per_fold: hold -> name -> stats.  Returns name -> key -> mean over folds."""
    names = per_fold[RUNS[0]].keys()
    return {n: {k: float(np.nanmean([per_fold[h][n][k] for h in RUNS])) for k in STAT_KEYS} for n in names}


def run_level_ci(per_fold, name, key, ref="epoch"):
    vals = np.array([per_fold[h][name][key] for h in RUNS])
    dv = vals - np.array([per_fold[h][ref][key] for h in RUNS])
    means = [vals[list(c)].mean() for c in itertools.product(range(3), repeat=3)]
    dmeans = [dv[list(c)].mean() for c in itertools.product(range(3), repeat=3)]
    return {"mean": float(vals.mean()), "ci": [float(np.quantile(means, .025, method="inverted_cdf")),
                                               float(np.quantile(means, .975, method="inverted_cdf"))],
            "per_run": dict(zip(RUNS, vals.tolist())),
            "delta_vs_epoch": float(dv.mean()),
            "delta_ci": [float(np.quantile(dmeans, .025, method="inverted_cdf")),
                         float(np.quantile(dmeans, .975, method="inverted_cdf"))],
            "delta_per_run": dict(zip(RUNS, dv.tolist())),
            "n_runs_better_than_epoch": int((dv > 1e-12).sum() if key == "spearman" else (dv < -1e-12).sum()),
            "n_runs_worse_than_epoch": int((dv < -1e-12).sum() if key == "spearman" else (dv > 1e-12).sum())}


def block_resample(n, L, rng):
    if L >= n:
        L = max(1, n // 2)
    nb = int(math.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=nb)
    return np.concatenate([np.arange(s, s + L) for s in starts])[:n]


# ============================================================================ main
_CTX: dict = {}


def _features(ck, runs, rows, Ek, Vk):
    return normalise(raw_oriented(ck, Ek, Vk), runs, rows)


def _boot_one(b):
    c = _CTX
    rng = np.random.default_rng([c["seed"], b])
    Y, E, V = c["Y"], c["E"], c["V"]
    ia = rng.choice(HA, size=len(HA), replace=True)
    ib = rng.choice(HB, size=len(HB), replace=True)
    ie = rng.integers(0, 10, size=10)
    yAb, yBb = Y[:, ia].mean(1), Y[:, ib].mean(1)
    if c["pool"] == "eval33":
        NRb = _features(c["ck"], c["runs"], c["rows"], E[:, ie], {j: a[:, ie] for j, a in V.items()})
    else:
        NRb = c["NR"]
    if c["no_blocks"]:   # episode-only bootstrap: same checkpoints, resampled env seeds
        idx_b = dict(c["base_idx"])
    else:
        idx_b = {r: c["base_idx"][r][block_resample(len(c["base_idx"][r]), c["Lb"][r], rng)] for r in RUNS}
    pf, wrec = {}, {}
    for h in RUNS:
        pf[h], wrec[h] = run_fold(c["specs"], NRb, c["runs"], yAb, yBb, idx_b, h, record_w=True)
    return aggregate(pf), pf, wrec


def analyse_pool(pool, ck, Y, E, V, has_eval, n_boot, seed, log, n_workers=8, no_blocks=False):
    if pool == "eval33":
        rows = has_eval.copy()
    elif pool == "late93":   # outcome-free subset: epoch >= run median epoch (plateau regime)
        med = ck.groupby("run_id").epoch.transform("median").values
        rows = ck.epoch.values >= med
    else:
        rows = np.ones(len(ck), bool)
    runs = ck.run_id.values
    specs = method_specs(pool)
    base_idx = {r: np.where(rows & (runs == r))[0] for r in RUNS}  # sorted by epoch

    NR = _features(ck, runs, rows, E, V)
    yA, yB = Y[:, HA].mean(1), Y[:, HB].mean(1)
    yA_swap, yB_swap = yB, yA

    t0 = time.time()
    point, weights = {}, {}
    for h in RUNS:
        point[h], weights[h] = run_fold(specs, NR, runs, yA, yB, base_idx, h, want_weights=True)
    swap = {h: run_fold(specs, NR, runs, yA_swap, yB_swap, base_idx, h)[0] for h in RUNS}
    log(f"[{pool}] point estimates done in {time.time() - t0:.1f}s ({len(specs)} methods)")

    # ---- block + episode bootstrap with refitting (parallel, per-replicate RNG)
    spacing = {r: float(np.median(np.diff(ck.epoch.values[base_idx[r]]))) for r in RUNS}
    Lb = {r: max(2, int(round(BLOCK_EPOCHS / spacing[r]))) if pool != "eval33" else 2 for r in RUNS}
    _CTX.clear()
    _CTX.update(dict(pool=pool, ck=ck, Y=Y, E=E, V=V, runs=runs, rows=rows, specs=specs, NR=NR,
                     base_idx=base_idx, Lb=Lb, seed=seed, no_blocks=no_blocks))
    boot = {n: {k: [] for k in STAT_KEYS} for n in specs}
    boot_fold = {h: {n: {k: [] for k in STAT_KEYS} for n in specs} for h in RUNS}
    boot_w = {h: {} for h in RUNS}
    t0 = time.time()
    with mp.get_context("fork").Pool(n_workers) as P:
        for b, (ag, pf, wrec) in enumerate(P.imap(_boot_one, range(n_boot), chunksize=8)):
            for n in specs:
                for k in STAT_KEYS:
                    boot[n][k].append(ag[n][k])
                    for h in RUNS:
                        boot_fold[h][n][k].append(pf[h][n][k])
            for h in RUNS:
                for n, w in wrec[h].items():
                    boot_w[h].setdefault(n, []).append(w)
            if (b + 1) % max(1, n_boot // 10) == 0:
                log(f"[{pool}] boot {b + 1}/{n_boot}  {time.time() - t0:.0f}s")

    agg = aggregate(point)
    agg_swap = aggregate(swap)
    out_methods = {}
    for n, sp in specs.items():
        d = {"kind": sp["kind"], "feats": sp.get("feats"), "heldout_HB": agg[n],
             "swap_fit_HB_eval_HA": agg_swap[n], "per_run": {h: point[h][n] for h in RUNS},
             "run_level": {k: run_level_ci(point, n, k) for k in STAT_KEYS}, "block_boot": {}}
        for k in STAT_KEYS:
            a = np.array(boot[n][k], float)
            e = np.array(boot["epoch"][k], float)
            dd = a - e
            ok = np.isfinite(dd)
            d["block_boot"][k] = {
                "ci": [float(np.nanquantile(a, .025)), float(np.nanquantile(a, .975))],
                "delta_vs_epoch_ci": [float(np.quantile(dd[ok], .025)), float(np.quantile(dd[ok], .975))],
                "p_delta_gt0": float((dd[ok] > 1e-12).mean()), "p_delta_lt0": float((dd[ok] < -1e-12).mean()),
                "delta_quantiles_10_25_50_75_90": [float(np.quantile(dd[ok], q)) for q in (.1, .25, .5, .75, .9)],
                "n_finite": int(ok.sum()),
                "per_run_delta_ci": {h: [float(np.nanquantile(np.array(boot_fold[h][n][k]) - np.array(boot_fold[h]["epoch"][k]), q))
                                         for q in (.025, .975)] for h in RUNS}}
        out_methods[n] = d
    # bootstrap sign stability of the learned weights (nested lambda, per fold)
    wboot = {}
    for h in RUNS:
        for n, ws in boot_w[h].items():
            W = np.array(ws)
            pt = np.sign(np.array(weights[h][n]["w"]))
            wboot.setdefault(n, {})[h] = {
                "feats": weights[h][n]["feats"],
                "frac_same_sign_as_point": dict(zip(weights[h][n]["feats"], (np.sign(W) == pt).mean(0).round(3).tolist())),
                "frac_positive": dict(zip(weights[h][n]["feats"], (W > 0).mean(0).round(3).tolist())),
                "median_normalised_w": dict(zip(weights[h][n]["feats"], np.median(W, 0).round(4).tolist()))}
    return {"n_checkpoints": int(rows.sum()), "per_run_n": {r: int(len(base_idx[r])) for r in RUNS},
            "block_len_ckpts": Lb, "methods": out_methods, "fold_details": weights, "weight_bootstrap": wboot}


def weight_stability(fold_details, specs_names):
    out = {}
    for n in specs_names:
        if n not in fold_details[RUNS[0]] or "w" not in fold_details[RUNS[0]][n]:
            continue
        feats = fold_details[RUNS[0]][n]["feats"]
        res = {"feats": feats}
        for key in ("w", "w_fixed_lam"):
            W = np.array([fold_details[h][n][key] for h in RUNS])
            Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-300)
            cos = [float(Wn[i] @ Wn[j]) for i, j in itertools.combinations(range(3), 2)]
            res[key] = {"per_fold_normalised": {h: dict(zip(feats, Wn[i].round(4).tolist())) for i, h in enumerate(RUNS)},
                        "sign_agree_all_folds": dict(zip(feats, (np.abs(np.sign(W).sum(0)) == 3).tolist())),
                        "pairwise_cosine": cos, "mean_cosine": float(np.mean(cos))}
        res["lam_chosen"] = {h: fold_details[h][n]["lam"] for h in RUNS}
        out[n] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", default=str(ROOT / f"results/hackathon/explore/{NAME}.json"))
    ap.add_argument("--pools", default="all93,eval33")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-blocks", action="store_true", help="episode-only bootstrap (no checkpoint blocks)")
    args = ap.parse_args()
    t_start = time.time()

    def log(msg):
        print(msg, flush=True)

    ck, Y, E, V, has_eval = load_data()
    log(f"loaded {len(ck)} checkpoints, {int(has_eval.sum())} eval checkpoints")
    result = {"name": NAME, "label": "EXPLORATORY / post-hoc",
              "protocol": {"folds": "leave-one-run-out (3)", "fit_labels": "HA seeds 100000-100024 of training runs",
                           "target": "HB seeds 100025-100049 of held-out run",
                           "lambda_grid": LAMS, "fixed_lambda_for_weight_report": FIXED_LAM,
                           "inner_cv": "leave-one-run-out over the 2 training runs, HA labels, mean within-run Spearman",
                           "features": "oriented (higher=better a priori): epoch, -log(cost) for cheap metrics, K10 rates;"
                                       " within-run z-score or rank in [0,1] over the pool rows",
                           "bootstrap": "moving blocks of checkpoints within each run + HA/HB/eval env seeds, refit per replicate",
                           "n_boot": args.n_boot, "seed": args.seed, "no_blocks": args.no_blocks},
              "pools": {}}
    for pool in args.pools.split(","):
        r = analyse_pool(pool, ck, Y, E, V, has_eval, args.n_boot, args.seed + (0 if pool == "all93" else 1), log,
                         n_workers=args.workers, no_blocks=args.no_blocks)
        r["weight_stability"] = weight_stability(r["fold_details"], r["methods"].keys())
        result["pools"][pool] = r

    # noise ceiling: Spearman(HA, HB) inside each run
    runs = ck.run_id.values
    result["noise_ceiling_HA_vs_HB"] = {
        pool: {h: exp_spearman(Y[m, :][:, HA].mean(1), Y[m, :][:, HB].mean(1))
               for h in RUNS for m in [((runs == h) & (has_eval if pool == "eval33" else True))]}
        for pool in ("all93", "eval33")}

    def sha(p):
        return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    result["inputs"] = {"baseline_metrics": sha("results/hackathon/baseline_metrics.csv"),
                        **{j: sha(f) for j, f in JUDGES.items()},
                        **{f"logs_{r}": sha(f"{d}/logs.json.txt") for r, d in RUN_DIRS.items()}}
    result["runtime_s"] = round(time.time() - t_start, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, default=float))
    log(f"wrote {args.out} in {result['runtime_s']}s")


if __name__ == "__main__":
    main()
