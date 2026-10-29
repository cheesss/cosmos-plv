#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- y1_plv_prior_budget.

Question
--------
"Cheap metrics first": does screening the checkpoint pool with an outcome-free PLV metric
(keep the top 30 %) reduce the rollout budget needed to find a near-best checkpoint,
compared with (a) no prior (all checkpoints), (b) a random top-30 % subset, and (c) the
free "latest epoch" top-30 % prior?

Protocol (no target leakage)
----------------------------
* Data: 93 Diffusion-Policy checkpoints in 3 runs; 50 training-time rollouts each
  (``test/sim_max_reward_<seed>``, seeds 100000-100049) read by x1's ``load_runs``.
* Seeds 100000-100024 = rollout POOL (the only outcomes a strategy can observe, in a random
  order shared by all checkpoints); seeds 100025-100049 = HELD-OUT, used only to score the pick:
  regret = max_c H(c) - H(pick).
* Priors use only outcome-free columns of results/hackathon/baseline_metrics.csv (PLV
  metrics, logged val loss, epoch).  The ``success`` column is read only for an alignment
  sanity check (it must equal the 50-episode mean of the logs) and never enters a prior.
* Strategies (x1 code, unmodified, imported): uniform, successive halving, Thompson,
  top-two Thompson (TTTS); textbook defaults, never tuned on held-out outcomes.
* Judges: perfect; Nemotron (eval_nebius_v2) and Cosmos3-Edge (eval_local_v6) error models
  re-estimated from the FINAL 330-episode eval JSONLs (seeds 200000+, disjoint from the
  training-time seeds):
    _iid : per-episode flips, task-specific FPR/FNR ~ Beta(1+k, 1+n-k) per repetition (x1).
    _bb  : checkpoint-specific FPR_c ~ Beta(mu*kappa, (1-mu)*kappa) (beta-binomial,
           within-checkpoint error correlation); mu ~ task Beta posterior, kappa drawn from
           the profile likelihood of the per-checkpoint FP counts.  FNR as _iid.
    _bbq : like _bb but logit mu_c = a_task + b*(s_c - 0.5): FPR rises with the
           checkpoint's success s_c (near-miss failures of good checkpoints are called
           success).  Fitted on eval truth; in the simulation s_c = the checkpoint's pool
           success (a world-model choice, never seen by strategies).  (a, b) ~ Laplace
           approximation, kappa ~ profile-likelihood weights on a grid.
* Uncertainty: seed bootstrap (resample the 25 pool seeds and the 25 held-out seeds with
  replacement, the same resample for every checkpoint and run = block bootstrap over seeds),
  stacked into the repetition axis: replicate 0 = original seeds (point estimate, R0 reps);
  replicates 1..B with R_in reps each (episode order / judge noise).  CIs are percentile
  intervals over replicates (they include residual Monte-Carlo noise -> conservative).
  Paired differences use common random numbers (order, judge noise, judge rates) and are
  computed within replicate.
* Truncated-history stress test: pretend training stopped at epoch E (can 60/100/150,
  square 90/150/240), keep checkpoints <= E, recompute priors and regret within that set.

Run from the repo root:
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y1_plv_prior_budget.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import betaln, expit  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x1_adaptive_budget as x1  # noqa: E402  (imported, never modified)
from cosmos_plv.provenance import git_commit, utc_now  # noqa: E402
from cosmos_plv.ranking import spearman  # noqa: E402

NAME = "y1_plv_prior_budget"
MULTS = (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, 25)
SUMMARY_MULTS = (0.5, 1, 1.5, 2, 3, 4, 6, 8)  # "average regret over 0.5N-8N"
THRS = (0.0, 0.01, 0.02)  # excess-regret thresholds vs the perfect no-prior uniform 25N reference
STRATS = ("uniform", "successive_halving", "thompson", "top_two_thompson")
SINGLE = {  # name -> (column, lower_is_better)
    "val_action_l1": ("val_action_l1", True),
    "open_loop_smoothness": ("open_loop_smoothness", True),
    "plv1_smoothness": ("plv1_smoothness", True),
    "replan_overlap_h16": ("replan_overlap_h16", True),
    "boundary_jump_h16": ("boundary_jump_h16", True),
    "val_loss_logged": ("val_loss_logged", True),
    "epoch": ("epoch", False),
}
COMBOS = {  # rank-average of two oriented metrics
    "epoch+replan_overlap_h16": ("epoch", "replan_overlap_h16"),
    "epoch+open_loop_smoothness": ("epoch", "open_loop_smoothness"),
    "epoch+val_action_l1": ("epoch", "val_action_l1"),
    "epoch+boundary_jump_h16": ("epoch", "boundary_jump_h16"),
    "epoch+plv1_smoothness": ("epoch", "plv1_smoothness"),
    "val_action_l1+replan_overlap_h16": ("val_action_l1", "replan_overlap_h16"),
}
PRIORS = ["none", "random", *SINGLE, *COMBOS]
PLV_PRIORS = [p for p in PRIORS if p not in ("none", "random", "epoch")]
JUDGES = ["perfect", "nemotron_iid", "nemotron_bb", "nemotron_bbq", "cosmos_iid", "cosmos_bb", "cosmos_bbq"]
JUDGE_SRC = {"nemotron": "nemotron_nebius_v2", "cosmos": "cosmos3_edge_local_v6"}
JUDGE_FILE = {"nemotron": "eval_nebius_v2.jsonl", "cosmos": "eval_local_v6.jsonl"}
TRUNC = {"can": (60, 100, 150), "square": (90, 150, 240)}
KAPPA_GRID = np.exp(np.linspace(math.log(0.25), math.log(1e4), 41))


def h(s):
    return zlib.crc32(s.encode())


# --------------------------------------------------------------------------- judge models
def per_checkpoint_counts(stem):
    """checkpoint_id -> dict(task, n_fail, fp, n_succ, fn, n, s_eval)."""
    d = {}
    for line in (ROOT / "data/outputs/hackathon_cosmos" / stem).read_text().splitlines():
        r = json.loads(line)
        if not (r.get("scored") and r.get("parse_status") == "ok"):
            continue
        if not isinstance(r.get("truth_success"), bool) or not isinstance(r.get("cosmos_task_success"), bool):
            continue
        c = d.setdefault(r["checkpoint_id"], {"task": r["task"], "n_fail": 0, "fp": 0, "n_succ": 0, "fn": 0})
        if r["truth_success"]:
            c["n_succ"] += 1
            c["fn"] += int(not r["cosmos_task_success"])
        else:
            c["n_fail"] += 1
            c["fp"] += int(r["cosmos_task_success"])
    for c in d.values():
        c["n"] = c["n_fail"] + c["n_succ"]
        c["s_eval"] = c["n_succ"] / c["n"]
    return d


def fit_bb(d):
    """Beta-binomial on per-checkpoint FP among failures: task mu, shared kappa; profile over kappa."""
    tasks = sorted({v["task"] for v in d.values()})
    data = [(tasks.index(v["task"]), v["n_fail"], v["fp"]) for v in d.values() if v["n_fail"] > 0]

    def nll_mu(th, k):
        mus = expit(th)
        a, b = mus * k, (1 - mus) * k
        return -sum(betaln(x + a[t], n - x + b[t]) - betaln(a[t], b[t]) for t, n, x in data)

    prof, mus_at = [], []
    th0 = np.full(len(tasks), -1.5)
    for k in KAPPA_GRID:
        r = minimize(lambda th: nll_mu(th, k), th0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9})
        prof.append(float(r.fun))
        mus_at.append(expit(r.x).tolist())
    prof = np.array(prof)
    w = np.exp(-(prof - prof.min()))
    w /= w.sum()
    i = int(prof.argmin())
    binom = float(prof[-1])
    return {"tasks": tasks, "kappa_grid": KAPPA_GRID.tolist(), "profile_nll": prof.tolist(), "kappa_weights": w.tolist(),
            "kappa_mle_grid": float(KAPPA_GRID[i]), "mu_at_mle": dict(zip(tasks, mus_at[i])),
            "lr_stat_vs_binomial_limit": float(2 * (binom - prof.min())),
            "kappa_posterior_median": float(KAPPA_GRID[np.searchsorted(np.cumsum(w), 0.5)]),
            "p_kappa_le_10": float(w[KAPPA_GRID <= 10].sum()),
            "per_checkpoint_fp_over_fail": {k: f"{v['fp']}/{v['n_fail']}" for k, v in sorted(d.items()) if v["n_fail"]}}


def fit_bbq(d):
    """logit mu_c = a_task + b*(s_eval_c - 0.5); FP_c ~ BetaBin(n_fail_c, mu_c*kappa, (1-mu_c)*kappa).
    Laplace approximation on (a_can, a_square, b, log kappa)."""
    tasks = sorted({v["task"] for v in d.values()})
    data = [(tasks.index(v["task"]), v["n_fail"], v["fp"], v["s_eval"]) for v in d.values() if v["n_fail"] > 0]
    lo_k, hi_k = math.log(0.25), math.log(1e4)

    def nll(th):
        lk = min(max(th[-1], lo_k), hi_k)
        k = math.exp(lk)
        s = 0.0
        for t, n, x, se in data:
            mu = expit(th[t] + th[len(tasks)] * (se - 0.5))
            mu = min(max(mu, 1e-9), 1 - 1e-9)
            s -= betaln(x + mu * k, n - x + (1 - mu) * k) - betaln(mu * k, (1 - mu) * k)
        return s

    best = None
    for lk0 in (0.0, 1.5, 3.0, 6.0):
        r = minimize(nll, np.array([-1.5] * len(tasks) + [2.0, lk0]), method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    th = best.x.copy()
    th[-1] = min(max(th[-1], lo_k), hi_k)
    # numerical Hessian
    p = len(th)
    eps = 1e-3
    H = np.zeros((p, p))
    f0 = nll(th)
    for i in range(p):
        for j in range(i, p):
            e_i, e_j = np.eye(p)[i] * eps, np.eye(p)[j] * eps
            v = (nll(th + e_i + e_j) - nll(th + e_i - e_j) - nll(th - e_i + e_j) + nll(th - e_i - e_j)) / (4 * eps * eps)
            H[i, j] = H[j, i] = v
    try:
        cov = np.linalg.inv(H)
        ok = bool(np.all(np.linalg.eigvalsh(cov) > 0))
    except np.linalg.LinAlgError:
        ok = False
    if not ok:  # fall back to diagonal curvature
        cov = np.diag(1.0 / np.maximum(np.diag(H), 1e-3))
    # binomial (no overdispersion) slope fit, for reference
    nll_b = lambda t: nll(np.concatenate([t, [hi_k]]))  # noqa: E731
    rb = minimize(nll_b, th[:-1], method="Nelder-Mead", options={"maxiter": 20000})
    # profile likelihood over the kappa grid (log kappa is weakly identified / at the bound, so the
    # simulation draws kappa from these weights and (a, b) from the Laplace block cov[:-1, :-1])
    prof = []
    for k in KAPPA_GRID:
        rk = minimize(lambda t: nll(np.concatenate([t, [math.log(k)]])), th[:-1], method="Nelder-Mead",
                      options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-10})
        prof.append(float(rk.fun))
    prof = np.array(prof)
    w = np.exp(-(prof - prof.min()))
    w /= w.sum()
    return {"tasks": tasks, "theta_mle": th.tolist(), "theta_names": [f"a_{t}" for t in tasks] + ["b_slope", "log_kappa"],
            "cov": cov.tolist(), "laplace_ok": ok, "nll": float(f0), "nll_no_overdispersion": float(rb.fun),
            "theta_no_overdispersion": rb.x.tolist(),
            "kappa_grid": KAPPA_GRID.tolist(), "profile_nll": prof.tolist(), "kappa_weights": w.tolist(),
            "p_kappa_le_10": float(w[KAPPA_GRID <= 10].sum()),
            "slope_se": float(math.sqrt(max(cov[len(tasks), len(tasks)], 0))),
            "implied_fpr_at_s": {t: {str(s): float(expit(th[ti] + th[len(tasks)] * (s - 0.5))) for s in (0.0, 0.5, 0.8, 0.9, 1.0)}
                                 for ti, t in enumerate(tasks)}}


def judge_rates(judge, task, R, n, rng, pool_succ, counts, fits):
    """(fpr [R,1|n], fnr [R,1]) or None (perfect)."""
    if judge == "perfect":
        return None
    fam, mode = judge.split("_")
    c = counts[JUDGE_SRC[fam]].get(task) or counts[JUDGE_SRC[fam]]["all"]
    fnr = rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"], size=(R, 1))
    if mode == "iid":
        return rng.beta(1 + c["fp"], 1 + c["n_fail"] - c["fp"], size=(R, 1)), fnr
    if mode == "bb":
        f = fits[fam]["bb"]
        mu = np.clip(rng.beta(1 + c["fp"], 1 + c["n_fail"] - c["fp"], size=(R, 1)), 1e-6, 1 - 1e-6)
        kap = rng.choice(np.array(f["kappa_grid"]), size=(R, 1), p=np.array(f["kappa_weights"]))
        return rng.beta(mu * kap, (1 - mu) * kap, size=(R, n)), fnr
    if mode == "bbq":
        f = fits[fam]["bbq"]
        th = rng.multivariate_normal(np.array(f["theta_mle"][:-1]), np.array(f["cov"])[:-1, :-1], size=R)
        ti = f["tasks"].index(task)
        nt = len(f["tasks"])
        mu = np.clip(expit(th[:, ti:ti + 1] + th[:, nt:nt + 1] * (pool_succ - 0.5)), 1e-6, 1 - 1e-6)
        kap = rng.choice(np.array(f["kappa_grid"]), size=(R, 1), p=np.array(f["kappa_weights"]))
        return rng.beta(mu * kap, (1 - mu) * kap), fnr
    raise ValueError(judge)


# --------------------------------------------------------------------------- data
def load(baseline_csv):
    import pandas as pd
    runs = x1.load_runs(ROOT / "data/outputs/hackathon_pool")
    df = pd.read_csv(baseline_csv).set_index("checkpoint_id")
    cols = sorted({c for c, _ in SINGLE.values()} - {"epoch"})
    checks = {}
    for r in runs:
        missing = [i for i in r["ids"] if i not in df.index]
        assert not missing, missing
        for c in cols:
            r["metrics"][c] = df.loc[r["ids"], c].astype(float).values
        # alignment sanity check only (never used by priors)
        s = df.loc[r["ids"], "success"].astype(float).values
        checks[r["run"]] = {"n": len(r["ids"]), "max_abs_diff_success_vs_logs": float(np.abs(s - r["Y"].mean(1)).max()),
                            "epochs_match": bool(np.all(df.loc[r["ids"], "epoch"].values == r["epochs"]))}
        r["short"] = str(df.loc[r["ids"][0], "run_id"])
    return runs, checks


def prior_order(metrics, epochs, prior):
    """Best-first local order for a deterministic prior (ties -> later epoch)."""
    n = len(epochs)

    def key(name):
        col, lower = SINGLE[name]
        v = np.asarray(metrics[col], float)
        return np.where(np.isfinite(v), v if lower else -v, np.inf)

    if prior in SINGLE:
        k = key(prior)
    elif prior in COMBOS:
        from scipy.stats import rankdata
        k = np.mean([rankdata(key(m)) for m in COMBOS[prior]], axis=0)
    else:
        raise ValueError(prior)
    return sorted(range(n), key=lambda i: (k[i], -epochs[i]))


def problems_list(runs):
    probs = []
    for ri, r in enumerate(runs):
        probs.append({"run_i": ri, "run": r["short"], "task": r["task"], "level": "full", "E": None,
                      "idx": np.arange(len(r["epochs"]))})
    for li, lvl in enumerate(("trunc_1", "trunc_2", "trunc_3")):
        for ri, r in enumerate(runs):
            E = TRUNC[r["task"]][li]
            probs.append({"run_i": ri, "run": r["short"], "task": r["task"], "level": lvl, "E": E,
                          "idx": np.flatnonzero(r["epochs"] <= E)})
    return probs


# --------------------------------------------------------------------------- simulation
G = {}  # fork-shared globals


def sim_task(args):
    pi, judge = args
    prob = G["probs"][pi]
    r = G["runs"][prob["run_i"]]
    idx = prob["idx"]
    Y = r["Y"][idx]
    n = len(idx)
    epochs = r["epochs"][idx]
    metrics = {k: np.asarray(v)[idx] for k, v in r["metrics"].items()}
    pool_cols, ho_cols, rep_b = G["pool_cols"], G["ho_cols"], G["rep_b"]
    R, P = pool_cols.shape
    nb = rep_b.max() + 1
    seed = G["seed"]
    base = np.random.SeedSequence([seed, pi])  # shared by all judges -> common random numbers
    c_order, c_noise, c_jit, c_sub = [np.random.default_rng(s) for s in base.spawn(4)]
    Yp = np.ascontiguousarray(Y[:, pool_cols].transpose(1, 0, 2))      # [R, n, P]
    Hm = Y[:, ho_cols].mean(2).T                                       # [R, n]
    perm = np.broadcast_to(np.argsort(c_order.random((R, 1, P)), axis=2), (R, n, P))
    Yord = np.take_along_axis(Yp, perm, axis=2)
    U = c_noise.random((R, n, P))
    jitter = c_jit.random((R, n)) * 1e-9
    budgets = [max(1, int(round(m * n))) for m in MULTS]
    k30 = max(1, math.ceil(0.3 * n))
    arms_by_prior = {"none": np.broadcast_to(np.arange(n), (R, n)),
                     "random": np.argsort(c_sub.random((R, n)), axis=1)[:, :k30]}
    for p in PRIORS[2:]:
        arms_by_prior[p] = np.broadcast_to(np.array(prior_order(metrics, epochs, p)[:k30]), (R, k30))
    rates = judge_rates(judge, prob["task"], R, n, np.random.default_rng([seed, pi, h(judge)]),
                        Yp.mean(2), G["counts"], G["fits"])
    if rates is None:
        Yobs = Yord
    else:
        fpr, fnr = rates
        fpr = fpr[:, :, None] if fpr.ndim == 2 else fpr
        Yobs = np.where(Yord > 0.5, U >= fnr[:, :, None], U < fpr).astype(np.float64)
    prefix_full = np.concatenate([np.zeros((R, n, 1)), np.cumsum(Yobs, axis=2)], axis=2).round().astype(np.int64)
    best = Hm.max(1)
    cnt = np.bincount(rep_b, minlength=nb).astype(float)
    out = {}
    for p, arms in arms_by_prior.items():
        prefix = np.take_along_axis(prefix_full, arms[:, :, None], axis=1)
        jit = np.take_along_axis(jitter, arms, axis=1)
        for st in STRATS:
            srng = np.random.default_rng([seed, pi, h(p + "|" + st)])
            res = x1.run_strategy(st, prefix, budgets, srng, jit)
            M = np.zeros((len(MULTS), nb))
            used = np.zeros(len(MULTS))
            for mi, B in enumerate(budgets):
                pick_local, u = res[B]
                pick = np.take_along_axis(arms, pick_local[:, None], axis=1)[:, 0]
                reg = best - Hm[np.arange(R), pick]
                M[mi] = np.bincount(rep_b, weights=reg, minlength=nb) / cnt
                used[mi] = u
            out[(p, st)] = (M, used)
    # zero-rollout baselines (judge independent; computed with every judge, identical)
    zero = {}
    for p, arms in arms_by_prior.items():
        if p == "none":
            continue
        Hsub = np.take_along_axis(Hm, arms, axis=1)
        floor = best - Hsub.max(1)
        top1 = best - Hsub[:, 0]  # random prior: an arbitrary member
        zero[p] = (np.bincount(rep_b, weights=floor, minlength=nb) / cnt,
                   np.bincount(rep_b, weights=top1, minlength=nb) / cnt)
    return pi, judge, out, zero, budgets, n, k30


# --------------------------------------------------------------------------- summaries
def pct(v, q):
    return [float(x) for x in np.percentile(v, q)]


def crossing(curve_b, ref_b, thr):
    """curve_b [len(MULTS), nb]; ref_b [nb] -> crossing multiple per replicate (inf if never)."""
    ex = curve_b - ref_b[None, :]
    ok = ex <= thr + 1e-12
    out = np.full(curve_b.shape[1], np.inf)
    for b in range(curve_b.shape[1]):
        k = np.flatnonzero(ok[:, b])
        if len(k):
            out[b] = MULTS[k[0]]
    return out


def cross_summary(c):
    fin = np.where(np.isfinite(c), c, 50.0)  # 50 = not reached within 25N
    boot = fin[1:]
    return {"point": None if not np.isfinite(c[0]) else float(c[0]),
            "boot_median": float(np.median(boot)), "boot_p10": float(np.percentile(boot, 10)),
            "boot_p90": float(np.percentile(boot, 90)),
            "share_reached_le_25N": float(np.isfinite(c[1:]).mean()),
            "share_le_1N": float((boot <= 1).mean()), "share_le_2N": float((boot <= 2).mean()),
            "share_le_4N": float((boot <= 4).mean())}


def main():
    ap = argparse.ArgumentParser(description="y1_plv_prior_budget (EXPLORATORY)")
    ap.add_argument("--baseline-csv", default=str(ROOT / "results/hackathon/baseline_metrics.csv"))
    ap.add_argument("--out", default=str(ROOT / f"results/hackathon/explore/{NAME}.json"))
    ap.add_argument("--fig", default=str(ROOT / f"docs/hackathon/explore/{NAME}_regret.png"))
    ap.add_argument("--r0", type=int, default=400, help="reps on the original seed split (point estimate)")
    ap.add_argument("--n-boot", type=int, default=200, help="seed-bootstrap replicates")
    ap.add_argument("--r-in", type=int, default=10, help="reps per bootstrap replicate")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--levels", default="full,trunc_1,trunc_2,trunc_3")
    ap.add_argument("--resample", choices=("bootstrap", "split"), default="bootstrap",
                    help="bootstrap = resample pool & held-out seeds with replacement; split = random 25/25 "
                         "re-partitions of the 50 seeds (no duplicated episodes; sensitivity)")
    a = ap.parse_args()
    t0 = time.time()

    runs, checks = load(a.baseline_csv)
    counts = x1.judge_counts()  # task-specific confusion counts from the FINAL eval JSONLs
    fits = {}
    for fam, stem in JUDGE_FILE.items():
        d = per_checkpoint_counts(stem)
        fits[fam] = {"bb": fit_bb(d), "bbq": fit_bbq(d), "n_episodes": int(sum(v["n"] for v in d.values())),
                     "n_checkpoints": len(d)}
    print(f"[{time.time() - t0:5.1f}s] judge fits done", flush=True)
    for fam in fits:
        print(fam, "bb kappa mle", fits[fam]["bb"]["kappa_mle_grid"], "LR", round(fits[fam]["bb"]["lr_stat_vs_binomial_limit"], 2),
              "| bbq theta", np.round(fits[fam]["bbq"]["theta_mle"], 2), "fpr(s)", fits[fam]["bbq"]["implied_fpr_at_s"], flush=True)

    # seed-bootstrap resample matrices (shared by every problem / run / judge)
    half = 25
    rng = np.random.default_rng(a.seed + 1)
    pc, hc, rb = [np.tile(np.arange(half), (a.r0, 1))], [np.tile(np.arange(half, 2 * half), (a.r0, 1))], [np.zeros(a.r0, int)]
    for b in range(1, a.n_boot + 1):
        if a.resample == "bootstrap":
            p_ = rng.integers(0, half, half)
            h_ = half + rng.integers(0, half, half)
        else:
            perm = rng.permutation(2 * half)
            p_, h_ = np.sort(perm[:half]), np.sort(perm[half:])
        pc.append(np.tile(p_, (a.r_in, 1)))
        hc.append(np.tile(h_, (a.r_in, 1)))
        rb.append(np.full(a.r_in, b))
    G.update(runs=runs, probs=problems_list(runs), pool_cols=np.concatenate(pc), ho_cols=np.concatenate(hc),
             rep_b=np.concatenate(rb), seed=a.seed, counts=counts, fits=fits)
    levels = a.levels.split(",")
    tasks = [(pi, j) for pi, p in enumerate(G["probs"]) if p["level"] in levels for j in JUDGES]
    # big problems first
    tasks.sort(key=lambda t: -len(G["probs"][t[0]]["idx"]))
    print(f"[{time.time() - t0:5.1f}s] {len(tasks)} sim tasks, R={len(G['rep_b'])} reps", flush=True)
    results, zeros, meta = {}, {}, {}
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, (pi, j, out, zero, budgets, n, k30) in enumerate(pool.imap_unordered(sim_task, tasks)):
                results[(pi, j)] = out
                zeros[pi] = zero
                meta[pi] = (budgets, n, k30)
                if k % 10 == 0:
                    print(f"[{time.time() - t0:6.1f}s] {k + 1}/{len(tasks)} done", flush=True)
    else:
        for k, t in enumerate(tasks):
            pi, j, out, zero, budgets, n, k30 = sim_task(t)
            results[(pi, j)], zeros[pi], meta[pi] = out, zero, (budgets, n, k30)
            print(f"[{time.time() - t0:6.1f}s] {k + 1}/{len(tasks)} done", flush=True)
    print(f"[{time.time() - t0:6.1f}s] simulation done", flush=True)

    probs = G["probs"]
    smi = [MULTS.index(m) for m in SUMMARY_MULTS]
    report = {}
    for lvl in levels:
        pis = [pi for pi, p in enumerate(probs) if p["level"] == lvl]
        L = {"problems": [{"run": probs[pi]["run"], "E": probs[pi]["E"], "n_checkpoints": meta[pi][1], "k30": meta[pi][2],
                           "budgets": dict(zip(map(str, MULTS), meta[pi][0])),
                           "epochs": runs[probs[pi]["run_i"]]["epochs"][probs[pi]["idx"]].tolist()} for pi in pis]}
        # curves: mean over runs per replicate
        curve = {}
        for j in JUDGES:
            for p in PRIORS:
                for st in STRATS:
                    curve[(j, p, st)] = np.mean([results[(pi, j)][(p, st)][0] for pi in pis], axis=0)  # [M, nb]
        ref = curve[("perfect", "none", "uniform")][MULTS.index(25)]  # [nb]
        L["reference_perfect_none_uniform_25N"] = {"point": float(ref[0]), "ci95": pct(ref[1:], [2.5, 97.5]),
                                                    "per_run_point": {probs[pi]["run"]: float(results[(pi, "perfect")][("none", "uniform")][0][MULTS.index(25), 0]) for pi in pis}}
        rows, cross_arr = [], {}
        for (j, p, st), c in curve.items():
            per_run = {probs[pi]["run"]: [round(float(results[(pi, j)][(p, st)][0][mi, 0]), 4) for mi in range(len(MULTS))] for pi in pis}
            used = np.mean([results[(pi, j)][(p, st)][1] / np.array(meta[pi][0]) for pi in pis], axis=0)
            avg = c[smi].mean(0)
            rows.append({"judge": j, "prior": p, "strategy": st,
                         "mean_regret": [round(float(x), 4) for x in c[:, 0]],
                         "ci95_lo": [round(float(x), 4) for x in np.percentile(c[:, 1:], 2.5, axis=1)],
                         "ci95_hi": [round(float(x), 4) for x in np.percentile(c[:, 1:], 97.5, axis=1)],
                         "avg_0.5N_8N": {"point": float(avg[0]), "ci95": pct(avg[1:], [2.5, 97.5])},
                         "frac_budget_used": [round(float(x), 3) for x in used],
                         "per_run_mean_regret": per_run,
                         "crossing_excess_le_thr_vs_ref": cross_summary(crossing(c, ref, a.threshold)),
                         "crossing_excess_le_thr_vs_same_judge_ref": cross_summary(
                             crossing(c, curve[(j, "none", "uniform")][MULTS.index(25)], a.threshold)),
                         "crossing_by_threshold_vs_ref": {str(t): cross_summary(crossing(c, ref, t)) for t in THRS}})
            for t in THRS:
                cross_arr[(j, p, st, t)] = np.where(np.isfinite(cx := crossing(c, ref, t)), cx, 50.0)
        L["rows"] = rows
        # paired differences: prior vs random / epoch / none (same judge & strategy), within replicate
        pairs = []
        for j in JUDGES:
            for st in STRATS:
                for p in [q for q in PRIORS if q != "none"]:
                    rec = {"judge": j, "strategy": st, "prior": p}
                    for ctrl in ("random", "epoch", "none"):
                        if ctrl == p:
                            continue
                        dlt = curve[(j, p, st)] - curve[(j, ctrl, st)]  # negative = prior better
                        avg = dlt[smi].mean(0)
                        rec[f"vs_{ctrl}"] = {
                            "by_mult": {str(m): {"point": round(float(dlt[mi, 0]), 4),
                                                 "ci95": [round(x, 4) for x in pct(dlt[mi, 1:], [2.5, 97.5])]}
                                        for mi, m in enumerate(MULTS) if m in (0.5, 1, 2, 4, 8)},
                            "avg_0.5N_8N": {"point": round(float(avg[0]), 4), "ci95": [round(x, 4) for x in pct(avg[1:], [2.5, 97.5])],
                                            "p_boot_prior_better": float((avg[1:] < 0).mean())}}
                    # paired crossing budgets (50 = not reached within 25N); ratio > 1 = control needs more
                    rec["crossing_paired"] = {}
                    for t in THRS:
                        cp = cross_arr[(j, p, st, t)]
                        dd = {}
                        for ctrl in ("random", "epoch", "none"):
                            if ctrl == p:
                                continue
                            cc_ = cross_arr[(j, ctrl, st, t)]
                            ratio = cc_[1:] / cp[1:]
                            dd[f"vs_{ctrl}"] = {"point_prior": float(cp[0]), "point_ctrl": float(cc_[0]),
                                                "ratio_ctrl_over_prior_median": float(np.median(ratio)),
                                                "ratio_p10_p90": pct(ratio, [10, 90]),
                                                "share_prior_earlier": float((cp[1:] < cc_[1:]).mean()),
                                                "share_prior_later": float((cp[1:] > cc_[1:]).mean())}
                        rec["crossing_paired"][str(t)] = dd
                    if p in PLV_PRIORS:
                        ar, ae = rec["vs_random"]["avg_0.5N_8N"], rec["vs_epoch"]["avg_0.5N_8N"]
                        rec["beats_both_avg_ci"] = bool(ar["ci95"][1] < 0 and ae["ci95"][1] < 0)
                        rec["beats_both_any_mult_ci"] = [m for m in ("0.5", "1", "2", "4", "8")
                                                         if rec["vs_random"]["by_mult"][m]["ci95"][1] < 0
                                                         and rec["vs_epoch"]["by_mult"][m]["ci95"][1] < 0]
                    pairs.append(rec)
        L["paired"] = pairs
        # zero-rollout: top-30% floor and top-1 pick
        zr = {}
        for p in PRIORS[1:]:
            fl = np.mean([zeros[pi][p][0] for pi in pis], axis=0)
            t1 = np.mean([zeros[pi][p][1] for pi in pis], axis=0)
            zr[p] = {"top30_floor_regret": {"point": float(fl[0]), "ci95": pct(fl[1:], [2.5, 97.5])},
                     "top1_pick_regret": {"point": float(t1[0]), "ci95": pct(t1[1:], [2.5, 97.5])},
                     "per_run_point_floor": {probs[pi]["run"]: float(zeros[pi][p][0][0]) for pi in pis},
                     "per_run_point_top1": {probs[pi]["run"]: float(zeros[pi][p][1][0]) for pi in pis}}
            if p != "random":
                fe = np.mean([zeros[pi]["epoch"][0] for pi in pis], axis=0)
                frd = np.mean([zeros[pi]["random"][0] for pi in pis], axis=0)
                zr[p]["floor_minus_epoch"] = {"point": float(fl[0] - fe[0]), "ci95": pct(fl[1:] - fe[1:], [2.5, 97.5])}
                zr[p]["floor_minus_random"] = {"point": float(fl[0] - frd[0]), "ci95": pct(fl[1:] - frd[1:], [2.5, 97.5])}
        L["zero_rollout"] = zr
        # subset composition + spearman vs held-out (original split)
        comp = {}
        for pi in pis:
            r = runs[probs[pi]["run_i"]]
            idx = probs[pi]["idx"]
            ep = r["epochs"][idx]
            mets = {k: np.asarray(v)[idx] for k, v in r["metrics"].items()}
            H0 = r["Y"][idx][:, 25:].mean(1)
            k30 = meta[pi][2]
            cc = {"heldout_best": float(H0.max()), "best_epochs": ep[H0 == H0.max()].tolist()}
            for p in PRIORS[2:]:
                o = prior_order(mets, ep, p)
                sc = np.empty(len(ep))
                sc[o] = -np.arange(len(ep))
                cc[p] = {"top30_epochs": sorted(int(ep[i]) for i in o[:k30]), "spearman_vs_heldout": spearman(sc, H0)}
            comp[probs[pi]["run"]] = cc
        L["subset_composition"] = comp
        report[lvl] = L
        print(f"[{time.time() - t0:6.1f}s] summarized {lvl}", flush=True)

    result = {
        "name": NAME, "exploratory": True,
        "label": "EXPLORATORY / post-hoc simulation on logged training-time rollouts; not a pre-registered test",
        "generated_utc": utc_now(), "git_commit": git_commit(), "runtime_s": round(time.time() - t0, 1),
        "command": " ".join([sys.executable, *sys.argv]), "args": vars(a),
        "protocol": {
            "pool_seeds": "100000-100024", "heldout_seeds": "100025-100049",
            "regret": "max_c H(c) - H(pick); H = held-out success (25 episodes) within the problem's checkpoint set",
            "budget_mults": list(MULTS), "budget": "B = round(mult * N), N = all checkpoints of the problem (prior or not)",
            "prior_subset": "top ceil(0.3 N) by the prior; ties -> later epoch; combos = mean of within-problem ranks",
            "strategies": list(STRATS), "priors": PRIORS, "plv_priors": PLV_PRIORS, "judges": JUDGES,
            "ci": "percentile over seed-bootstrap replicates (pool & held-out seeds resampled with replacement, shared "
                  "across checkpoints and runs); replicate 0 = original seeds, used for point estimates",
            "r0": a.r0, "n_boot": a.n_boot, "r_in": a.r_in, "resample": a.resample,
            "summary_avg": f"mean regret over mults {list(SUMMARY_MULTS)}",
            "crossing": f"smallest grid mult with (mean-over-runs regret - reference) <= {a.threshold}; reference = "
                        "perfect-judge no-prior uniform at 25N (all 25 pool episodes); 50 = not reached",
            "truncation": TRUNC,
        },
        "alignment_checks": checks,
        "judge_counts_final_eval": {k: {t: {kk: vv for kk, vv in v.items()} for t, v in d.items()}
                                    for k, d in counts.items() if k in JUDGE_SRC.values()},
        "judge_fits": fits,
        "levels": report,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"wrote {out} ({time.time() - t0:.1f}s)")
    if a.fig:
        try:
            make_fig(report, Path(a.fig))
            print("wrote", a.fig)
        except Exception as e:  # optional
            print("figure failed:", repr(e))


def make_fig(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("full", "perfect"), ("full", "nemotron_bb"), ("full", "cosmos_bb"), ("trunc_1", "perfect")]
    panels = [(lv, j) for lv, j in panels if lv in report]
    priors = ["none", "random", "epoch", "val_action_l1", "replan_overlap_h16", "epoch+replan_overlap_h16"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lv, j) in zip(axes, panels):
        ref = report[lv]["reference_perfect_none_uniform_25N"]["point"]
        for p in priors:
            row = [x for x in report[lv]["rows"] if x["judge"] == j and x["prior"] == p and x["strategy"] == "top_two_thompson"][0]
            ax.plot(MULTS, row["mean_regret"], marker="o", ms=3, label=p)
            ax.fill_between(MULTS, row["ci95_lo"], row["ci95_hi"], alpha=0.12)
        ax.axhline(ref + 0.02, color="k", ls=":", lw=1)
        ax.axhline(ref, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"{lv} / {j} / TTTS", fontsize=9)
        ax.set_xlabel("budget B / N")
    axes[0].set_ylabel("held-out regret (mean over runs)")
    axes[0].legend(fontsize=7)
    fig.suptitle("y1_plv_prior_budget (EXPLORATORY): dashed = perfect no-prior uniform 25N; dotted = +0.02", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
