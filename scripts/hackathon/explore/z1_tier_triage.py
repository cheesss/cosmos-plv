#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- z1_tier_triage.

Question
--------
A practitioner often needs a coarse verdict per checkpoint, not a fine ranking:
BAD (true success < 0.5), OK (0.5-0.9) or GOOD (> 0.9).  How many rollouts, judged by
(a) a perfect success detector, (b) Nemotron (Nebius, eval_nebius_v2), (c) Cosmos3-Edge
(local, eval_local_v6), are needed to assign the tier with >= 90 % accuracy, and does a
cheap PLV metric / epoch prior reduce that number?

Worlds (all CPU, no Nebius / GPU)
---------------------------------
A  "split" (primary, non-parametric): 93 checkpoints x 50 training-time rollouts
   (test/sim_max_reward_<seed>, seeds 100000-100049).  Per replicate a random 25/25 seed
   split (shared by all checkpoints): TRUTH tier = tier of the held-out 25-episode rate;
   the K rollouts are drawn without replacement from the other 25 (K = 1..25).  The fixed
   split (pool 100000-24 / held-out 100025-49) is reported separately (y1 verifier: it is a
   low-regret outlier).
B  "parametric" (secondary): p_c = 50-episode rate is the truth (tier(p_c)); rollouts are
   Bernoulli(p_c), K = 1..50.  Removes the noise of a 25-episode truth label.
R  "real" (validation): the 33 eval checkpoints x 10 recorded rollouts (seeds 200000-9)
   with the REAL VLM verdicts; truth tier from the 50 training-time episodes (disjoint
   seeds).  The same tier rules are run on (i) the real verdicts and (ii) simulated
   verdicts applied to the same real truth outcomes -> does the simulated judge reproduce
   the real tier accuracy?

Judge error model (fit on the 330-episode eval JSONLs, seeds 200000+, disjoint from every
seed used in worlds A/B):
  FNR_task ~ Beta(1 + fn, 1 + n_succ - fn)
  quality-dependent FPR ("q"): logit FPR_c = a_task + b * (s_c - 0.5), binomial on the
      per-checkpoint (n_fail, fp) with s_c = the checkpoint's eval truth success rate
      (y1 bbq model; y1 found kappa -> no residual overdispersion, so binomial).  MAP with a
      weak N(0, 5^2) prior on (a, b) (guards separation in leave-one-out fits); parameter
      uncertainty = Laplace.  In the simulation s_c = the checkpoint's pool success (A) or
      p_c (B) -- a world-model choice, never seen by the decision rules.
  constant FPR ("iid", sensitivity world): FPR_task ~ Beta(1 + fp, 1 + n_fail - fp).
  Each replicate draws one judge-parameter vector (world); the decision rules use the
  point fit (decision-maker's model), i.e. realistic model mismatch.

Decision rules
--------------
  naive     : tier of the judged success rate k/K (VLM verdict taken at face value).
  bayes_*   : posterior over p on a 400-point grid, likelihood with judged-success prob
              q(p) = p (1 - FNR) + (1 - p) FPR(p)  (quality-aware judge correction);
              tier = argmax posterior tier probability.  Priors: uniform, or a
              leave-one-RUN-out prior from an outcome-free metric (epoch, PLV metrics):
              Gaussian kernel (bw 0.1) over within-run metric percentile, mixing the
              Beta(1+y, 51-y) posteriors of the OTHER two runs' checkpoints, 80/20 mixed
              with uniform.  Fixed a priori, never tuned on outcomes.
  sequential: stop at the first K >= 1 with max posterior tier probability >= 0.9 (also
              0.95), else at K_max; report mean rollouts and accuracy.

Uncertainty: each replicate = random seed split (A) + run-stratified checkpoint bootstrap
+ one judge-parameter draw + 8 random episode orders per checkpoint.  Point estimates
average the un-bootstrapped accuracy over replicates; 95 % CIs are percentiles of the
bootstrapped accuracy (they include residual Monte-Carlo noise -> conservative).

Run from the repo root:
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z1_tier_triage.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, gammaln  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x1_adaptive_budget as x1  # noqa: E402  (imported, never modified; log parser only)

NAME = "z1_tier_triage"
TIERS = ("BAD", "OK", "GOOD")
JUDGE_FILES = {"nemotron": "eval_nebius_v2.jsonl", "cosmos": "eval_local_v6.jsonl",
               "nemotron_pad201": "eval_nebius_v2_pad201.jsonl"}
SIM_JUDGES = ("perfect", "nemotron", "cosmos")
PRIOR_METRICS = {  # name -> (column, lower_is_better)
    "epoch": ("epoch", False),
    "replan_overlap_h16": ("replan_overlap_h16", True),
    "val_action_l1": ("val_action_l1", True),
    "plv1_smoothness": ("plv1_smoothness", True),
    "open_loop_smoothness": ("open_loop_smoothness", True),
    "val_loss_logged": ("val_loss_logged", True),
}
G = 400
PGRID = (np.arange(G) + 0.5) / G  # never exactly 0.5 / 0.9
TMASK = np.stack([PGRID < 0.5, (PGRID >= 0.5) & (PGRID <= 0.9), PGRID > 0.9], 1).astype(float)  # [G,3]
STOP_THRS = (0.9, 0.95)
CONF_K = (1, 3, 5, 10, 15, 20, 25, 50)
PRIOR_SD = 5.0


def tier_int(k, K):
    """integer-exact tier of the rate k/K: 0 BAD (<0.5), 1 OK, 2 GOOD (>0.9)."""
    k = np.asarray(k)
    K = np.asarray(K)
    return np.where(2 * k < K, 0, np.where(10 * k > 9 * K, 2, 1))


# --------------------------------------------------------------------------- data
def load_pool():
    import pandas as pd
    runs = x1.load_runs(ROOT / "data/outputs/hackathon_pool")
    df = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv").set_index("checkpoint_id")
    ids, run_idx, task, epoch, Y, met = [], [], [], [], [], {m: [] for m in PRIOR_METRICS}
    for ri, r in enumerate(runs):
        sub = df.loc[r["ids"]]
        assert np.allclose(sub["success"].astype(float).values, r["Y"].mean(1)), "alignment"  # sanity only
        ids += r["ids"]
        run_idx += [ri] * len(r["ids"])
        task += [r["task"]] * len(r["ids"])
        epoch += list(r["epochs"])
        Y.append(r["Y"])
        for m, (col, low) in PRIOR_METRICS.items():
            v = sub[col].astype(float).values
            met[m].append(-v if low else v)  # oriented: higher = expected better
    Y = np.concatenate(Y).astype(np.int8)
    run_idx = np.array(run_idx)
    # within-run percentile of each oriented metric (outcome-free)
    pct = {}
    for m in PRIOR_METRICS:
        v = np.concatenate(met[m])
        u = np.zeros(len(v))
        for ri in np.unique(run_idx):
            s = run_idx == ri
            rk = np.argsort(np.argsort(v[s], kind="stable"), kind="stable")
            u[s] = rk / max(s.sum() - 1, 1)
        pct[m] = u
    return {"ids": ids, "run_idx": run_idx, "run_names": [r["run"] for r in runs], "task": np.array(task),
            "epoch": np.array(epoch), "Y": Y, "pct": pct}


def load_eval(stem):
    d = {}
    for line in (ROOT / "data/outputs/hackathon_cosmos" / stem).read_text().splitlines():
        r = json.loads(line)
        if not (r.get("scored") and r.get("parse_status") == "ok"):
            continue
        if not isinstance(r.get("truth_success"), bool) or not isinstance(r.get("cosmos_task_success"), bool):
            continue
        c = d.setdefault(r["checkpoint_id"], {"task": r["task"], "eps": []})
        c["eps"].append((int(r["env_seed"]), int(r["truth_success"]), int(r["cosmos_task_success"])))
    for c in d.values():
        c["eps"].sort()
        t = np.array([e[1] for e in c["eps"]])
        j = np.array([e[2] for e in c["eps"]])
        c.update(n=len(t), n_succ=int(t.sum()), n_fail=int((1 - t).sum()), fn=int(((t == 1) & (j == 0)).sum()),
                 fp=int(((t == 0) & (j == 1)).sum()), s_eval=float(t.mean()), J=int(j.sum()), truth=t, judged=j)
    return d


# --------------------------------------------------------------------------- judge model
def fit_judge(ev, exclude=()):
    tasks = ["can", "square"]
    rows = [(tasks.index(v["task"]), v["n_fail"], v["fp"], v["s_eval"]) for k, v in ev.items()
            if k not in exclude and v["n_fail"] > 0]
    T = np.array([r[0] for r in rows])
    N = np.array([r[1] for r in rows], float)
    X = np.array([r[2] for r in rows], float)
    S = np.array([r[3] for r in rows]) - 0.5
    Xd = np.zeros((len(rows), 3))
    Xd[np.arange(len(rows)), T] = 1
    Xd[:, 2] = S

    def nll(th):
        eta = Xd @ th
        return -(X * eta - N * np.logaddexp(0, eta)).sum() + 0.5 * (th ** 2).sum() / PRIOR_SD ** 2

    def grad(th):
        mu = expit(Xd @ th)
        return -(Xd.T @ (X - N * mu)) + th / PRIOR_SD ** 2

    r = minimize(nll, np.array([-1.5, -1.5, 2.0]), jac=grad, method="BFGS")
    th = r.x
    mu = expit(Xd @ th)
    H = (Xd * (N * mu * (1 - mu))[:, None]).T @ Xd + np.eye(3) / PRIOR_SD ** 2
    cov = np.linalg.inv(H)
    out = {"theta": th, "cov": cov, "tasks": tasks}
    for t in tasks:
        sel = [v for k, v in ev.items() if k not in exclude and v["task"] == t]
        fn = sum(v["fn"] for v in sel)
        ns = sum(v["n_succ"] for v in sel)
        fp = sum(v["fp"] for v in sel)
        nf = sum(v["n_fail"] for v in sel)
        out[t] = {"fn": fn, "n_succ": ns, "fp": fp, "n_fail": nf, "fnr": (1 + fn) / (2 + ns), "fpr_iid": (1 + fp) / (2 + nf)}
    return out


def fpr_q(fit, task, s, th=None):
    th = fit["theta"] if th is None else th
    ti = fit["tasks"].index(task)
    return expit(th[..., ti] + th[..., 2] * (np.asarray(s) - 0.5))


def draw_judge(fit, rng):
    """one world parameter draw: dict task -> (fnr, theta) ; theta shared."""
    th = rng.multivariate_normal(fit["theta"], fit["cov"])
    d = {"theta": th}
    for t in fit["tasks"]:
        c = fit[t]
        d[t] = {"fnr": rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"]),
                "fpr_iid": rng.beta(1 + c["fp"], 1 + c["n_fail"] - c["fp"])}
    return d


def qcurve(fit, task, mode):
    """judged-success probability as a function of latent p on the grid."""
    if mode == "perfect":
        return PGRID.copy()
    fnr = fit[task]["fnr"]
    fpr = fpr_q(fit, task, PGRID) if mode == "q" else fit[task]["fpr_iid"]
    return PGRID * (1 - fnr) + (1 - PGRID) * fpr


# --------------------------------------------------------------------------- posteriors
def lik_table(q, Kmax):
    """L[K, k, g] = q^k (1-q)^(K-k), normalised per (K,k) for stability (binomial coef cancels)."""
    K = np.arange(Kmax + 1)[:, None, None]
    k = np.arange(Kmax + 1)[None, :, None]
    lq, l1q = np.log(np.clip(q, 1e-12, 1)), np.log(np.clip(1 - q, 1e-12, 1))
    ll = k * lq[None, None, :] + np.clip(K - k, 0, None) * l1q[None, None, :]
    ll = np.where(k <= K, ll, -np.inf)
    ll = ll - np.max(np.where(np.isfinite(ll), ll, -1e300), axis=2, keepdims=True)
    return np.exp(ll)


def lik_table_marg(fit, task, Kmax, D=120, seed=0):
    """judge-parameter-marginalised likelihood: mean over D Laplace/Beta draws of q_theta(p)^k (1-q_theta(p))^(K-k)."""
    rng = np.random.default_rng([seed, 5, Kmax, 0 if task == "can" else 1])
    K = np.arange(Kmax + 1)[:, None, None]
    k = np.arange(Kmax + 1)[None, :, None]
    valid = k <= K
    acc = None
    c = fit[task]
    for _ in range(D):
        th = rng.multivariate_normal(fit["theta"], fit["cov"])
        fnr = rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"])
        q = PGRID * (1 - fnr) + (1 - PGRID) * fpr_q(fit, task, PGRID, th)
        ll = k * np.log(np.clip(q, 1e-12, 1))[None, None] + np.clip(K - k, 0, None) * np.log(np.clip(1 - q, 1e-12, 1))[None, None]
        acc = ll if acc is None else np.logaddexp(acc, ll)
    acc = np.where(valid, acc, -np.inf)
    acc = acc - np.max(np.where(np.isfinite(acc), acc, -1e300), axis=2, keepdims=True)
    return np.exp(acc)


def tier_post(L, prior):
    """L [K+1,K+1,G], prior [C,G] -> tier probs [C, K+1, K+1, 3]."""
    num = np.einsum("Kkg,cg,gt->cKkt", L, prior, TMASK, optimize=True)
    den = num.sum(-1, keepdims=True)
    return num / np.where(den > 0, den, 1)


TIERFLAT = (TMASK / TMASK.sum(0, keepdims=True)).sum(1) / 3.0  # 1/3 prior mass per tier, flat within tier
KDE_METRICS = ("epoch", "replan_overlap_h16", "val_action_l1")
PROG_METRICS = ("epoch", "replan_overlap_h16", "val_action_l1", "plv1_smoothness", "val_loss_logged")


def _beta_dens(Y):
    y = Y.sum(1).astype(float)
    n = Y.shape[1]
    lb = gammaln(n + 2) - gammaln(y + 1) - gammaln(n - y + 1)
    dens = np.exp(lb[:, None] + y[:, None] * np.log(PGRID)[None] + (n - y)[:, None] * np.log(1 - PGRID)[None])
    return dens / dens.sum(1, keepdims=True)


def build_priors(pool):
    """All priors [C, G]; every outcome-based prior is leave-one-RUN-out (other two runs, all 50 ep).
    flat      : uniform on p (puts only 10 % mass on GOOD)
    tierflat  : 1/3 mass per tier, flat within tier (neutral reference)
    pop       : 0.5 * mixture of other runs' Beta(1+y, 51-y) + 0.5 * tierflat (population, no metric)
    kde_<m>   : 0.5 * kernel(bw 0.1 on within-run metric percentile) mixture of other runs' Betas + 0.5 * tierflat
    prog_<m>  : only P(BAD | metric percentile) is learned (kernel-weighted share of other-run checkpoints with
                50-ep rate < 0.5, +0.5/+1 pseudo-counts); the remaining mass is split 50/50 between OK and GOOD,
                flat within tier.  Transfers 'training progress' but not the other task's plateau level."""
    C = len(pool["ids"])
    dens = _beta_dens(pool["Y"])
    bad = (pool["Y"].mean(1) < 0.5).astype(float)
    flatT = (TMASK / TMASK.sum(0, keepdims=True))  # [G,3] flat density within each tier
    pri = {"flat": np.full((C, G), 1.0 / G), "tierflat": np.tile(TIERFLAT, (C, 1)), "pop": np.zeros((C, G))}
    for m in KDE_METRICS:
        pri[f"kde_{m}"] = np.zeros((C, G))
    for m in PROG_METRICS:
        pri[f"prog_{m}"] = np.zeros((C, G))
    for c in range(C):
        other = pool["run_idx"] != pool["run_idx"][c]
        pri["pop"][c] = 0.5 * dens[other].mean(0) + 0.5 * TIERFLAT
        for m in sorted(set(KDE_METRICS) | set(PROG_METRICS)):
            u = pool["pct"][m]
            w = np.exp(-0.5 * ((u[c] - u[other]) / 0.1) ** 2)
            if m in KDE_METRICS:
                pri[f"kde_{m}"][c] = 0.5 * (w[:, None] * dens[other]).sum(0) / w.sum() + 0.5 * TIERFLAT
            if m in PROG_METRICS:
                pb = ((w * bad[other]).sum() + 0.5) / (w.sum() + 1.0)
                pri[f"prog_{m}"][c] = flatT @ np.array([pb, 0.5 * (1 - pb), 0.5 * (1 - pb)])
    return {k: v / v.sum(1, keepdims=True) for k, v in pri.items()}


# --------------------------------------------------------------------------- core simulation
def decide(post, k_cum, Kmax):
    """post [C, Kmax+1, Kmax+1, 3]; k_cum [O, C, Kmax+1] cumulative judged successes (k_cum[...,0]=0).
    returns fixed-K decisions [O,C,Kmax+1] and, per stop threshold, (K*, decision)."""
    O, C, _ = k_cum.shape
    cidx = np.arange(C)[None, :, None]
    Kidx = np.arange(Kmax + 1)[None, None, :]
    P = post[cidx, Kidx, k_cum]  # [O,C,K+1,3]
    dec = P.argmax(-1)
    seq = {}
    mx = P.max(-1)
    for thr in STOP_THRS:
        hit = mx[:, :, 1:] >= thr
        first = np.where(hit.any(-1), hit.argmax(-1) + 1, Kmax)
        d = np.take_along_axis(dec, first[..., None], axis=2)[..., 0]
        seq[thr] = (first, d)
    return dec, seq, P


def run_world(world, pool, fits, priors, args, rng_master):
    """world 'A' (split) or 'B' (parametric)."""
    Y = pool["Y"]
    C = Y.shape[0]
    tasks = pool["task"]
    run_idx = pool["run_idx"]
    Kmax = 25 if world == "A" else 50
    O = args.orders
    # posterior tables that do not depend on the replicate
    rules = {}  # rule -> (judge, lik_mode, prior_name)
    for j in SIM_JUDGES:
        rules[f"{j}|naive"] = (j, "naive", None)
        if j != "perfect":
            rules[f"{j}|naive_bayes_tierflat"] = (j, "perfect", "tierflat")
            rules[f"{j}|bayes_iid_tierflat"] = (j, "iid", "tierflat")
        for pn in priors:
            rules[f"{j}|bayes_{pn}"] = (j, "perfect" if j == "perfect" else "q", pn)
        if j != "perfect":
            for pn in ("flat", "tierflat", "prog_epoch"):
                rules[f"{j}|bayes_qmarg_{pn}"] = (j, "qmarg", pn)
    post_cache = {}
    lmarg_cache = {}
    for name, (j, lm, pn) in rules.items():
        if lm == "naive":
            continue
        key = (j, lm, pn)
        if key in post_cache:
            continue
        P = np.zeros((C, Kmax + 1, Kmax + 1, 3))
        for t in ("can", "square"):
            s = tasks == t
            fit = fits[j] if j != "perfect" else None
            if lm == "qmarg":
                if (j, t) not in lmarg_cache:
                    lmarg_cache[(j, t)] = lik_table_marg(fit, t, Kmax, seed=args.seed)
                Lt = lmarg_cache[(j, t)]
            else:
                Lt = lik_table(qcurve(fit, t, "perfect" if j == "perfect" or lm == "perfect" else lm), Kmax)
            P[s] = tier_post(Lt, priors[pn][s])
        post_cache[key] = P
    # prior-only (K = 0) tier calls
    prior_only = {pn: (priors[pn] @ TMASK).argmax(1) for pn in priors}

    B = args.reps
    acc = {r: np.zeros((B, Kmax + 1)) for r in rules}
    acc_bs = {r: np.zeros((B, Kmax + 1)) for r in rules}
    recall = {r: np.zeros((B, 3, Kmax + 1)) for r in rules}
    conf = {r: np.zeros((len([k for k in CONF_K if k <= Kmax]), 3, 3)) for r in rules}
    seq_stats = {r: {thr: {"acc": np.zeros(B), "acc_bs": np.zeros(B), "meanK": np.zeros(B), "meanK_bs": np.zeros(B),
                           "meanK_by_tier": np.zeros((B, 3)), "acc_by_tier": np.zeros((B, 3))} for thr in STOP_THRS}
                 for r in rules if rules[r][1] != "naive"}
    per_ckpt_acc = {r: np.zeros((C, Kmax + 1)) for r in rules}
    p50_all = Y.mean(1)
    clear = ~(((p50_all >= 0.45) & (p50_all <= 0.55)) | ((p50_all >= 0.85) & (p50_all <= 0.95)))  # reporting stratum only
    acc_clear = {r: np.zeros((B, Kmax + 1)) for r in rules}
    acc_by_task = {r: {t: np.zeros((B, Kmax + 1)) for t in ("can", "square")} for r in rules}
    per_ckpt_seqK = {r: np.zeros(C) for r in seq_stats}
    prior_acc = {pn: np.zeros(B) for pn in priors}
    truth_counts = np.zeros(3)
    for b in range(B):
        rng = np.random.default_rng([args.seed, 1 if world == "A" else 2, b])
        if world == "A":
            perm = np.arange(50) if b == 0 and args.fixed_first else rng.permutation(50)
            pool_idx, held_idx = perm[:25], perm[25:]
            truth = tier_int(Y[:, held_idx].sum(1), 25)
            s_q = Y[:, pool_idx].mean(1)
            # K rollouts without replacement from the pool half, random order per (o, c)
            order = np.argsort(rng.random((O, C, 25)), axis=2)
            outc = np.take_along_axis(np.broadcast_to(Y[:, pool_idx], (O, C, 25)), order, axis=2).astype(bool)
        else:
            p = Y.mean(1)
            truth = tier_int(Y.sum(1), 50)
            s_q = p
            outc = rng.random((O, C, Kmax)) < p[None, :, None]
        truth_counts += np.bincount(truth, minlength=3)
        # run-stratified checkpoint bootstrap weights
        w = np.zeros(C)
        for ri in np.unique(run_idx):
            idx = np.where(run_idx == ri)[0]
            w[idx] = np.bincount(rng.choice(len(idx), len(idx)), minlength=len(idx))
        w_unit = np.ones(C)
        U = rng.random((O, C, Kmax))
        judged = {"perfect": outc}
        for j in SIM_JUDGES[1:]:
            dj = draw_judge(fits[j], rng)
            fnr = np.array([dj[t]["fnr"] for t in tasks])
            if args.world_judge == "q":
                fpr = np.array([float(fpr_q(fits[j], t, s, dj["theta"])) for t, s in zip(tasks, s_q)])
            else:
                fpr = np.array([dj[t]["fpr_iid"] for t in tasks])
            judged[j] = np.where(outc, U >= fnr[None, :, None], U < fpr[None, :, None])
        for pn in priors:
            prior_acc[pn][b] = np.average(prior_only[pn] == truth, weights=w)
        for name, (j, lm, pn) in rules.items():
            kc = np.concatenate([np.zeros((O, C, 1), int), np.cumsum(judged[j], axis=2)], axis=2)
            if lm == "naive":
                Ks = np.arange(Kmax + 1)[None, None, :]
                dec = tier_int(kc, np.maximum(Ks, 1))
                dec[..., 0] = 1  # K = 0: no information -> call OK (unused)
                seq = None
            else:
                dec, seq, _ = decide(post_cache[(j, lm, pn)], kc, Kmax)
            corr = dec == truth[None, :, None]  # [O,C,K+1]
            cm = corr.mean(0)  # [C,K+1]
            per_ckpt_acc[name] += cm
            acc[name][b] = cm.mean(0)
            acc_bs[name][b] = (cm * w[:, None]).sum(0) / w.sum()
            acc_clear[name][b] = (cm * (w * clear)[:, None]).sum(0) / (w * clear).sum()
            for t in ("can", "square"):
                acc_by_task[name][t][b] = cm[tasks == t].mean(0)
            for t in range(3):
                s = truth == t
                recall[name][b, t] = cm[s].mean(0) if s.any() else np.nan
            for ki, K in enumerate([k for k in CONF_K if k <= Kmax]):
                for t in range(3):
                    s = truth == t
                    if s.any():
                        conf[name][ki, t] += np.bincount(dec[:, s, K].ravel(), minlength=3)
            if seq is not None:
                for thr, (Kst, dst) in seq.items():
                    st = seq_stats[name][thr]
                    cs = (dst == truth[None, :]).mean(0)
                    ks = Kst.mean(0)
                    st["acc"][b] = cs.mean()
                    st["acc_bs"][b] = (cs * w).sum() / w.sum()
                    st["meanK"][b] = ks.mean()
                    st["meanK_bs"][b] = (ks * w).sum() / w.sum()
                    for t in range(3):
                        s = truth == t
                        st["meanK_by_tier"][b, t] = ks[s].mean() if s.any() else np.nan
                        st["acc_by_tier"][b, t] = cs[s].mean() if s.any() else np.nan
                    if thr == 0.9:
                        per_ckpt_seqK[name] += ks
    # ---- summarise
    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]

    def minK(curve, target=0.9):
        hit = np.where(curve[1:] >= target)[0]
        return int(hit[0] + 1) if len(hit) else None

    res = {"Kmax": Kmax, "reps": B, "orders": O, "world_judge": args.world_judge,
           "clear_cut_stratum": {"definition": "50-ep rate outside [0.45,0.55] and [0.85,0.95] (reporting only, never used by rules)",
                                 "n_ckpt": int(clear.sum()), "by_task": {t: int((clear & (tasks == t)).sum()) for t in ("can", "square")}},
           "truth_tier_share": (truth_counts / truth_counts.sum()).round(4).tolist(), "rules": {}}
    if world == "A" and args.fixed_first:
        res["note_replicate0"] = "replicate 0 = fixed split (pool 100000-24 / held-out 100025-49), reported in 'fixed_split'"
    for name in rules:
        a = acc[name][1:] if (world == "A" and args.fixed_first) else acc[name]
        abs_ = acc_bs[name][1:] if (world == "A" and args.fixed_first) else acc_bs[name]
        point = a.mean(0)
        mk_rep = [minK(x) for x in abs_]
        mk_arr = np.array([m if m is not None else Kmax + 1 for m in mk_rep], float)
        rr = {
            "acc_point": point.round(4).tolist(),
            "acc_ci95": [ci(abs_[:, K]) for K in range(Kmax + 1)],
            "recall_point": {TIERS[t]: np.nanmean(recall[name][:, t], 0).round(4).tolist() for t in range(3)},
            "minK_90_point": minK(point),
            "minK_90_bootstrap": {"median": float(np.median(mk_arr)), "p2.5": float(np.percentile(mk_arr, 2.5)),
                                  "p97.5": float(np.percentile(mk_arr, 97.5)),
                                  "frac_reached": float(np.mean([m is not None for m in mk_rep])),
                                  "note": f"{Kmax + 1} = not reached within K_max"},
            "minK_80_point": minK(point, 0.8), "minK_85_point": minK(point, 0.85), "minK_95_point": minK(point, 0.95),
            "clear_cut_acc_ci95_at": {str(K): ci(acc_clear[name][:, K]) for K in (1, 3, 5, 10, 15, 20, 25, 50) if K <= Kmax},
            "clear_cut_minK_90_bootstrap_median": float(np.median([minK(x) or Kmax + 1 for x in acc_clear[name]])),
            "acc_by_task_point": {t: acc_by_task[name][t].mean(0).round(4).tolist() for t in ("can", "square")},
            "confusion_rows_truth_cols_call": {str(K): (conf[name][ki] / conf[name][ki].sum(1, keepdims=True).clip(1)).round(4).tolist()
                                              for ki, K in enumerate([k for k in CONF_K if k <= Kmax])},
        }
        if world == "A" and args.fixed_first:
            rr["fixed_split_acc"] = acc[name][0].round(4).tolist()
            rr["fixed_split_minK_90"] = minK(acc[name][0])
        if name in seq_stats:
            rr["sequential"] = {}
            for thr, st in seq_stats[name].items():
                sl = slice(1, None) if (world == "A" and args.fixed_first) else slice(None)
                rr["sequential"][str(thr)] = {
                    "acc_point": float(st["acc"][sl].mean()), "acc_ci95": ci(st["acc_bs"][sl]),
                    "mean_rollouts_point": float(st["meanK"][sl].mean()), "mean_rollouts_ci95": ci(st["meanK_bs"][sl]),
                    "mean_rollouts_by_truth_tier": {TIERS[t]: float(np.nanmean(st["meanK_by_tier"][sl, t])) for t in range(3)},
                    "acc_by_truth_tier": {TIERS[t]: float(np.nanmean(st["acc_by_tier"][sl, t])) for t in range(3)},
                    "fixed_K_acc_at_same_mean_budget": float(np.interp(st["meanK"][sl].mean(), np.arange(Kmax + 1), point)),
                }
        res["rules"][name] = rr
    res["prior_only_K0_acc"] = {pn: {"point": float(v.mean()), "ci95": ci(v)} for pn, v in prior_acc.items()}
    # failure cases: checkpoints with the lowest accuracy at K = 10 (per judge, quality-aware uniform rule)
    fails = {}
    p50 = pool["Y"].mean(1)
    for j in SIM_JUDGES:
        name = f"{j}|bayes_tierflat"
        pc = per_ckpt_acc[name] / B
        order = np.argsort(pc[:, 10])
        fails[j] = [{"checkpoint_id": pool["ids"][c], "p50": float(p50[c]), "tier50": TIERS[int(tier_int(pool['Y'][c].sum(), 50))],
                     "acc_K10": float(pc[c, 10]), "acc_Kmax": float(pc[c, Kmax]),
                     "seq0.9_mean_rollouts": float(per_ckpt_seqK[name][c] / B)} for c in order[:10]]
        nv = f"{j}|naive"
        pcn = per_ckpt_acc[nv] / B
        res.setdefault("per_checkpoint_acc_K10_vs_p50", {})[j] = {
            "bayes_tierflat": [[float(p50[c]), float(pc[c, 10])] for c in range(C)],
            "naive": [[float(p50[c]), float(pcn[c, 10])] for c in range(C)]}
        # accuracy by p50 band (failure anatomy)
        bands = [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
        res.setdefault("acc_K10_by_p50_band", {})[j] = {
            f"[{lo},{hi})": {"n": int(((p50 >= lo) & (p50 < hi)).sum()),
                             "bayes_tierflat": float(pc[(p50 >= lo) & (p50 < hi), 10].mean()) if ((p50 >= lo) & (p50 < hi)).any() else None,
                             "naive": float(pcn[(p50 >= lo) & (p50 < hi), 10].mean()) if ((p50 >= lo) & (p50 < hi)).any() else None}
            for lo, hi in bands}
    res["worst_checkpoints_K10"] = fails
    return res


# --------------------------------------------------------------------------- validation (33 eval checkpoints)
def judged_pmf(n_succ, n_fail, fnr, fpr):
    """pmf of judged successes = Binom(n_succ, 1-fnr) + Binom(n_fail, fpr) (vectorised over draws)."""
    from scipy.stats import binom
    a = binom.pmf(np.arange(n_succ + 1)[None, :], n_succ, (1 - fnr)[:, None])
    bq = binom.pmf(np.arange(n_fail + 1)[None, :], n_fail, fpr[:, None])
    out = np.zeros((len(fnr), n_succ + n_fail + 1))
    for i in range(n_succ + 1):
        out[:, i:i + n_fail + 1] += a[:, i:i + 1] * bq
    return out.mean(0)


def validate(evs, pool, args):
    rng = np.random.default_rng([args.seed, 99])
    p50 = dict(zip(pool["ids"], pool["Y"].mean(1)))
    out = {}
    for jname, ev in evs.items():
        rows = []
        for cid, v in sorted(ev.items()):
            fit = fit_judge(ev, exclude=(cid,))
            D = 4000
            th = rng.multivariate_normal(fit["theta"], fit["cov"], size=D)
            c = fit[v["task"]]
            fnr = rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"], size=D)
            fpr_q_ = fpr_q(fit, v["task"], v["s_eval"], th)
            fpr_i = rng.beta(1 + c["fp"], 1 + c["n_fail"] - c["fp"], size=D)
            res = {"checkpoint_id": cid, "task": v["task"], "n": v["n"], "truth_succ": v["n_succ"], "real_judged": v["J"],
                   "p50_train": float(p50[cid])}
            for mode, fpr in (("q", fpr_q_), ("iid", fpr_i)):
                pmf = judged_pmf(v["n_succ"], v["n_fail"], fnr, fpr)
                cdf = np.cumsum(pmf)
                J = v["J"]
                lo = cdf[J - 1] if J > 0 else 0.0
                res[f"{mode}_pred_mean"] = float((np.arange(len(pmf)) * pmf).sum())
                res[f"{mode}_logp"] = float(np.log(max(pmf[J], 1e-12)))
                res[f"{mode}_pit_mid"] = float(lo + 0.5 * pmf[J])
                # central 90 % predictive interval
                lo_i = int(np.searchsorted(cdf, 0.05))
                hi_i = int(np.searchsorted(cdf, 0.95))
                res[f"{mode}_pi90"] = [lo_i, hi_i]
                res[f"{mode}_in_pi90"] = bool(lo_i <= J <= hi_i)
                res[f"{mode}_pred_fp"] = float(v["n_fail"] * fpr.mean())
            # end-to-end: judged rate predicted from training-time quality only (no eval truth)
            q_e2e = p50[cid] * (1 - fit[v["task"]]["fnr"]) + (1 - p50[cid]) * float(fpr_q(fit, v["task"], p50[cid]))
            res["e2e_pred_rate_from_p50"] = float(q_e2e)
            res["real_fp"] = v["fp"]
            res["real_fn"] = v["fn"]
            res["n_fail"] = v["n_fail"]
            rows.append(res)
        summ = {}
        for mode in ("q", "iid"):
            err = np.array([r[f"{mode}_pred_mean"] - r["real_judged"] for r in rows]) / 10
            summ[mode] = {"mae_judged_rate": float(np.abs(err).mean()), "bias_judged_rate": float(err.mean()),
                          "coverage_pi90": float(np.mean([r[f"{mode}_in_pi90"] for r in rows])),
                          "loo_logscore_sum": float(sum(r[f"{mode}_logp"] for r in rows)),
                          "fp_pred_vs_real_by_quality": {}}
            for lo, hi in ((0, 0.5), (0.5, 0.85), (0.85, 1.01)):
                sel = [r for r in rows if lo <= r["truth_succ"] / r["n"] < hi]
                summ[mode]["fp_pred_vs_real_by_quality"][f"s_eval[{lo},{hi})"] = {
                    "n_ckpt": len(sel), "n_fail": int(sum(r["n_fail"] for r in sel)),
                    "pred_fp": float(sum(r[f"{mode}_pred_fp"] for r in sel)), "real_fp": int(sum(r["real_fp"] for r in sel))}
        e2e = np.array([r["e2e_pred_rate_from_p50"] - r["real_judged"] / r["n"] for r in rows])
        tr = np.array([r["p50_train"] - r["truth_succ"] / r["n"] for r in rows])
        summ["end_to_end"] = {"mae_judged_rate_pred_from_p50": float(np.abs(e2e).mean()),
                              "bias": float(e2e.mean()),
                              "reference_mae_truth_rate_eval_vs_p50": float(np.abs(tr).mean()),
                              "note": "end-to-end error is dominated by eval(10 ep, seeds 200000+) vs training-time(50 ep) truth differences"}
        out[jname] = {"rows": rows, "summary": summ, "full_fit": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                                   for k, v in fit_judge(ev).items()}}
        f = fit_judge(ev)
        out[jname]["implied_fpr"] = {t: {str(s): float(fpr_q(f, t, s)) for s in (0.0, 0.5, 0.8, 0.9, 1.0)} for t in f["tasks"]}
    return out


def real_tier_check(evs, pool, args):
    """World R: tier calls from the REAL VLM verdicts on the 33 eval checkpoints (K = 1..10)
    vs the same rules on simulated verdicts applied to the same real truth outcomes."""
    rng = np.random.default_rng([args.seed, 77])
    idx = {c: i for i, c in enumerate(pool["ids"])}
    out = {}
    O = 400
    for jname in ("nemotron", "cosmos"):
        ev = evs[jname]
        cids = sorted(ev)
        truthT = np.array([int(tier_int(pool["Y"][idx[c]].sum(), 50)) for c in cids])
        runs = np.array([pool["run_idx"][idx[c]] for c in cids])
        n = 10
        acc = {k: np.zeros((len(cids), n + 1)) for k in ("perfect|naive", "perfect|bayes_tierflat", "real|naive", "real|bayes_tierflat", "sim|naive", "sim|bayes_tierflat")}
        for ci_, cid in enumerate(cids):
            v = ev[cid]
            assert v["n"] == n
            fit = fit_judge(ev, exclude=(cid,))  # leave-one-checkpoint-out decision AND world model
            q = qcurve(fit, v["task"], "q")
            post = tier_post(lik_table(q, n), TIERFLAT[None])[0]  # [K+1,k+1,3]  quality-aware judge model
            post_perf = tier_post(lik_table(PGRID.copy(), n), TIERFLAT[None])[0]
            order = np.argsort(rng.random((O, n)), axis=1)
            t_ord = v["truth"][order]
            j_ord = v["judged"][order]
            D = O
            th = rng.multivariate_normal(fit["theta"], fit["cov"], size=D)
            c = fit[v["task"]]
            fnr = rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"], size=D)
            fpr = fpr_q(fit, v["task"], v["s_eval"], th)
            U = rng.random((O, n))
            sim_ord = np.where(t_ord == 1, U >= fnr[:, None], U < fpr[:, None]).astype(int)
            Ks = np.arange(1, n + 1)
            for key, seqs in (("perfect", t_ord), ("real", j_ord), ("sim", sim_ord)):
                kc = np.cumsum(seqs, axis=1)
                dn = tier_int(kc, Ks[None, :])
                acc[f"{key}|naive"][ci_, 1:] = (dn == truthT[ci_]).mean(0)
                db = (post_perf if key == "perfect" else post)[Ks[None, :], kc].argmax(-1)
                acc[f"{key}|bayes_tierflat"][ci_, 1:] = (db == truthT[ci_]).mean(0)
        # bootstrap over checkpoints (run-stratified)
        B = 2000
        res = {"n_ckpt": len(cids), "truth_tier_share_p50": np.bincount(truthT, minlength=3).tolist(), "rules": {}}
        bs_w = np.zeros((B, len(cids)))
        for b in range(B):
            for ri in np.unique(runs):
                ii = np.where(runs == ri)[0]
                bs_w[b, ii] = np.bincount(rng.choice(len(ii), len(ii)), minlength=len(ii))
        for k, a in acc.items():
            bs = bs_w @ a / bs_w.sum(1, keepdims=True)
            res["rules"][k] = {"acc_point": a.mean(0).round(4).tolist(),
                               "acc_ci95": [[float(np.percentile(bs[:, K], 2.5)), float(np.percentile(bs[:, K], 97.5))] for K in range(n + 1)]}
        for rule in ("naive", "bayes_tierflat"):
            d = acc[f"real|{rule}"] - acc[f"sim|{rule}"]
            bs = bs_w @ d / bs_w.sum(1, keepdims=True)
            res[f"real_minus_sim_{rule}"] = {"point": d.mean(0).round(4).tolist(),
                                             "ci95": [[float(np.percentile(bs[:, K], 2.5)), float(np.percentile(bs[:, K], 97.5))] for K in range(n + 1)],
                                             "mean_abs_per_ckpt_K10": float(np.abs(d[:, n]).mean())}
        res["per_checkpoint_K10"] = [{"checkpoint_id": c, "truth_tier_p50": TIERS[truthT[i]],
                                      **{k: float(acc[k][i, n]) for k in acc}} for i, c in enumerate(cids)]
        out[jname] = res
    return out


# --------------------------------------------------------------------------- figure
def make_fig(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"perfect": "#2a78d6", "nemotron": "#eb6834", "cosmos": "#1baf7a"}
    lab = {"perfect": "perfect detector", "nemotron": "Nemotron (Nebius)", "cosmos": "Cosmos3-Edge (local)"}
    ink2 = "#52514e"
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.6))

    def curves(ax, W, title, ylabel):
        R = W["rules"]
        K = np.arange(W["Kmax"] + 1)
        for j in SIM_JUDGES:
            r = R[f"{j}|naive"]
            y = np.array(r["acc_point"])
            lo = np.array([c[0] for c in r["acc_ci95"]])
            hi = np.array([c[1] for c in r["acc_ci95"]])
            ax.plot(K[1:], y[1:], "-", color=col[j], lw=2, label=f"{lab[j]}: plug-in k/K")
            ax.fill_between(K[1:], lo[1:], hi[1:], color=col[j], alpha=0.12, lw=0)
            r2 = R[f"{j}|bayes_prog_epoch"] if j == "perfect" else R[f"{j}|bayes_qmarg_prog_epoch"]
            ax.plot(K[1:], np.array(r2["acc_point"])[1:], "--", color=col[j], lw=1.6,
                    label=f"{lab[j]}: Bayes, epoch-progress prior" + ("" if j == "perfect" else ", judge-corrected"))
        ax.axhline(0.9, color=ink2, lw=1)
        ax.text(K[-1], 0.905, "90 %", ha="right", va="bottom", fontsize=8, color=ink2)
        ax.set_xlabel("rollouts per checkpoint K")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.25)

    curves(axs[0, 0], rep["world_A_split"], "A. World A: truth = held-out 25 ep, K drawn from other 25\n(mean of random splits; band = 95% CI)", "tier accuracy (93 checkpoints)")
    axs[0, 0].legend(fontsize=7, loc="lower right")
    curves(axs[0, 1], rep["world_B_parametric"], "B. World B: truth = 50-ep rate, Bernoulli rollouts (K to 50)", "tier accuracy")
    # C: sequential trade-off (world B)
    ax = axs[0, 2]
    W = rep["world_B_parametric"]["rules"]
    K = np.arange(rep["world_B_parametric"]["Kmax"] + 1)
    for j in SIM_JUDGES:
        ax.plot(K[1:], np.array(W[f"{j}|naive"]["acc_point"])[1:], "-", color=col[j], lw=1.2, alpha=0.6)
        names = ["bayes_flat", "bayes_prog_epoch"] if j == "perfect" else ["bayes_qmarg_flat", "bayes_qmarg_prog_epoch", "naive_bayes_tierflat"]
        for nm, mk in zip(names, ("o", "s", "^")):
            sq = W[f"{j}|{nm}"]["sequential"]["0.9"]
            ax.errorbar(sq["mean_rollouts_point"], sq["acc_point"],
                        yerr=[[sq["acc_point"] - sq["acc_ci95"][0]], [sq["acc_ci95"][1] - sq["acc_point"]]],
                        fmt=mk, color=col[j], ms=8, mec="white", mew=1.5, capsize=2, lw=1)
    ax.axhline(0.9, color=ink2, lw=1)
    ax.set_xlabel("mean rollouts per checkpoint")
    ax.set_ylabel("tier accuracy")
    ax.set_title("C. World B: sequential stop at P(tier) >= 0.9 (markers)\nvs fixed K plug-in (lines). o flat prior, s epoch-progress, ^ judge-naive", loc="left", fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.25)
    # D: judge calibration
    ax = axs[1, 0]
    for j in ("nemotron", "cosmos"):
        rows = rep["judge_validation"][j]["rows"]
        x = np.array([r["q_pred_mean"] / r["n"] for r in rows])
        y = np.array([r["real_judged"] / r["n"] for r in rows])
        ax.scatter(x + (0.004 if j == "cosmos" else -0.004), y, s=40, color=col[j], edgecolor="white", lw=1.2, label=f"{lab[j]}", zorder=3)
    ax.plot([0, 1], [0, 1], color=ink2, lw=1)
    ax.set_xlabel("simulated judged-success rate (leave-one-checkpoint-out fit)")
    ax.set_ylabel("real judged-success rate (10 eval rollouts)")
    ax.set_title("D. Judge simulator vs real verdicts, 33 eval checkpoints", loc="left", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    # E: real vs sim tier accuracy
    ax = axs[1, 1]
    R = rep["world_R_real_tier_check"]
    K = np.arange(11)
    for j in ("nemotron", "cosmos"):
        for key, ls in (("real|naive", "-"), ("sim|naive", "--")):
            r = R[j]["rules"][key]
            ax.plot(K[1:], np.array(r["acc_point"])[1:], ls, color=col[j], lw=2, label=f"{lab[j]}: {key.split('|')[0]} verdicts")
    r = R["nemotron"]["rules"]["perfect|naive"]
    ax.plot(K[1:], np.array(r["acc_point"])[1:], "-", color=col["perfect"], lw=2, label="truth_success (perfect)")
    ax.axhline(0.9, color=ink2, lw=1)
    ax.set_xlabel("rollouts K (subsets of the 10 recorded eval rollouts)")
    ax.set_ylabel("tier accuracy (truth = 50 training-time ep)")
    ax.set_title("E. Real vs simulated verdicts, plug-in rule, 33 checkpoints", loc="left", fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")
    # F: confusion, world B, K = 25, plug-in
    ax = axs[1, 2]
    W = rep["world_B_parametric"]["rules"]
    mats = [np.array(W[f"{j}|naive"]["confusion_rows_truth_cols_call"]["25"]) for j in SIM_JUDGES]
    M = np.concatenate([np.pad(m, ((0, 0), (0, 1)), constant_values=np.nan) for m in mats], 1)[:, :-1]
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for jj in range(M.shape[1]):
            if np.isfinite(M[i, jj]):
                ax.text(jj, i, f"{M[i, jj]:.2f}", ha="center", va="center", fontsize=8, color="#0b0b0b" if M[i, jj] < 0.6 else "white")
    ax.set_yticks(range(3), [f"truth {t}" for t in TIERS])
    xt = []
    for j in SIM_JUDGES:
        xt += [f"{j[:4]}\n{t}" for t in TIERS] + [""]
    ax.set_xticks(range(M.shape[1]), xt[:-1], fontsize=7)
    ax.set_title("F. World B, K = 25, plug-in: row-normalised calls\n(perfect | Nemotron | Cosmos)", loc="left", fontsize=10)
    fig.suptitle("z1_tier_triage (EXPLORATORY / post-hoc): how many rollouts to call a checkpoint BAD / OK / GOOD", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--orders", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--world-judge", default="q", choices=("q", "iid"))
    ap.add_argument("--no-fixed-first", dest="fixed_first", action="store_false")
    ap.add_argument("--worlds", default="A,B")
    ap.add_argument("--out", default=str(ROOT / f"results/hackathon/explore/{NAME}.json"))
    ap.add_argument("--fig", default=str(ROOT / f"docs/hackathon/explore/{NAME}.png"))
    args = ap.parse_args()
    t0 = time.time()
    pool = load_pool()
    evs = {k: load_eval(v) for k, v in JUDGE_FILES.items()}
    fits = {j: fit_judge(evs[j]) for j in ("nemotron", "cosmos")}
    for j, f in fits.items():
        print(j, "theta", np.round(f["theta"], 3), "se", np.round(np.sqrt(np.diag(f["cov"])), 3),
              {t: (round(f[t]["fnr"], 4), round(f[t]["fpr_iid"], 4)) for t in f["tasks"]}, flush=True)
    C = len(pool["ids"])
    priors = build_priors(pool)
    rep = {"name": NAME, "exploratory": True,
           "label": "EXPLORATORY / post-hoc; simulation from recorded training-time rollouts + judge error models fit on eval JSONLs; CPU only",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "command": " ".join(sys.argv), "args": vars(args),
           "data": {"n_ckpt": C, "runs": pool["run_names"],
                    "tier50_counts": {r: np.bincount(tier_int(pool["Y"][pool["run_idx"] == i].sum(1), 50), minlength=3).tolist()
                                      for i, r in enumerate(pool["run_names"])},
                    "half_vs_half_tier_agreement_fixed_split": float(np.mean(tier_int(pool["Y"][:, :25].sum(1), 25) == tier_int(pool["Y"][:, 25:].sum(1), 25)))},
           "judge_fits": {j: {"theta_a_can_a_square_b": f["theta"].tolist(), "theta_se": np.sqrt(np.diag(f["cov"])).tolist(),
                              **{t: f[t] for t in f["tasks"]},
                              "implied_fpr": {t: {str(s): float(fpr_q(f, t, s)) for s in (0.0, 0.5, 0.8, 0.9, 1.0)} for t in f["tasks"]},
                              "judged_rate_q_at_p": {t: {str(p): float(np.interp(p, PGRID, qcurve(f, t, 'q'))) for p in (0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)} for t in f["tasks"]}}
                          for j, f in fits.items()}}
    # prior diagnostics: mean prior P(tier) vs 50-ep tier (descriptive only)
    t50 = tier_int(pool["Y"].sum(1), 50)
    rep["prior_diagnostics"] = {m: {"mean_prior_tierprob_by_tier50": {TIERS[t]: (priors[m][t50 == t] @ TMASK).mean(0).round(3).tolist() for t in range(3)},
                                    "square_run_mean_prior_P_GOOD": float((priors[m][pool["task"] == "square"] @ TMASK)[:, 2].mean()),
                                    "can_runs_mean_prior_P_GOOD": float((priors[m][pool["task"] == "can"] @ TMASK)[:, 2].mean())}
                                for m in priors}
    print("validation ...", flush=True)
    rep["judge_validation"] = validate(evs, pool, args)
    print("real tier check ...", flush=True)
    rep["world_R_real_tier_check"] = real_tier_check(evs, pool, args)
    for w in args.worlds.split(","):
        print(f"world {w} ...", flush=True)
        key = "world_A_split" if w == "A" else "world_B_parametric"
        rep[key] = run_world(w, pool, fits, priors, args, None)
        print(f"  done {time.time() - t0:.0f}s", flush=True)
    rep["runtime_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    if args.fig and "world_A_split" in rep and "world_B_parametric" in rep:
        make_fig(rep, args.fig)
    print("wrote", args.out, f"{rep['runtime_s']}s")


if __name__ == "__main__":
    main()
