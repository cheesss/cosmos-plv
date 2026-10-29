#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- x1_adaptive_budget.

Question
--------
How many simulator (or real-robot) rollouts are needed to find a near-best
Diffusion-Policy checkpoint, and how much do
  (i)   adaptive allocation (successive halving, UCB1, Thompson, top-two Thompson),
  (ii)  an outcome-free cheap prior (keep the top 30 % by val_loss /
        train_action_mse_error / epoch, plus PLV metrics if
        results/hackathon/baseline_metrics.csv exists), and
  (iii) observing each rollout through a noisy VLM success judge
        (Cosmos3-Edge local v6, Nemotron Nebius v2, local->Nebius cascade)
change that number?

Protocol (no target leakage)
----------------------------
* Data: per-episode training-time outcomes ``test/sim_max_reward_<seed>`` (50
  per checkpoint) from each run's ``logs.json.txt``.  Each run is one selection
  problem over its saved checkpoints.
* Primary split: seeds 100000-100024 are the *rollout pool* (the only outcomes a
  strategy may observe, in a random order); seeds 100025-100049 are *held out*
  and are used ONLY to score the final pick:
      regret = max_c H(c) - H(pick),  H = held-out success over 25 episodes.
* Strategy hyper-parameters are fixed a priori (textbook defaults), never tuned
  on held-out outcomes.  Cheap priors use logged losses only (no outcomes).
* VLM-judge error rates are recomputed from the eval-split diagnostics JSONL
  (episodes of seeds 200000+, disjoint from the training-time episodes); each
  repetition draws (FPR, FNR) from their Beta(1+k, 1+n-k) posterior.
* Robustness: (a) independent (unpaired) episode orders, (b) many random 25/25
  re-splits of the 50 seeds.

Run from the repo root with env B:
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/x1_adaptive_budget.py
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
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.checkpoint_adapter import CKPT_RE  # noqa: E402  (no omegaconf needed)
from cosmos_plv.provenance import git_commit, utc_now  # noqa: E402
from cosmos_plv.ranking import bootstrap_ci, spearman  # noqa: E402

NAME = "x1_adaptive_budget"
MAIN_MULTS = (1, 2, 4, 8, 16, 25)
GRID_MULTS = (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, 25)
BASE_STRATS = ("uniform", "successive_halving", "ucb1", "thompson", "top_two_thompson")
PRIOR_INNER = ("uniform", "successive_halving", "top_two_thompson")
# outcome-free cheap priors: name -> (column, lower_is_better)
PRIORS = {
    "val_loss": ("val_loss", True),
    "train_action_mse": ("train_action_mse_error", True),
    "late_epoch": ("epoch", False),
    "random": (None, None),  # control: random 30 % subset per repetition
}
OPTIONAL_PLV = {  # read from results/hackathon/baseline_metrics.csv if it exists
    "val_action_l1": True, "open_loop_smoothness": True, "plv1_smoothness": True,
    "boundary_jump_h16": True, "replan_overlap_h16": True,
}
COSMOS = ROOT / "data/outputs/hackathon_cosmos"
JUDGE_FILES = {
    "cosmos3_edge_local_v6": COSMOS / "eval_local_v6.jsonl",
    "nemotron_nebius_v2": COSMOS / "eval_nebius_v2.jsonl",
}


# --------------------------------------------------------------------------- data
def load_runs(pool_root: Path):
    runs = []
    for run_dir in sorted(p for p in pool_root.iterdir() if p.is_dir()):
        logs = run_dir / "logs.json.txt"
        if not logs.is_file():
            continue
        cfgp = run_dir / ".hydra" / "config.yaml"
        cfg = yaml.safe_load(cfgp.read_text()) if cfgp.is_file() else {}
        task = str(((cfg.get("task") or {}).get("task_name")) or "unknown")
        seed = int(((cfg.get("training") or {}).get("seed")) or -1)
        er = ((cfg.get("task") or {}).get("env_runner") or {})
        n_test = int(er.get("n_test", 50))
        start = int(er.get("test_start_seed", 100000))
        seeds = list(range(start, start + n_test))
        by_epoch = {}
        for line in logs.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" not in row:
                continue
            cur = by_epoch.setdefault(int(row["epoch"]), {})
            cur.update({k: v for k, v in row.items() if isinstance(v, (int, float))})
        ck_dir = run_dir / "checkpoints"
        ck_epochs = set()
        if ck_dir.is_dir():
            for f in ck_dir.glob("epoch=*.ckpt"):
                m = CKPT_RE.search(f.name)
                if m:
                    ck_epochs.add(int(m.group("epoch")))
        eps, Y, met = [], [], {k: [] for k in ("val_loss", "train_loss", "train_action_mse_error")}
        for ep, r in sorted(by_epoch.items()):
            if "test/mean_score" not in r or ep not in ck_epochs:
                continue
            if not all(f"test/sim_max_reward_{s}" in r for s in seeds):
                continue
            eps.append(ep)
            Y.append([1.0 if float(r[f"test/sim_max_reward_{s}"]) > 0.5 else 0.0 for s in seeds])
            for k in met:
                met[k].append(float(r[k]) if k in r else np.nan)
        if len(eps) < 3:
            continue
        eps = np.array(eps)
        metrics, imputed = {"epoch": eps.astype(float)}, {}
        for k, v in met.items():
            v = np.array(v, dtype=float)
            ok = np.isfinite(v)
            imputed[k] = int((~ok).sum())
            if 0 < ok.sum() < len(v):  # outcome-free: linear interpolation over epoch
                v = np.interp(eps, eps[ok], v[ok])
            metrics[k] = v
        runs.append({
            "run": run_dir.name, "task": task, "seed": seed, "seeds": seeds,
            "epochs": eps, "ids": [f"{task}_s{seed}_e{e:04d}" for e in eps],
            "Y": np.array(Y), "metrics": metrics, "imputed": imputed,
        })
    return runs


def attach_plv_metrics(runs, csv_path: Path):
    """Optional PLV cheap metrics (outcome columns are never read)."""
    if not csv_path.is_file():
        return []
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in OPTIONAL_PLV if c in df.columns and df[c].notna().any()]
    if "checkpoint_id" not in df.columns:
        return []
    df = df.set_index("checkpoint_id")
    used = []
    for c in cols:
        ok_all = True
        for r in runs:
            v = np.array([float(df[c].get(i, np.nan)) if i in df.index else np.nan for i in r["ids"]])
            r["metrics"][c] = v
            ok_all &= bool(np.isfinite(v).sum() >= 3)
        if ok_all:
            used.append(c)
            PRIORS[c] = (c, OPTIONAL_PLV[c])
    return used


def judge_counts():
    """Confusion counts per judge and task, recomputed from the eval JSONL files."""
    rows = {}
    for name, path in JUDGE_FILES.items():
        rows[name] = {}
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("scored") or r.get("parse_status") != "ok":
                continue
            if not isinstance(r.get("truth_success"), bool) or not isinstance(r.get("cosmos_task_success"), bool):
                continue
            rows[name][(r["checkpoint_id"], int(r["env_seed"]))] = (r["task"], r["truth_success"], r["cosmos_task_success"])
            FTYPE.setdefault(name, {})[(r["checkpoint_id"], int(r["env_seed"]))] = r.get("truth_failure_type")
    # cascade (rule fixed a priori by x0_cascade on dev): local says fail -> fail; else ask Nemotron
    L, N = rows.get("cosmos3_edge_local_v6", {}), rows.get("nemotron_nebius_v2", {})
    rows["cascade_local_then_nebius"] = {k: (L[k][0], L[k][1], L[k][2] and N[k][2]) for k in sorted(set(L) & set(N))}
    FTYPE["cascade_local_then_nebius"] = FTYPE.get("cosmos3_edge_local_v6", {})
    out = {}
    for name, d in rows.items():
        out[name] = {}
        for task in sorted({v[0] for v in d.values()}) + ["all"]:
            vals = [v for v in d.values() if task == "all" or v[0] == task]
            nf = sum(1 for _, t, _ in vals if not t)
            ns = sum(1 for _, t, _ in vals if t)
            fp = sum(1 for _, t, j in vals if not t and j)
            fn = sum(1 for _, t, j in vals if t and not j)
            out[name][task] = {"n_fail": nf, "fp": fp, "n_succ": ns, "fn": fn,
                               "fpr": fp / nf if nf else float("nan"), "fnr": fn / ns if ns else float("nan"),
                               "fpr_wilson95": list(wilson(fp, nf)), "fnr_wilson95": list(wilson(fn, ns))}
        # failure-type split of the false positives (all tasks pooled; too few per task)
        ft = FTYPE.get(name, {})
        fd = [(k, v) for k, v in d.items() if not v[1]]
        nd = sum(1 for k, v in fd if ft.get(k) == "dropped")
        kd = sum(1 for k, v in fd if ft.get(k) == "dropped" and v[2])
        no = sum(1 for k, v in fd if ft.get(k) != "dropped")
        ko = sum(1 for k, v in fd if ft.get(k) != "dropped" and v[2])
        out[name]["by_failure_type"] = {"dropped": {"n_fail": nd, "fp": kd}, "not_dropped": {"n_fail": no, "fp": ko}}
    return out


FTYPE = {}


def dropped_share_from_eval(eval_root: Path, hi_threshold=0.6):
    """Share of failures that are near-miss 'dropped' failures, pooled over eval checkpoints
    with eval success >= hi_threshold (hi) or < hi_threshold (lo).  Recorded robosuite rollouts."""
    acc = {"hi": [0, 0], "lo": [0, 0]}
    for sp in sorted(eval_root.glob("*/summary.json")):
        eps = json.loads(sp.read_text()).get("episodes", [])
        if not eps:
            continue
        sr = np.mean([bool(e["success"]) for e in eps])
        b = "hi" if sr >= hi_threshold else "lo"
        for e in eps:
            if not e["success"]:
                acc[b][1] += 1
                acc[b][0] += int(e.get("failure_type") == "dropped")
    return {"threshold": hi_threshold,
            "hi": {"dropped": acc["hi"][0], "failures": acc["hi"][1]},
            "lo": {"dropped": acc["lo"][0], "failures": acc["lo"][1]}}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def judge_rate_draws(spec, counts, task, R, rng, pool_success=None, dshare=None):
    """Per-repetition (FPR, FNR) arrays [(R,1) or (R,N)], or None for a perfect judge.
    *_ftype: FPR depends on the checkpoint's failure mix (near-miss 'dropped' failures are
    judged success far more often); P(dropped | failure) = hi/lo share from the eval rollouts,
    chosen by the checkpoint's POOL success (>= threshold) -- a world-model choice, never seen
    by the strategies."""
    if spec == "perfect":
        return None
    if spec.endswith("_ftype"):
        base = spec[:-len("_ftype")]
        c = counts[base].get(task) or counts[base]["all"]
        bf = counts[base]["by_failure_type"]
        fd = rng.beta(1 + bf["dropped"]["fp"], 1 + bf["dropped"]["n_fail"] - bf["dropped"]["fp"], size=R)
        fo = rng.beta(1 + bf["not_dropped"]["fp"], 1 + bf["not_dropped"]["n_fail"] - bf["not_dropped"]["fp"], size=R)
        qh = rng.beta(1 + dshare["hi"]["dropped"], 1 + dshare["hi"]["failures"] - dshare["hi"]["dropped"], size=R)
        ql = rng.beta(1 + dshare["lo"]["dropped"], 1 + dshare["lo"]["failures"] - dshare["lo"]["dropped"], size=R)
        hi = (np.asarray(pool_success) >= dshare["threshold"])[None, :]
        q = np.where(hi, qh[:, None], ql[:, None])
        fpr = q * fd[:, None] + (1 - q) * fo[:, None]
        fnr = rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"], size=R)[:, None]
        return fpr, fnr
    base, mode = (spec[:-len("_pessimistic")], "pess") if spec.endswith("_pessimistic") else (spec, "post")
    c = counts[base].get(task) or counts[base]["all"]
    if c["n_fail"] < 5 or c["n_succ"] < 5:
        c = counts[base]["all"]
    if mode == "pess":
        return (np.full((R, 1), c["fpr_wilson95"][1]), np.full((R, 1), c["fnr_wilson95"][1]))
    return (rng.beta(1 + c["fp"], 1 + c["n_fail"] - c["fp"], size=R)[:, None],
            rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"], size=R)[:, None])


# --------------------------------------------------------------------------- strategies
# All strategies get prefix [R, k, P+1] (cumulative observed successes of each arm's
# episodes in the per-repetition order) and return an index into the k arms.
# Recommendation rule for every strategy: max Beta(1,1) posterior mean, random tie-break.

def _score(s, n, jitter):
    return (s + 1.0) / (n + 2.0) + jitter


def strat_uniform(prefix, budgets, rng, jitter):
    R, k, P1 = prefix.shape
    P = P1 - 1
    pri = rng.random((R, k))
    rank = (-pri).argsort(1).argsort(1)
    out = {}
    for B in budgets:
        n = np.minimum(B // k + (rank < (B % k)), P)
        s = np.take_along_axis(prefix, n[:, :, None], axis=2)[:, :, 0]
        out[B] = (_score(s, n, jitter).argmax(1), float(n.sum(1).mean()))
    return out


def strat_successive_halving(prefix, budgets, rng, jitter):
    """Karnin et al. (2013) SH: ceil(log2 k) rounds, budget spread evenly over the
    remaining rounds, survivors halved by posterior mean; per-arm cap = pool size."""
    R, k, P1 = prefix.shape
    P = P1 - 1
    out = {}
    for B in budgets:
        rounds = max(1, math.ceil(math.log2(k))) if k > 1 else 1
        surv = np.ones((R, k), bool)
        m, cnt, remaining, used = k, 0, int(B), 0
        for r in range(rounds):
            per = max(1, (remaining // (rounds - r)) // m)
            per = max(0, min(per, remaining // m, P - cnt))
            cnt += per
            remaining -= per * m
            used += per * m
            if m == 1:
                break
            sc = np.where(surv, _score(prefix[:, :, cnt], cnt, jitter), -np.inf)
            m_new = math.ceil(m / 2)
            keep = np.argsort(-sc, axis=1)[:, :m_new]
            surv = np.zeros((R, k), bool)
            np.put_along_axis(surv, keep, True, axis=1)
            m = m_new
            if m == 1:
                break
        sc = np.where(surv, _score(prefix[:, :, cnt], cnt, jitter), -np.inf)
        out[B] = (sc.argmax(1), float(used))
    return out


def strat_anytime(prefix, budgets, rng, jitter, rule):
    """Sequential one-rollout-at-a-time allocation; recommendation recorded at each budget."""
    R, k, P1 = prefix.shape
    P = P1 - 1
    n = np.zeros((R, k), np.int64)
    s = np.zeros((R, k), np.int64)
    ar = np.arange(R)
    bset = sorted(set(int(b) for b in budgets))
    out = {}
    Bmax = bset[-1]
    for t in range(1, Bmax + 1):
        avail = n < P
        if not avail.any():
            for b in bset:
                if b >= t and b not in out:
                    out[b] = (_score(s, n, jitter).argmax(1), float(n.sum(1).mean()))
            break
        if rule == "ucb1":
            nn = np.maximum(n, 1)
            idx = np.where(n == 0, 1e6, s / nn + np.sqrt(2.0 * math.log(t) / nn)) + jitter
            a = np.where(avail, idx, -np.inf).argmax(1)
        elif rule == "thompson":
            th = rng.beta(1 + s, 1 + n - s)
            a = np.where(avail, th, -np.inf).argmax(1)
        elif rule == "top_two_thompson":  # Russo (2016), beta = 0.5, one challenger draw
            lead = rng.beta(1 + s, 1 + n - s).argmax(1)
            th2 = np.where(avail, rng.beta(1 + s, 1 + n - s), -np.inf)
            th2[ar, lead] = -np.inf
            chal = th2.argmax(1)
            lead_ok = avail[ar, lead]
            chal_ok = np.isfinite(th2[ar, chal])
            coin = rng.random(R) < 0.5
            a = np.where(lead_ok & (coin | ~chal_ok), lead, chal)
        else:
            raise ValueError(rule)
        can = avail[ar, a]
        na = n[ar, a]
        y = prefix[ar, a, np.minimum(na + 1, P)] - prefix[ar, a, na]
        rows = ar[can]
        n[rows, a[can]] += 1
        s[rows, a[can]] += y[can]
        if t in bset:
            out[t] = (_score(s, n, jitter).argmax(1), float(n.sum(1).mean()))
    for b in bset:
        if b not in out:
            out[b] = (_score(s, n, jitter).argmax(1), float(n.sum(1).mean()))
    return out


def run_strategy(name, prefix, budgets, rng, jitter):
    if name == "uniform":
        return strat_uniform(prefix, budgets, rng, jitter)
    if name == "successive_halving":
        return strat_successive_halving(prefix, budgets, rng, jitter)
    return strat_anytime(prefix, budgets, rng, jitter, name)


# --------------------------------------------------------------------------- simulation
def prior_subset(run, prior, k30, R, rng):
    """[R, k30] arm indices of the cheap-prior subset (outcome-free)."""
    N = len(run["epochs"])
    col, lower = PRIORS[prior]
    if col is None:
        return np.argsort(rng.random((R, N)), axis=1)[:, :k30]
    v = np.asarray(run["metrics"][col], float)
    key = np.where(np.isfinite(v), v if lower else -v, np.inf)
    order = sorted(range(N), key=lambda i: (key[i], -run["epochs"][i]))  # tie -> later epoch
    return np.broadcast_to(np.array(order[:k30]), (R, k30)).copy()


def simulate_split(run, pool_idx, ho_idx, judges, counts, R, seed, paired=True, strategies=None, dshare=None):
    """Returns {(judge, strategy): {B: regret[R]}} plus bookkeeping."""
    Y = run["Y"]
    N = Y.shape[0]
    Yp, H = Y[:, pool_idx], Y[:, ho_idx].mean(1)
    P = Yp.shape[1]
    ss = np.random.SeedSequence(seed)
    c_order, c_noise, c_jit, c_rates, c_sub, c_strat = [np.random.default_rng(x) for x in ss.spawn(6)]
    if paired:
        perm = np.broadcast_to(np.argsort(c_order.random((R, 1, P)), axis=2), (R, N, P))
    else:
        perm = np.argsort(c_order.random((R, N, P)), axis=2)
    Yord = np.take_along_axis(np.broadcast_to(Yp, (R, N, P)), perm, axis=2)
    U = c_noise.random((R, N, P))  # common random numbers across judges
    jitter_full = c_jit.random((R, N)) * 1e-9
    budgets = sorted({max(1, int(round(m * N))) for m in GRID_MULTS})
    k30 = max(1, math.ceil(0.3 * N))
    subsets = {p: prior_subset(run, p, k30, R, c_sub) for p in PRIORS}
    strategies = strategies or ([*BASE_STRATS] + [f"{p}_top30+{s}" for p in PRIORS for s in PRIOR_INNER])
    res, used = {}, {}
    for j in judges:
        rates = judge_rate_draws(j, counts, run["task"], R, np.random.default_rng([seed, hash_str(j)]),
                                 pool_success=Yp.mean(1), dshare=dshare)
        if rates is None:
            Yobs = Yord
        else:
            fpr, fnr = rates
            Yobs = np.where(Yord > 0.5, U >= fnr[:, :, None], U < fpr[:, :, None]).astype(np.float64)
        prefix_full = np.concatenate([np.zeros((R, N, 1)), np.cumsum(Yobs, axis=2)], axis=2).round().astype(np.int64)
        for st in strategies:
            if "_top30+" in st:
                p, inner = st.split("_top30+")
                arms = subsets[p]
            else:
                inner, arms = st, np.broadcast_to(np.arange(N), (R, N))
            prefix = np.take_along_axis(prefix_full, arms[:, :, None], axis=1)
            jit = np.take_along_axis(jitter_full, arms, axis=1)
            srng = np.random.default_rng([seed, hash_str(st)])
            out = run_strategy(inner, prefix, budgets, srng, jit)
            res[(j, st)] = {}
            used[(j, st)] = {}
            for B, (pick_local, u) in out.items():
                pick = np.take_along_axis(arms, pick_local[:, None], axis=1)[:, 0]
                res[(j, st)][B] = H.max() - H[pick]
                used[(j, st)][B] = u
    return res, used, budgets, H


def hash_str(s):
    import zlib
    return zlib.crc32(s.encode())


def boot_mean(values, rng, nb):
    """Percentile-bootstrap draws of the mean of iid repetitions (exact via multinomial)."""
    vals, cnt = np.unique(values, return_counts=True)
    return rng.multinomial(len(values), cnt / cnt.sum(), size=nb) @ vals / len(values)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pool", default="data/outputs/hackathon_pool")
    ap.add_argument("--baseline-csv", default="results/hackathon/baseline_metrics.csv")
    ap.add_argument("--eval-rollouts", default="data/outputs/hackathon_rollouts_eval")
    ap.add_argument("--out", default=f"results/hackathon/explore/{NAME}.json")
    ap.add_argument("--fig", default=f"docs/hackathon/explore/{NAME}_regret.png")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=30)
    ap.add_argument("--split-reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--threshold", type=float, default=0.02)
    a = ap.parse_args()
    t0 = time.time()
    runs = load_runs(ROOT / a.pool)
    plv_used = attach_plv_metrics(runs, ROOT / a.baseline_csv)
    counts = judge_counts()
    dshare = dropped_share_from_eval(ROOT / a.eval_rollouts)
    judges_main = ["perfect", "cosmos3_edge_local_v6", "nemotron_nebius_v2", "cascade_local_then_nebius",
                   "cosmos3_edge_local_v6_pessimistic", "nemotron_nebius_v2_pessimistic",
                   "cosmos3_edge_local_v6_ftype", "nemotron_nebius_v2_ftype"]
    judges_split = ["perfect", "cosmos3_edge_local_v6", "nemotron_nebius_v2",
                    "cosmos3_edge_local_v6_ftype", "nemotron_nebius_v2_ftype"]
    rng_boot = np.random.default_rng(a.seed + 7)

    # ---- context: data summary, cheap-metric correlations, prior-only picks, reference
    n_seeds = len(runs[0]["seeds"])
    half = n_seeds // 2
    pool_idx, ho_idx = np.arange(half), np.arange(half, n_seeds)
    context = []
    for r in runs:
        Y = r["Y"]
        H = Y[:, ho_idx].mean(1)
        Pm = Y[:, pool_idx].mean(1)
        N = len(H)
        k30 = max(1, math.ceil(0.3 * N))
        best_pool = np.flatnonzero(Pm == Pm.max())
        ref_regret = float(H.max() - H[best_pool].mean())  # full-pool uniform, random tie-break
        pri = {}
        for p, (col, lower) in PRIORS.items():
            if col is None:
                continue
            v = np.asarray(r["metrics"][col], float)
            sc = -v if lower else v
            sub = prior_subset(r, p, k30, 1, np.random.default_rng(0))[0]
            rho = spearman(sc, H)
            ci = bootstrap_ci(lambda ix: spearman(sc[ix], H[ix]), N, n_boot=2000, seed=1)
            pri[p] = {"spearman_vs_heldout": rho, "spearman_ci95": list(ci),
                      "top30_epochs": [int(r["epochs"][i]) for i in sub],
                      "top30_floor_regret": float(H.max() - H[sub].max()),
                      "prior_only_pick_epoch": int(r["epochs"][sub[0]]),
                      "prior_only_regret": float(H.max() - H[sub[0]])}
        context.append({
            "run": r["run"], "task": r["task"], "seed": r["seed"], "n_checkpoints": N,
            "epochs": r["epochs"].tolist(), "checkpoint_ids": r["ids"],
            "pool_success": Pm.round(4).tolist(), "heldout_success": H.round(4).tolist(),
            "heldout_best": float(H.max()), "n_heldout_best_ties": int((H == H.max()).sum()),
            "n_within_0.04_of_best": int((H >= H.max() - 0.04 - 1e-9).sum()),
            "full_pool_uniform_regret_noise_floor": ref_regret,
            "imputed_metric_values": r["imputed"], "priors": pri,
            "paper_budget_rollouts": 50 * N,
        })
        r["ref_regret"] = ref_regret

    # ---- primary split
    prim, prim_used, budgets_by_run = {}, {}, {}
    for ri, r in enumerate(runs):
        res, used, budgets, _ = simulate_split(r, pool_idx, ho_idx, judges_main, counts, a.reps,
                                               seed=a.seed * 10 + ri, dshare=dshare)
        prim[r["run"]], prim_used[r["run"]], budgets_by_run[r["run"]] = res, used, budgets
        print(f"[{time.time() - t0:6.1f}s] primary {r['run']} done", flush=True)
    strategies = list(next(iter(prim.values())).keys())

    def summarize(resd, usedd, label):
        rows = []
        for (j, st) in sorted({k for d in resd.values() for k in d}):
            for mi, m in enumerate(GRID_MULTS):
                per, boots = [], []
                for r in runs:
                    N = len(r["epochs"])
                    B = max(1, int(round(m * N)))
                    reg = resd[r["run"]][(j, st)][B]
                    bm = boot_mean(reg, rng_boot, a.n_boot)
                    boots.append(bm)
                    per.append({"run": r["run"], "B": B, "mean_regret": float(reg.mean()),
                                "ci95": [float(x) for x in np.percentile(bm, [2.5, 97.5])],
                                "p_regret0": float((reg < 1e-9).mean()),
                                "p_regret_le_0.04": float((reg < 0.04 + 1e-9).mean()),
                                "excess_vs_full_pool": float(reg.mean() - r["ref_regret"]),
                                "mean_rollouts_used": usedd[r["run"]][(j, st)][B]})
                mb = np.mean(boots, axis=0)
                rows.append({"set": label, "judge": j, "strategy": st, "budget_mult": m,
                             "fraction_of_paper_budget": m / 50.0,
                             "mean_regret": float(np.mean([p["mean_regret"] for p in per])),
                             "ci95": [float(x) for x in np.percentile(mb, [2.5, 97.5])],
                             "mean_excess_vs_full_pool": float(np.mean([p["excess_vs_full_pool"] for p in per])),
                             "mean_rollouts_used_frac_of_B": float(np.mean([p["mean_rollouts_used"] / p["B"] for p in per])),
                             "per_run": per})
        return rows

    rows_primary = summarize(prim, prim_used, "primary_paired")
    print(f"[{time.time() - t0:6.1f}s] primary summarized", flush=True)

    def thresholds(rows, thr):
        out = []
        for (j, st) in sorted({(x["judge"], x["strategy"]) for x in rows}):
            rs = sorted([x for x in rows if x["judge"] == j and x["strategy"] == st], key=lambda x: x["budget_mult"])
            def first(pred):
                for x in rs:
                    if pred(x):
                        return x["budget_mult"]
                return None
            rec = {"judge": j, "strategy": st,
                   "mult_mean_regret_le_thr": first(lambda x: x["mean_regret"] <= thr + 1e-12),
                   "mult_ci_upper_le_thr": first(lambda x: x["ci95"][1] <= thr + 1e-12),
                   "mult_mean_regret_le_0.05": first(lambda x: x["mean_regret"] <= 0.05 + 1e-12),
                   "mult_excess_vs_full_pool_le_thr": first(lambda x: x["mean_excess_vs_full_pool"] <= thr + 1e-12),
                   "per_run_mult_mean_regret_le_thr": {}}
            for r in runs:
                m_ok = None
                for x in rs:
                    pr = [p for p in x["per_run"] if p["run"] == r["run"]][0]
                    if pr["mean_regret"] <= thr + 1e-12:
                        m_ok = x["budget_mult"]
                        break
                rec["per_run_mult_mean_regret_le_thr"][r["run"]] = m_ok
            for key in ("mult_mean_regret_le_thr", "mult_ci_upper_le_thr", "mult_excess_vs_full_pool_le_thr"):
                rec[key.replace("mult", "paper_budget_fraction")] = None if rec[key] is None else rec[key] / 50.0
            out.append(rec)
        return out

    thr_primary = thresholds(rows_primary, a.threshold)

    # ---- sensitivity: independent (unpaired) episode order, perfect + nebius judge
    ind, ind_used = {}, {}
    for ri, r in enumerate(runs):
        res, used, _, _ = simulate_split(r, pool_idx, ho_idx, ["perfect", "nemotron_nebius_v2"], counts, a.reps,
                                         seed=a.seed * 10 + 100 + ri, paired=False, dshare=dshare)
        ind[r["run"]], ind_used[r["run"]] = res, used
    rows_ind = summarize(ind, ind_used, "independent_order")
    thr_ind = thresholds(rows_ind, a.threshold)
    print(f"[{time.time() - t0:6.1f}s] independent-order sensitivity done", flush=True)

    # ---- robustness: random 25/25 re-splits of the 50 seeds
    split_rng = np.random.default_rng(a.seed + 99)
    split_curves = {}  # (judge, strategy, mult) -> list over splits of mean-over-runs regret
    split_floor = []
    for si in range(a.n_splits):
        perm = split_rng.permutation(n_seeds)
        pi, hi = np.sort(perm[:half]), np.sort(perm[half:])
        per_run_means = {}
        floors = []
        for ri, r in enumerate(runs):
            res, _, _, H = simulate_split(r, pi, hi, judges_split, counts, a.split_reps,
                                          seed=a.seed * 1000 + si * 10 + ri, dshare=dshare)
            Pm = r["Y"][:, pi].mean(1)
            floors.append(float(H.max() - H[Pm == Pm.max()].mean()))
            N = len(r["epochs"])
            for key, d in res.items():
                for m in GRID_MULTS:
                    B = max(1, int(round(m * N)))
                    per_run_means.setdefault((*key, m), []).append(float(d[B].mean()))
        for key, v in per_run_means.items():
            split_curves.setdefault(key, []).append(float(np.mean(v)))
        split_floor.append(float(np.mean(floors)))
        print(f"[{time.time() - t0:6.1f}s] split {si + 1}/{a.n_splits}", flush=True)
    rows_split = []
    for (j, st, m), v in sorted(split_curves.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        v = np.array(v)
        bm = [v[rng_boot.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        rows_split.append({"judge": j, "strategy": st, "budget_mult": m, "mean_regret_over_splits": float(v.mean()),
                           "boot_ci95_over_splits": [float(x) for x in np.percentile(bm, [2.5, 97.5])],
                           "split_spread_2.5_97.5": [float(x) for x in np.percentile(v, [2.5, 97.5])]})
    thr_split = []
    for (j, st) in sorted({(x["judge"], x["strategy"]) for x in rows_split}):
        rs = sorted([x for x in rows_split if x["judge"] == j and x["strategy"] == st], key=lambda x: x["budget_mult"])
        f = next((x["budget_mult"] for x in rs if x["mean_regret_over_splits"] <= a.threshold + 1e-12), None)
        fl = float(np.mean(split_floor))
        fe = next((x["budget_mult"] for x in rs if x["mean_regret_over_splits"] - fl <= a.threshold + 1e-12), None)
        thr_split.append({"judge": j, "strategy": st, "mult_mean_regret_le_thr": f,
                          "paper_budget_fraction": None if f is None else f / 50.0,
                          "mult_excess_vs_mean_full_pool_floor_le_thr": fe,
                          "paper_budget_fraction_excess": None if fe is None else fe / 50.0})

    # ---- analytic judge-noise inflation (signal shrink factor 1 - FPR - FNR)
    analytic = {}
    for jn, d in counts.items():
        for task, c in d.items():
            if task == "by_failure_type":
                continue
            sh = 1 - c["fpr"] - c["fnr"]
            analytic.setdefault(jn, {})[task] = {"signal_factor": sh, "approx_budget_inflation": 1 / sh ** 2 if sh > 0 else None}

    result = {
        "name": NAME, "exploratory": True,
        "label": "EXPLORATORY / post-hoc simulation on logged training-time rollouts; not a pre-registered test",
        "generated_utc": utc_now(), "git_commit": git_commit(), "runtime_s": round(time.time() - t0, 1),
        "command": " ".join([sys.executable, *sys.argv]),
        "args": vars(a),
        "protocol": {
            "pool_seeds": [int(runs[0]["seeds"][i]) for i in pool_idx][::len(pool_idx) - 1],
            "heldout_seeds": [int(runs[0]["seeds"][i]) for i in ho_idx][::len(ho_idx) - 1],
            "regret": "max_c H(c) - H(pick); H = held-out success (25 episodes)",
            "recommendation_rule": "max Beta(1,1) posterior mean on observed (judge) outcomes, random tie-break",
            "episode_order": "paired: one random permutation of pool seeds per repetition, shared by all checkpoints",
            "judge_noise_ftype": "FPR_c = q_c*FPR_dropped + (1-q_c)*FPR_other, q_c = P(dropped|fail) from eval rollouts "
                                 "(hi if checkpoint pool success >= 0.6), all rates Beta-posterior draws per repetition",
            "judge_noise": "per-episode flips with P(judge success|true fail)=FPR, P(judge fail|true success)=FNR; "
                           "per-repetition FPR/FNR ~ Beta(1+k,1+n-k) from eval JSONL, task-specific; "
                           "_pessimistic = Wilson 95% upper bounds; cascade = local AND nebius",
            "priors": {k: {"column": v[0], "lower_is_better": v[1]} for k, v in PRIORS.items()},
            "plv_metrics_from_baseline_csv": plv_used,
            "budgets_mults": list(GRID_MULTS), "main_mults": list(MAIN_MULTS),
            "paper_budget": "50 rollouts x all checkpoints",
            "ci": "percentile bootstrap over repetitions (episode orderings / judge draws); "
                  "mean_over_runs CI from independent per-run bootstraps",
        },
        "judge_counts": counts, "analytic_noise_inflation": analytic, "dropped_share_eval_rollouts": dshare,
        "context": context,
        "primary": rows_primary, "thresholds_primary": thr_primary,
        "independent_order": rows_ind, "thresholds_independent_order": thr_ind,
        "random_splits": {"n_splits": a.n_splits, "reps_per_split": a.split_reps, "rows": rows_split,
                          "thresholds": thr_split, "full_pool_noise_floor_per_split": split_floor},
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"wrote {out}  ({time.time() - t0:.1f}s)")
    if a.fig:
        try:
            make_figure(rows_primary, ROOT / a.fig, context)
            print(f"wrote {ROOT / a.fig}")
        except Exception as e:  # figure is optional
            print("figure failed:", repr(e))
    print_tables(rows_primary, thr_primary)


def make_figure(rows, path, context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    judges = ["perfect", "cosmos3_edge_local_v6", "nemotron_nebius_v2"]
    strats = ["uniform", "successive_halving", "top_two_thompson", "late_epoch_top30+top_two_thompson",
              "val_loss_top30+top_two_thompson", "train_action_mse_top30+top_two_thompson"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    floor = np.mean([c["full_pool_uniform_regret_noise_floor"] for c in context])
    for ax, j in zip(axes, judges):
        for st in strats:
            rs = sorted([x for x in rows if x["judge"] == j and x["strategy"] == st], key=lambda x: x["budget_mult"])
            if not rs:
                continue
            m = [x["budget_mult"] for x in rs]
            ax.plot(m, [x["mean_regret"] for x in rs], marker="o", ms=3, label=st)
            ax.fill_between(m, [x["ci95"][0] for x in rs], [x["ci95"][1] for x in rs], alpha=0.15)
        ax.axhline(0.02, color="k", ls=":", lw=1)
        ax.axhline(floor, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"judge = {j}")
        ax.set_xlabel("budget B / N (rollouts per checkpoint); paper = 50")
    axes[0].set_ylabel("mean held-out regret (mean over runs)")
    axes[0].legend(fontsize=7)
    fig.suptitle("x1_adaptive_budget (EXPLORATORY): dotted = 0.02, dashed = full-pool (25N) uniform noise floor")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def print_tables(rows, thr):
    for j in ["perfect", "cosmos3_edge_local_v6", "nemotron_nebius_v2", "cascade_local_then_nebius"]:
        print(f"\n== judge={j}: mean regret over runs (95% CI) ==")
        strats = sorted({x["strategy"] for x in rows if x["judge"] == j})
        print("strategy".ljust(42) + "".join(f"{m}N".rjust(20) for m in MAIN_MULTS))
        for st in strats:
            line = st.ljust(42)
            for m in MAIN_MULTS:
                x = [r for r in rows if r["judge"] == j and r["strategy"] == st and r["budget_mult"] == m][0]
                line += f"{x['mean_regret']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}]".rjust(20)
            print(line)
    print("\n== judge=perfect, per run: mean regret at main budgets ==")
    for st in ["uniform", "successive_halving", "ucb1", "thompson", "top_two_thompson",
               "late_epoch_top30+top_two_thompson", "train_action_mse_top30+top_two_thompson",
               "val_loss_top30+top_two_thompson", "random_top30+top_two_thompson"]:
        for m in MAIN_MULTS:
            xs = [r for r in rows if r["judge"] == "perfect" and r["strategy"] == st and r["budget_mult"] == m]
            if xs:
                print(f"{st:42s} {m:>4}N " + "  ".join(f"{p['run'][9:24]}={p['mean_regret']:.3f}" for p in xs[0]["per_run"]))
    print("\n== budget multiple to reach mean regret <= thr (mean over runs) ==")
    for t in thr:
        print(f"{t['judge']:36s} {t['strategy']:42s} mean<=thr: {t['mult_mean_regret_le_thr']}  ciU<=thr: {t['mult_ci_upper_le_thr']}  "
              f"<=0.05: {t['mult_mean_regret_le_0.05']}  excess<=thr: {t['mult_excess_vs_full_pool_le_thr']}")


if __name__ == "__main__":
    main()
