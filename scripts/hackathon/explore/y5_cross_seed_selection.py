#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- y5_cross_seed_selection.

Question
--------
A practitioner trains several seeds and must ship ONE checkpoint across runs.
"Latest epoch" cannot separate two seeds at the same epoch.  Does a cheap PLV
metric (or K=10 eval rollouts judged by a VLM) help in that cross-seed selection
problem?

Data
----
* The two Can runs (can_seed1: 41 ckpts every 5 epochs 0-200; can_seed2: 21 ckpts
  every 10 epochs 0-200).  Cheap metrics from results/hackathon/baseline_metrics.csv
  (Experiment 1, frozen paper flags).
* Targets: training-time per-episode outcomes ``test/sim_max_reward_<seed>``
  (seeds 100000-100049) parsed from each run's logs.json.txt.
    S50 = mean over all 50 (used only for outcome-free methods),
    HA  = seeds 100000-100024, HB = seeds 100025-100049.
  Methods that USE outcomes (K=25 training-rollout success, "metric chosen on a
  half") are fit on one half and evaluated ONLY on the other half.
* Eval checkpoints (multiples of 20): K=10 eval rollouts (seeds 200000-200009,
  disjoint from the target seeds) -> instrumented success and VLM-judged success
  (Cosmos3-Edge local v6/v7, Nemotron Nebius v2 and its length-neutral pad201).

Selection pools (one pooled selection problem across the two seeds)
  common10      : epochs that are multiples of 10 (21 per seed, 42 ckpts)
  all           : all 62 checkpoints
  late_common10 : common10 with epoch >= 100 (22 ckpts)
  eval20        : multiples of 20 (11 per seed, 22 ckpts) -- the only pool where
                  K=10 eval / VLM methods exist

Statistics (score: higher = better ranked)
  * Spearman(score, target), Regret@q (q = 0.1, 0.3) = max S - max S in top
    ceil(qN), top-1 regret (k = 1).  All are EXACT expectations over uniformly
    random tie-breaks of the score (epoch ties between seeds, K=10 ties at 1.0):
      E[Spearman] = midrank covariance / sqrt(untied-rank var * target var)
      E[top-k max] computed combinatorially for the tied group at the cut.
    A Monte-Carlo check of the epoch numbers is stored in the JSON.
  * 95% CIs: joint bootstrap of (i) moving blocks of epoch bins (block = 40
    epochs; a bin carries the checkpoints of BOTH seeds at those epochs, which
    preserves the cross-seed pairing and the serial correlation of neighbouring
    checkpoints), (ii) training-time env seeds (same resample for every
    checkpoint, stratified by half), (iii) eval env seeds 200000-200009 (same
    resample for every checkpoint and judge).  Paired differences vs epoch use
    the same replicate.  A run-level bootstrap is impossible with 2 runs.
  * Epoch-matched pairwise test: at each common epoch, does the metric pick the
    seed whose checkpoint has the higher success?  Primary: pairs with
    |dS50| >= 0.06; robustness: pairs filtered on one half and scored on the
    other half.  Clopper-Pearson CIs, exact binomial test vs 0.5, a
    constant-pick control and a circular-shift null (keeps serial correlation).

Run from the repo root:
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y5_cross_seed_selection.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import comb  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.ranking import regret_at_q, spearman  # noqa: E402  (cross-check only)

NAME = "y5_cross_seed_selection"
RUN_DIRS = {
    "can_seed1": "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100",
    "can_seed2": "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed2_e200_s200_r10_inf100",
}
TRAIN_SEEDS = list(range(100000, 100050))
HALF = {"A": np.arange(0, 25), "B": np.arange(25, 50)}
EVAL_SEEDS = list(range(200000, 200010))
JUDGES = {
    "vlm_local_v6": "data/outputs/hackathon_cosmos/eval_local_v6.jsonl",
    "vlm_nebius_v2": "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl",
    "vlm_nebius_v2_pad201": "data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl",
    "vlm_local_v7": "data/outputs/hackathon_cosmos/eval_local_v7.jsonl",
}
# cheap, outcome-free metrics from Experiment 1 (all are costs: lower = better)
CHEAP = {
    "val_action_l1": "Action L1 (val)",
    "open_loop_smoothness": "Open-loop smoothness",
    "plv1_smoothness": "PLV-1 smoothness",
    "replan_overlap_h16": "Replan overlap H=16 (model fb)",
    "replan_overlap_h10": "Replan overlap H=10 (model fb)",
    "boundary_jump_h16": "Boundary jump H=16",
    "boundary_jump_h10": "Boundary jump H=10",
    "val_loss": "Validation loss",
    "train_loss": "Train loss",
    "support_drift": "Support drift",
    "dynamics_uncertainty": "Dynamics uncertainty",
    "teacher_forced_latent_mse": "Teacher-forced latent MSE",
}
K10 = ["k10_sim"] + list(JUDGES)
POOLS = {
    "common10": dict(sel=lambda e: e % 10 == 0, bin_w=10, block=4),
    "all": dict(sel=lambda e: np.ones_like(e, bool), bin_w=10, block=4),
    "late_common10": dict(sel=lambda e: (e % 10 == 0) & (e >= 100), bin_w=10, block=4),
    "eval20": dict(sel=lambda e: e % 20 == 0, bin_w=20, block=2),
}
QS = (0.1, 0.3)
PAIR_THR = 0.06


# ----------------------------------------------------------------------------- data
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
    Y = []
    for e in epochs:
        r = by_epoch[int(e)]
        Y.append([1.0 if float(r[f"test/sim_max_reward_{s}"]) > 0.5 else 0.0 for s in TRAIN_SEEDS])
    return np.array(Y)


def load_data():
    base = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv")
    ck = base[base.run_id.isin(RUN_DIRS)].sort_values(["run_id", "epoch"]).reset_index(drop=True)
    Y = np.zeros((len(ck), 50))
    for run, rd in RUN_DIRS.items():
        m = (ck.run_id == run).values
        Y[m] = load_train_outcomes(ROOT / rd, ck.epoch.values[m])
    err = np.abs(Y.mean(1) - ck.success.values).max()
    assert err < 1e-9, f"per-episode outcomes do not reproduce S_i (max err {err})"
    # eval rollouts + VLM judgements (only checkpoints with eval rollouts)
    E = np.full((len(ck), 10), np.nan)
    V = {j: np.full((len(ck), 10), np.nan) for j in JUDGES}
    pos = {c: i for i, c in enumerate(ck.checkpoint_id)}
    for c, i in pos.items():
        p = ROOT / "data/outputs/hackathon_rollouts_eval" / c / "summary.json"
        if not p.is_file():
            continue
        s = json.loads(p.read_text())
        for ep in s["episodes"]:
            E[i, EVAL_SEEDS.index(int(ep["env_seed"]))] = float(bool(ep["success"]))
    n_rows = {}
    for j, f in JUDGES.items():
        n_rows[j] = 0
        for line in open(ROOT / f):
            r = json.loads(line)
            if r["checkpoint_id"] not in pos:
                continue
            i, k = pos[r["checkpoint_id"]], EVAL_SEEDS.index(int(r["env_seed"]))
            assert bool(r.get("scored")), f"unscored row {j} {r['checkpoint_id']} {r['env_seed']}"
            assert float(bool(r["truth_success"])) == E[i, k], "VLM truth != eval summary"
            V[j][i, k] = float(bool(r["cosmos_task_success"]))
            n_rows[j] += 1
    has_eval = np.isfinite(E).all(1)
    assert set(ck.epoch.values[has_eval] % 20) == {0} and has_eval.sum() == 22
    for j in JUDGES:
        assert np.isfinite(V[j][has_eval]).all(), f"missing VLM rows for {j}"
    return ck, Y, E, V, has_eval, n_rows


# ----------------------------------------------------------------------------- stats
def exp_spearman(score, y):
    """E over uniform random tie-breaks of the score of Spearman(score, y).
    Target ties use midranks (standard)."""
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


def selection_stats(score, y):
    n = len(score)
    best = float(y.max())
    out = {"spearman": exp_spearman(score, y), "n": n}
    for q in QS:
        k = max(1, int(math.ceil(q * n)))
        out[f"regret@{int(q * 100)}"] = best - exp_topk_max(score, y, k)
    out["top1_regret"] = best - exp_topk_max(score, y, 1)
    return out


STAT_KEYS = ["spearman", "regret@10", "regret@30", "top1_regret"]


def ranks(x):
    return stats.rankdata(x)


# ----------------------------------------------------------------------------- methods
def outcome_free_methods(cki, Ei, Vi, with_k10):
    """cki: dataframe rows of the pool; Ei/Vi: eval arrays of the pool (episode-resampled)."""
    ep = cki.epoch.values.astype(float)
    n = len(ep)
    M = {"epoch": ep, "random": np.zeros(n)}
    r_ep = ranks(ep)
    for c in CHEAP:
        sc = -cki[c].values.astype(float)
        M[c] = sc
        M[f"rankavg_epoch+{c}"] = 0.5 * (r_ep + ranks(sc))
        M[f"lex_epoch>{c}"] = ep + 0.5 * ranks(sc) / (n + 1)  # epochs are >= 5 apart
    if with_k10:
        k10 = {"k10_sim": Ei.mean(1), **{j: Vi[j].mean(1) for j in JUDGES}}
        l1 = -cki["val_action_l1"].values.astype(float)
        for k, v in k10.items():
            M[k] = v
            M[f"rankavg_epoch+{k}"] = 0.5 * (r_ep + ranks(v))
            M[f"rankavg_l1+{k}"] = 0.5 * (ranks(l1) + ranks(v))  # Experiment-3 style hybrid
            M[f"lex_{k}>l1"] = v + 1e-3 * ranks(l1) / (n + 1)    # K=10 success, ties -> Action L1
    return M


def outcome_using_methods(cki, Yfit, free):
    """Fit on one half (Yfit = that half's per-episode outcomes of the pool)."""
    sfit = Yfit.mean(1)
    M = {"k25_rollouts_fit_half": sfit}
    # pick the cheap metric / any outcome-free candidate with the best Spearman on the fit half
    cand_cheap = list(CHEAP)
    cand_any = ["epoch"] + list(CHEAP) + [f"rankavg_epoch+{c}" for c in CHEAP]
    sel = {}
    for name, cands in (("fit_half_best_cheap", cand_cheap), ("fit_half_best_any", cand_any)):
        rhos = np.array([exp_spearman(free[c], sfit) for c in cands], dtype=float)
        if np.isfinite(rhos).any():
            b = cands[int(np.nanargmax(rhos))]
            sel[name] = b
        else:  # fit-half success constant in this (resampled) pool: no information -> first candidate
            b = cands[0]
            sel[name] = b + " (fallback: constant fit-half target)"
        M[name] = free[b]
    # rank-average of epoch and the K=25 fit-half success (cheap prior + rollouts)
    M["rankavg_epoch+k25_fit_half"] = 0.5 * (ranks(cki.epoch.values.astype(float)) + ranks(sfit))
    return M, sel


def evaluate_pool(ck, Y, E, V, idx, tr_idx, ev_idx, with_k10):
    """idx: checkpoint indices (may repeat); tr_idx: dict half->episode indices;
    ev_idx: eval-episode indices.  Returns {target: {method: stats}}, selections."""
    cki = ck.iloc[idx]
    Yi = Y[idx]
    YA, YB = Yi[:, HALF["A"]][:, tr_idx["A"]], Yi[:, HALF["B"]][:, tr_idx["B"]]
    Ei = E[idx][:, ev_idx] if with_k10 else None
    Vi = {j: V[j][idx][:, ev_idx] for j in JUDGES} if with_k10 else None
    free = outcome_free_methods(cki, Ei, Vi, with_k10)
    tgt = {"S50": np.concatenate([YA, YB], 1).mean(1), "HA": YA.mean(1), "HB": YB.mean(1)}
    res = {"S50": {}, "HB": {}, "HA": {}}
    for m, s in free.items():
        for t in res:
            res[t][m] = selection_stats(s, tgt[t])
    sels = {}
    for fit, ev in (("A", "HB"), ("B", "HA")):
        M, sel = outcome_using_methods(cki, YA if fit == "A" else YB, free)
        sels[ev] = sel
        for m, s in M.items():
            res[ev][m] = selection_stats(s, tgt[ev])
    return res, sels


def block_bins(epochs, bin_w, block, rng):
    bins = np.floor(epochs / bin_w).astype(int)
    ub = np.unique(bins)
    nb = len(ub)
    nblk = int(math.ceil(nb / block))
    starts = rng.integers(0, nb - block + 1, size=nblk)
    drawn = np.concatenate([np.arange(s, s + block) for s in starts])[:nb]
    members = {b: np.nonzero(bins == ub[b])[0] for b in range(nb)}
    return np.concatenate([members[b] for b in drawn])


# ----------------------------------------------------------------------------- pairwise
def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def pair_decisions(ck, E, V):
    """For each common epoch: dict of per-method 'pick seed2' in {0, 0.5, 1}."""
    rows = []
    s1 = ck[ck.run_id == "can_seed1"].set_index("epoch")
    s2 = ck[ck.run_id == "can_seed2"].set_index("epoch")
    ids = {c: i for i, c in enumerate(ck.checkpoint_id)}
    for e in sorted(set(s1.index) & set(s2.index)):
        i1, i2 = ids[s1.loc[e, "checkpoint_id"]], ids[s2.loc[e, "checkpoint_id"]]
        r = {"epoch": int(e), "i1": i1, "i2": i2}
        for c in CHEAP:
            a, b = float(ck.loc[i1, c]), float(ck.loc[i2, c])
            r[c] = 0.5 if a == b else float(b < a)
        r["epoch_method"] = 0.5
        if np.isfinite(E[i1]).all() and np.isfinite(E[i2]).all():
            for k, arr in (("k10_sim", E), *V.items()):
                a, b = arr[i1].mean(), arr[i2].mean()
                r[k] = 0.5 if a == b else float(b > a)
        rows.append(r)
    return rows


def pairwise_eval(rows, Y, methods, thr_from, score_on, thr=PAIR_THR, exclude_ties_target=True):
    """thr_from / score_on in {'S50','HA','HB'}.  Pair kept if |d(thr_from)| >= thr;
    correctness judged by sign of d(score_on)."""
    def sval(i, w):
        return Y[i].mean() if w == "S50" else Y[i][HALF[w[1]]].mean()
    kept = []
    for r in rows:
        d_f = sval(r["i2"], thr_from) - sval(r["i1"], thr_from)
        d_s = sval(r["i2"], score_on) - sval(r["i1"], score_on)
        if abs(d_f) >= thr - 1e-9:
            kept.append((r, d_f, d_s))
    out = {"n_pairs_passing_filter": len(kept), "epochs": [k[0]["epoch"] for k in kept]}
    target_ties = [k for k in kept if k[2] == 0]
    usable = [k for k in kept if k[2] != 0] if exclude_ties_target else kept
    out["n_target_ties_excluded"] = len(target_ties)
    out["n_pairs"] = len(usable)
    out["frac_seed2_better"] = float(np.mean([k[2] > 0 for k in usable])) if usable else np.nan
    per = {}
    for m in methods:
        us = [k for k in usable if m in k[0]]
        if not us:
            continue
        dec = np.array([k[0][m] for k in us])
        truth = np.array([float(k[2] > 0) for k in us])
        corr = np.where(dec == 0.5, 0.5, (dec == truth).astype(float))
        n_t = int((dec == 0.5).sum())
        k_c = int((corr == 1).sum())
        n_nt = len(us) - n_t
        ci = clopper_pearson(k_c, n_nt)
        p = float(stats.binomtest(k_c, n_nt, 0.5).pvalue) if n_nt > 0 else np.nan
        per[m] = {"n": len(us), "n_method_ties(scored_0.5)": n_t, "n_correct": k_c,
                  "accuracy_ties_half": float(corr.mean()),
                  "accuracy_excl_ties": (k_c / n_nt) if n_nt else np.nan,
                  "cp95_excl_ties": ci, "binom_p_vs_0.5": p,
                  "frac_pick_seed2": float(np.mean(dec[dec != 0.5])) if n_nt else np.nan}
    out["per_method"] = per
    return out


def circular_shift_null(rows, Y, method, thr=PAIR_THR):
    """Shift the method's decision sequence over the common epochs (circularly, epochs
    with a defined decision only), keep the target and filter fixed."""
    seq = [r for r in rows if method in r]
    dec = np.array([r[method] for r in seq])
    d = np.array([Y[r["i2"]].mean() - Y[r["i1"]].mean() for r in seq])
    keep = np.abs(d) >= thr - 1e-9
    truth = (d > 0).astype(float)

    def acc(dd):
        c = np.where(dd == 0.5, 0.5, (dd == truth).astype(float))
        return c[keep].mean() if keep.any() else np.nan

    obs = acc(dec)
    null = np.array([acc(np.roll(dec, s)) for s in range(1, len(dec))])
    return {"observed": float(obs), "null_mean": float(np.nanmean(null)),
            "p_shift_ge_obs": float(np.mean(null >= obs - 1e-12)), "n_shifts": int(len(null))}


def pair_diff_spearman(rows, Y, method_vals, target="S50"):
    """Continuous version: Spearman over all common-epoch pairs of
    (cost(s1) - cost(s2)) vs (S(s2) - S(s1)), excluding epoch 0 where both are 0."""
    xs, ys = [], []
    for r in rows:
        if r["epoch"] == 0:
            continue
        a, b = method_vals[r["i1"]], method_vals[r["i2"]]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        y = (Y[r["i2"]].mean() - Y[r["i1"]].mean()) if target == "S50" else \
            (Y[r["i2"]][HALF[target[1]]].mean() - Y[r["i1"]][HALF[target[1]]].mean())
        xs.append(a - b)  # positive = seed2 has lower cost (metric prefers seed2)
        ys.append(y)
    return spearman(xs, ys), len(xs)


# ----------------------------------------------------------------------------- main
def summarize(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if len(v) == 0:
        return [np.nan, np.nan]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"results/hackathon/explore/{NAME}.json")
    a = ap.parse_args()
    t0 = time.time()
    ck, Y, E, V, has_eval, n_rows = load_data()
    ep_all = ck.epoch.values
    rng = np.random.default_rng(a.seed)

    pool_idx = {p: np.nonzero(cfg["sel"](ep_all))[0] for p, cfg in POOLS.items()}
    identity_tr = {"A": np.arange(25), "B": np.arange(25)}
    identity_ev = np.arange(10)

    # ---- point estimates
    point, sel_point = {}, {}
    for p, idx in pool_idx.items():
        point[p], sel_point[p] = evaluate_pool(ck, Y, E, V, idx, identity_tr, identity_ev, p == "eval20")

    # ---- cross-check exact tie expectation by Monte Carlo (epoch, common10, S50)
    idx = pool_idx["common10"]
    y = Y[idx].mean(1)
    ep = ck.epoch.values[idx].astype(float)
    mc = {"spearman": [], "regret@10": [], "regret@30": [], "top1_regret": []}
    rmc = np.random.default_rng(123)
    for _ in range(20000):
        jit = ep + rmc.random(len(ep)) * 1e-3
        mc["spearman"].append(spearman(jit, y))
        for q, key in ((0.1, "regret@10"), (0.3, "regret@30"), (1e-9, "top1_regret")):
            mc[key].append(regret_at_q(jit, y, q)["regret"])
    mc_check = {k: {"monte_carlo_20000": float(np.mean(v)), "exact": point["common10"]["S50"]["epoch"][k]}
                for k, v in mc.items()}
    mc_check["path_tiebreak_ranking_py"] = {  # what cosmos_plv.ranking would report (seed1 wins ties)
        "regret@10": regret_at_q(ep, y, 0.1, ck.checkpoint_path.values[idx])["regret"],
        "regret@30": regret_at_q(ep, y, 0.3, ck.checkpoint_path.values[idx])["regret"],
        "top1": regret_at_q(ep, y, 1e-9, ck.checkpoint_path.values[idx])["regret"]}

    # ---- bootstrap
    boot = {p: {} for p in pool_idx}
    sel_freq = {p: {} for p in pool_idx}
    for b in range(a.n_boot):
        tr = {"A": rng.integers(0, 25, 25), "B": rng.integers(0, 25, 25)}
        ev = rng.integers(0, 10, 10)
        for p, idx in pool_idx.items():
            cfg = POOLS[p]
            sub = block_bins(ep_all[idx], cfg["bin_w"], cfg["block"], rng)
            res, sels = evaluate_pool(ck, Y, E, V, idx[sub], tr, ev, p == "eval20")
            for t, mm in res.items():
                for m, st in mm.items():
                    d = boot[p].setdefault(t, {}).setdefault(m, {k: [] for k in STAT_KEYS})
                    for k in STAT_KEYS:
                        d[k].append(st[k])
            for t, sel in sels.items():
                for name, chosen in sel.items():
                    f = sel_freq[p].setdefault(t, {}).setdefault(name, {})
                    f[chosen] = f.get(chosen, 0) + 1
        if (b + 1) % 250 == 0:
            print(f"  bootstrap {b + 1}/{a.n_boot}  ({time.time() - t0:.0f}s)", flush=True)

    table = {}
    for p in pool_idx:
        table[p] = {}
        for t in point[p]:
            table[p][t] = {}
            for m, st in point[p][t].items():
                bt = boot[p][t][m]
                row = {"n": st["n"]}
                for k in STAT_KEYS:
                    row[k] = st[k]
                    row[f"{k}_ci95"] = summarize(bt[k])
                if "epoch" in boot[p][t] and m != "epoch":
                    be = boot[p][t]["epoch"]
                    for k in STAT_KEYS:
                        diff = np.array(bt[k]) - np.array(be[k])
                        row[f"{k}_minus_epoch"] = st[k] - point[p][t]["epoch"][k]
                        row[f"{k}_minus_epoch_ci95"] = summarize(diff)
                        # fraction of replicates where the method is better than epoch
                        better = diff > 0 if k == "spearman" else diff < 0
                        row[f"{k}_p_better_than_epoch"] = float(np.mean(better[np.isfinite(diff)]))
                table[p][t][m] = row

    # ---- split-half reliability of the target (noise ceiling)
    reliab = {}
    for p, idx in pool_idx.items():
        sa, sb = Y[idx][:, HALF["A"]].mean(1), Y[idx][:, HALF["B"]].mean(1)
        r = spearman(sa, sb)
        reliab[p] = {"spearman_halfA_halfB": r,
                     "spearman_brown_full50": (2 * r / (1 + r)) if np.isfinite(r) and r > -1 else np.nan}

    # ---- pairwise epoch-matched test
    rows = pair_decisions(ck, E, V)
    methods_pw = list(CHEAP) + ["epoch_method"] + K10
    pair_table = []
    for r in rows:
        i1, i2 = r["i1"], r["i2"]
        pr = {"epoch": r["epoch"], "S50_seed1": float(Y[i1].mean()), "S50_seed2": float(Y[i2].mean()),
              "dS50(s2-s1)": float(Y[i2].mean() - Y[i1].mean()),
              "dHA": float(Y[i2][HALF["A"]].mean() - Y[i1][HALF["A"]].mean()),
              "dHB": float(Y[i2][HALF["B"]].mean() - Y[i1][HALF["B"]].mean())}
        if "k10_sim" in r:
            pr["k10_sim_seed1"], pr["k10_sim_seed2"] = float(E[i1].mean()), float(E[i2].mean())
            for j in JUDGES:
                pr[f"{j}_seed1"], pr[f"{j}_seed2"] = float(V[j][i1].mean()), float(V[j][i2].mean())
        for c in CHEAP:
            pr[f"{c}_pick_seed2"] = r[c]
        pair_table.append(pr)
    pairwise = {
        "primary_S50_thr0.06": pairwise_eval(rows, Y, methods_pw, "S50", "S50"),
        "any_difference_S50": pairwise_eval(rows, Y, methods_pw, "S50", "S50", thr=0.02),
        "filter_HA_score_HB": pairwise_eval(rows, Y, methods_pw, "HA", "HB"),
        "filter_HB_score_HA": pairwise_eval(rows, Y, methods_pw, "HB", "HA"),
    }
    # target self-agreement among primary pairs: does half A's sign match half B's?
    agree = []
    for r in rows:
        d = Y[r["i2"]].mean() - Y[r["i1"]].mean()
        if abs(d) >= PAIR_THR - 1e-9:
            da = Y[r["i2"]][HALF["A"]].mean() - Y[r["i1"]][HALF["A"]].mean()
            db = Y[r["i2"]][HALF["B"]].mean() - Y[r["i1"]][HALF["B"]].mean()
            agree.append({"epoch": r["epoch"], "dA": float(da), "dB": float(db),
                          "agree": None if (da == 0 or db == 0) else bool(np.sign(da) == np.sign(db))})
    pairwise["target_split_half_sign_agreement_primary_pairs"] = agree
    # K=25 fit-half picker as a pairwise method (filter and score on the other half)
    for fit, sc in (("A", "B"), ("B", "A")):
        kept, cor = 0, 0.0
        for r in rows:
            df_ = Y[r["i2"]][HALF[fit]].mean() - Y[r["i1"]][HALF[fit]].mean()
            ds = Y[r["i2"]][HALF[sc]].mean() - Y[r["i1"]][HALF[sc]].mean()
            if abs(ds) >= PAIR_THR - 1e-9:
                kept += 1
                cor += 0.5 if df_ == 0 else float(np.sign(df_) == np.sign(ds))
        pairwise[f"k25_half{fit}_picks_on_half{sc}_pairs(|d{sc}|>=0.06)"] = {"n": kept, "accuracy_ties_half": cor / kept if kept else np.nan}
    pairwise["circular_shift_null_primary"] = {m: circular_shift_null(rows, Y, m) for m in methods_pw if m != "epoch_method"}
    pairwise["circular_shift_null_any_difference"] = {m: circular_shift_null(rows, Y, m, thr=0.02)
                                                      for m in methods_pw if m != "epoch_method"}
    pairwise["constant_pick_controls_any_difference"] = {
        "always_seed2": pairwise["any_difference_S50"]["frac_seed2_better"],
        "always_seed1": 1 - pairwise["any_difference_S50"]["frac_seed2_better"]}
    # Holm correction across the 12 cheap metrics (any-difference setting, ties excluded)
    pv = {m: pairwise["any_difference_S50"]["per_method"][m]["binom_p_vs_0.5"] for m in CHEAP}
    order = sorted(pv, key=lambda m: pv[m])
    holm, run_max = {}, 0.0
    for r_, m in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - r_) * pv[m]))
        holm[m] = run_max
    pairwise["holm_adjusted_p_any_difference_cheap"] = holm
    pairwise["constant_pick_controls_primary"] = {
        "always_seed2": pairwise["primary_S50_thr0.06"]["frac_seed2_better"],
        "always_seed1": 1 - pairwise["primary_S50_thr0.06"]["frac_seed2_better"]}
    # continuous within-pair correlation, with train-episode bootstrap CI
    pdc = {}
    rb = np.random.default_rng(7)
    for m in list(CHEAP) + K10:
        vals = ck[m].values.astype(float) if m in CHEAP else None
        if m in K10:
            arr = E if m == "k10_sim" else V[m]
            vals = -arr.mean(1)  # as a cost
        rho, n = pair_diff_spearman(rows, Y, vals)
        bs = []
        for _ in range(1000):
            ix = rb.integers(0, 50, 50)
            Yb = Y[:, ix]
            if m in K10:
                ev = rb.integers(0, 10, 10)
                arr = E if m == "k10_sim" else V[m]
                vb = -arr[:, ev].mean(1)
            else:
                vb = vals
            bs.append(pair_diff_spearman(rows, Yb, vb)[0])
        rho_b, _ = pair_diff_spearman(rows, Y, vals, target="HB")
        pdc[m] = {"spearman_dcost_vs_dS50": rho, "n_pairs": n, "ci95_episode_boot": summarize(bs),
                  "spearman_dcost_vs_dHB": rho_b}
    pairwise["pair_difference_spearman"] = pdc
    # held-out selection of the pairwise picker: choose the cheap metric with the best
    # any-difference pairwise accuracy on one half, score it on the other half's pairs
    ho = {}
    for fit, sc in (("A", "B"), ("B", "A")):
        pf = pairwise_eval(rows, Y, list(CHEAP), f"H{fit}", f"H{fit}", thr=0.04)
        accs = {m: v["accuracy_ties_half"] for m, v in pf["per_method"].items()}
        best = max(accs, key=lambda m: (accs[m], -list(CHEAP).index(m)))
        pe = pairwise_eval(rows, Y, [best], f"H{sc}", f"H{sc}", thr=0.04)
        ho[f"select_on_H{fit}_test_on_H{sc}"] = {"selected": best, "fit_accuracy": accs[best],
                                                 "fit_n_pairs": pf["n_pairs"],
                                                 "test": pe["per_method"].get(best), "test_n_pairs": pe["n_pairs"],
                                                 "test_frac_seed2_better": pe["frac_seed2_better"]}
    pairwise["heldout_selected_pairwise_metric"] = ho
    # context: within-run Spearman of each cheap metric (all checkpoints, S50)
    pairwise["context_within_run_spearman"] = {
        c: {r: spearman(-g[c].values, g.success.values) for r, g in ck.groupby("run_id")} for c in CHEAP}

    # ---- seed-level (one bit) summary of late checkpoints
    late = ck[(ck.epoch >= 100) & (ck.epoch % 10 == 0)]
    seed_level = {"late_mean_S50": late.groupby("run_id").success.mean().to_dict(),
                  "late_n_failures_of_550": {r: int(round((1 - g.success).sum() * 50)) for r, g in late.groupby("run_id")},
                  "late_mean_cost_prefers_seed2": {c: bool(late.groupby("run_id")[c].mean()["can_seed2"] <
                                                          late.groupby("run_id")[c].mean()["can_seed1"]) for c in CHEAP}}
    # 2x2 Fisher test on pooled late episodes (ignores within-run correlation; descriptive only)
    li = late.index.values
    seed_level["late_n_failures_by_half"] = {
        r: {h: int((1 - Y[li[(late.run_id == r).values]][:, HALF[h]]).sum()) for h in HALF} for r in RUN_DIRS}
    late_all = ck[ck.epoch >= 100]
    seed_level["late_all_ckpts_failures_per_env_seed"] = {
        r: {str(TRAIN_SEEDS[i]): int(v) for i, v in enumerate((1 - Y[late_all.index.values[(late_all.run_id == r).values]]).sum(0)) if v > 0}
        for r in RUN_DIRS}
    seed_level["late_all_ckpts_n"] = late_all.run_id.value_counts().to_dict()
    # POST-HOC sensitivity: the env seed with the most late failures (chosen after looking)
    worst = max(TRAIN_SEEDS, key=lambda s_: sum(seed_level["late_all_ckpts_failures_per_env_seed"][r].get(str(s_), 0) for r in RUN_DIRS))
    wi = TRAIN_SEEDS.index(worst)
    n1 = int(seed_level["late_all_ckpts_n"]["can_seed1"]); n2 = int(seed_level["late_all_ckpts_n"]["can_seed2"])
    k1 = seed_level["late_all_ckpts_failures_per_env_seed"]["can_seed1"].get(str(worst), 0)
    k2 = seed_level["late_all_ckpts_failures_per_env_seed"]["can_seed2"].get(str(worst), 0)
    seed_level["posthoc_worst_env_seed"] = {"env_seed": worst, "seed1_fail": f"{k1}/{n1}", "seed2_fail": f"{k2}/{n2}",
                                            "fisher_p_posthoc_not_corrected": float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])}
    keep_cols = np.array([i for i in range(50) if i != wi])
    Yx = Y[:, keep_cols]
    pw_x = {}
    for r in rows:
        d = Yx[r["i2"]].mean() - Yx[r["i1"]].mean()
        pw_x[r["epoch"]] = float(d)
    any_x = [r for r in rows if abs(pw_x[r["epoch"]]) > 1e-9]
    acc_x = {}
    for m in CHEAP:
        dec = np.array([r[m] for r in any_x]); tr_ = np.array([float(pw_x[r["epoch"]] > 0) for r in any_x])
        acc_x[m] = float(np.where(dec == 0.5, 0.5, (dec == tr_).astype(float)).mean()) if len(any_x) else np.nan
    seed_level["sensitivity_drop_worst_env_seed_any_difference_pairs"] = {
        "n_pairs": len(any_x), "frac_seed2_better": float(np.mean([pw_x[r["epoch"]] > 0 for r in any_x])) if any_x else np.nan,
        "accuracy": acc_x}
    f1, f2 = seed_level["late_n_failures_of_550"]["can_seed1"], seed_level["late_n_failures_of_550"]["can_seed2"]
    seed_level["fisher_p_pooled_episodes_descriptive"] = float(stats.fisher_exact([[f1, 550 - f1], [f2, 550 - f2]])[1])

    def sha(p):
        return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()

    out = {
        "name": NAME, "label": "EXPLORATORY / post-hoc (not pre-registered)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"baseline_metrics": {"path": "results/hackathon/baseline_metrics.csv",
                                        "sha256": sha("results/hackathon/baseline_metrics.csv")},
                   "logs": {r: {"path": f"{d}/logs.json.txt", "sha256": sha(f"{d}/logs.json.txt")} for r, d in RUN_DIRS.items()},
                   "judges": {j: {"path": f, "sha256": sha(f), "n_can_rows": n_rows[j]} for j, f in JUDGES.items()}},
        "protocol": {
            "pools": {p: {"n_checkpoints": int(len(i)), "epochs": sorted(set(int(e) for e in ep_all[i]))} for p, i in pool_idx.items()},
            "targets": {"S50": "training-time success, seeds 100000-100049 (outcome-free methods only)",
                        "HB": "seeds 100025-100049; outcome-using methods fit on 100000-100024",
                        "HA": "seeds 100000-100024; outcome-using methods fit on 100025-100049 (reverse fold)"},
            "ties": "exact expectation over uniform random tie-breaks of the score",
            "bootstrap": f"{a.n_boot} reps; moving blocks of epoch bins (40 epochs, both seeds per bin) x train env seeds (stratified by half) x eval env seeds",
            "pairwise_threshold": PAIR_THR, "qs": QS,
        },
        "tie_handling_check_common10_S50_epoch": mc_check,
        "split_half_reliability": reliab,
        "results": table,
        "fit_half_selection_point": sel_point,
        "fit_half_selection_bootstrap_freq": sel_freq,
        "pairwise": pairwise,
        "pair_table": pair_table,
        "seed_level_late": seed_level,
        "runtime_s": round(time.time() - t0, 1),
    }
    op = ROOT / a.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {op}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
