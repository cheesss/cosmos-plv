#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- adversarial verification of z2_loro_hybrid.

Independent re-implementation (does NOT import z2_loro_hybrid.py) of the z2
leave-one-run-out (LORO) analysis, plus checks z2 did not run:

  V1  labels + point estimates on the fixed HA/HB split and the swap, for every
      z2 method, compared with z2's JSON (per-run rho).
  V2  repeated random 25/25 splits of the 50 training-time seeds (fit half /
      target half), full LORO with nested lambda per split -> distribution of
      d rho (method - epoch).  This replaces the single swap with many splits and
      avoids the duplicated-seed refit bias of a with-replacement bootstrap.
  V3  LORO with all 50 episodes: training runs fit on 50 eps, held-out run scored
      on its 50 eps (no outcome of the held-out run is ever used for fitting).
  V4  free (unfitted) scores against the full 50-episode target, seed-cluster
      bootstrap CI of d rho (no refit, so the percentile bootstrap is usable).
  V5  bootstrap-bias diagnostics read from z2's JSONs, and the moving-block
      edge effect (inclusion rate of first/last checkpoints).
  V6  claim / arithmetic checks (ranges, +vlm vs VLM-free pairs, power).

No outcome leakage: in every LORO fold the held-out run's outcomes are used only
as the evaluation target.  CPU only.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/z2_loro_hybrid_verify.py --n-splits 200 --workers 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
import multiprocessing as mp
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RUN_DIRS = {
    "can_seed1": "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100",
    "can_seed2": "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed2_e200_s200_r10_inf100",
    "square_seed1": "data/outputs/hackathon_pool/paper_dp_square_abs_ph_seed1_e300_s200_r10_inf100",
}
RUNS = list(RUN_DIRS)
JUDGES = {
    "vlm_cosmos_v6": "data/outputs/hackathon_cosmos/eval_local_v6.jsonl",
    "vlm_nemotron_v2": "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl",
    "vlm_nemotron_pad201": "data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl",
}
CHEAP = ["val_action_l1", "open_loop_smoothness", "plv1_smoothness", "replan_overlap_h16",
         "boundary_jump_h16", "val_loss", "train_loss", "support_drift",
         "dynamics_uncertainty", "teacher_forced_latent_mse"]
SMALL = ["epoch", "val_action_l1", "open_loop_smoothness", "replan_overlap_h16", "val_loss"]
VLM2 = ["vlm_cosmos_v6", "vlm_nemotron_v2"]
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
Z2 = ROOT / "results/hackathon/explore"


# ------------------------------------------------------------------ data (independent parser)
def load():
    ck = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv").sort_values(["run_id", "epoch"]).reset_index(drop=True)
    Y = np.full((len(ck), 50), np.nan)
    for run, d in RUN_DIRS.items():
        rows = {}
        for line in open(ROOT / d / "logs.json.txt"):
            if "test/sim_max_reward_100000" not in line:
                continue
            r = json.loads(line)
            e = int(r["epoch"])
            assert e not in rows, (run, e)          # exactly one eval row per epoch
            rows[e] = r
        for i in np.where(ck.run_id.values == run)[0]:
            r = rows[int(ck.epoch[i])]
            v = np.array([float(r[f"test/sim_max_reward_{s}"]) for s in range(100000, 100050)])
            assert set(np.unique(v)) <= {0.0, 1.0}, (run, ck.epoch[i], np.unique(v))
            Y[i] = v
    assert np.abs(Y.mean(1) - ck.success.values).max() < 1e-12
    E = np.full((len(ck), 10), np.nan)
    pos = {c: i for i, c in enumerate(ck.checkpoint_id)}
    for c, i in pos.items():
        p = ROOT / "data/outputs/hackathon_rollouts_eval" / c / "summary.json"
        if p.is_file():
            for ep in json.loads(p.read_text())["episodes"]:
                E[i, int(ep["env_seed"]) - 200000] = float(bool(ep["success"]))
    V = {}
    for j, f in JUDGES.items():
        A = np.full((len(ck), 10), np.nan)
        n = 0
        for line in open(ROOT / f):
            r = json.loads(line)
            i, k = pos[r["checkpoint_id"]], int(r["env_seed"]) - 200000
            assert r["scored"] and float(bool(r["truth_success"])) == E[i, k]
            A[i, k] = float(bool(r["cosmos_task_success"]))
            n += 1
        assert n == 330
        V[j] = A
    has = np.isfinite(E).all(1)
    assert has.sum() == 33 and all(np.isfinite(V[j][has]).all() for j in V)
    return ck, Y, E, V, has


# ------------------------------------------------------------------ statistics
def rho_tb(score, y):
    """Expected Spearman under random tie-breaking of the score (z2's estimand)."""
    n = len(score)
    a = stats.rankdata(score) - (n + 1) / 2.0
    b = stats.rankdata(y) - (n + 1) / 2.0
    syy = float(b @ b)
    if syy == 0:
        return np.nan
    return float(a @ b) / math.sqrt(n * (n * n - 1) / 12.0 * syy)


def rank01(x):
    return (stats.rankdata(x) - 1.0) / (len(x) - 1.0)


# ------------------------------------------------------------------ features
def features(ck, E, V, rows):
    raw = {"epoch": ck.epoch.values.astype(float)}
    for m in CHEAP:
        raw[m] = -np.log(ck[m].values.astype(float))
    with np.errstate(all="ignore"):
        raw["k10_sim"] = np.nanmean(E, 1)
    for j, a in V.items():
        raw[j] = a.mean(1)
    runs = ck.run_id.values
    Z, R = {}, {}
    for f, x in raw.items():
        z = np.full(len(x), np.nan)
        r = np.full(len(x), np.nan)
        for run in RUNS:
            idx = np.where(rows & (runs == run))[0]
            v = x[idx]
            sd = v.std()
            z[idx] = (v - v.mean()) / sd if sd > 0 else 0.0
            r[idx] = rank01(v)
        Z[f], R[f] = z, r
    return {"z": Z, "rank": R}


# ------------------------------------------------------------------ learners (own implementation)
def ridge_fit(X, y, runs, lam):
    ur = np.unique(runs)
    p = X.shape[1]
    A, b = np.zeros((p, p)), np.zeros(p)
    for r in ur:
        m = runs == r
        Xi = X[m] - X[m].mean(0)
        t = rank01(y[m])
        t = t - t.mean()
        A += Xi.T @ Xi / m.sum() / len(ur)
        b += Xi.T @ t / m.sum() / len(ur)
    return np.linalg.solve(A + lam * np.eye(p), b)


def _pairs(X, y, runs):
    D, W = [], []
    for r in np.unique(runs):
        idx = np.where(runs == r)[0]
        i, j = np.triu_indices(len(idx), 1)
        dy = y[idx[i]] - y[idx[j]]
        k = dy != 0
        if k.any():
            D.append((X[idx[i[k]]] - X[idx[j[k]]]) * np.sign(dy[k])[:, None])
            W.append(np.abs(dy[k]) / np.abs(dy[k]).sum())
    if not D:
        return None, None
    return np.vstack(D), np.concatenate(W) / len(D)


def pairlogit_fit(X, y, runs, lam):
    """min sum W*log(1+exp(-Dw)) + lam/2 |w|^2, damped Newton with backtracking."""
    D, W = _pairs(X, y, runs)
    p = X.shape[1]
    w = np.zeros(p)
    if D is None:
        return w

    def f(w_):
        return float(W @ np.logaddexp(0.0, -(D @ w_)) + 0.5 * lam * w_ @ w_)
    fv = f(w)
    for _ in range(200):
        z = D @ w
        s = 1.0 / (1.0 + np.exp(z))             # sigmoid(-z)
        g = -(D.T @ (W * s)) + lam * w
        H = (D * (W * s * (1 - s))[:, None]).T @ D + lam * np.eye(p)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            wn = w - t * step
            fn = f(wn)
            if fn <= fv - 1e-4 * t * (g @ step) or t < 1e-8:
                break
            t *= 0.5
        w, fv = wn, fn
        if np.abs(t * step).max() < 1e-11:
            break
    return w


FIT = {"ridge": ridge_fit, "pairlogit": pairlogit_fit}


def nested(kind, X, y, runs):
    tr = np.unique(runs)
    inner = []
    for lam in LAMS:
        rh = []
        for h in tr:
            a = runs != h
            w = FIT[kind](X[a], y[a], runs[a], lam)
            rh.append(rho_tb(X[runs == h] @ w, y[runs == h]))
        rh = np.array(rh, float)
        inner.append(np.nanmean(rh) if np.isfinite(rh).any() else -np.inf)
    inner = np.array(inner)
    k = max(i for i in range(len(LAMS)) if inner[i] >= inner.max() - 1e-12)
    return FIT[kind](X, y, runs, LAMS[k]), LAMS[k]


# ------------------------------------------------------------------ methods
def specs(pool, full=True):
    S = {"epoch": ("free", ["epoch"])}
    if full:
        for m in CHEAP:
            S[m] = ("free", [m])
            S[f"RA(epoch,{m})"] = ("free", ["epoch", m])
        S["RA(all cheap, no epoch)"] = ("free", list(CHEAP))
        S["RA(epoch + all cheap)"] = ("free", ["epoch"] + CHEAP)
    else:
        for m in ["val_action_l1", "train_loss", "replan_overlap_h16"]:
            S[m] = ("free", [m])
        S["RA(epoch,val_action_l1)"] = ("free", ["epoch", "val_action_l1"])
    all_f = ["epoch"] + CHEAP
    sets = {"all": all_f, "noepoch": list(CHEAP), "small": SMALL}
    if pool == "eval33":
        vl = ["k10_sim"] + list(JUDGES) if full else ["vlm_nemotron_v2", "vlm_cosmos_v6"]
        for j in vl:
            S[j] = ("free", [j])
            S[f"RA(epoch,{j})"] = ("free", ["epoch", j])
        if full:
            S["RA(epoch,val_action_l1,vlm_nemotron_v2)"] = ("free", ["epoch", "val_action_l1", "vlm_nemotron_v2"])
        sets = {"all": all_f, "small": SMALL, "all+vlm": all_f + VLM2, "epoch+vlm": ["epoch"] + VLM2,
                "small+vlm": SMALL + VLM2}
    if not full:
        sets = {k: v for k, v in sets.items() if k in ("all", "small", "all+vlm", "small+vlm")}
    for sn, fs in sets.items():
        S[f"pairlogit_z[{sn}]"] = ("pairlogit", fs, "z")
        S[f"pairlogit_rank[{sn}]"] = ("pairlogit", fs, "rank")
        S[f"ridge_rank[{sn}]"] = ("ridge", fs, "rank")
        S[f"unitweight[{sn}]"] = ("unit", fs)
    if full:
        S["select_best_on_train"] = ("select", [k for k, v in S.items() if v[0] == "free"])
    return S


def fold(S, NR, runs, yfit, ytgt, idx, hold, record=None):
    tr = np.concatenate([idx[r] for r in RUNS if r != hold])
    te = idx[hold]
    rtr = runs[tr]
    out = {}
    for name, sp in S.items():
        k = sp[0]
        if k == "free":
            sc = np.mean([NR["rank"][f][te] for f in sp[1]], 0)
        elif k in FIT:
            Xtr = np.column_stack([NR[sp[2]][f][tr] for f in sp[1]])
            Xte = np.column_stack([NR[sp[2]][f][te] for f in sp[1]])
            w, lam = nested(k, Xtr, yfit[tr], rtr)
            sc = Xte @ w
            if record is not None:
                record[(hold, name)] = {"lam": lam, "w": w.tolist(), "score_te": sc.tolist()}
        elif k == "unit":
            sg = []
            for f in sp[1]:
                rr = np.nanmean([rho_tb(NR["rank"][f][tr][rtr == r], yfit[tr][rtr == r]) for r in np.unique(rtr)])
                sg.append(-1.0 if (np.isfinite(rr) and rr < 0) else 1.0)
            sc = np.sum([s * NR["rank"][f][te] for s, f in zip(sg, sp[1])], 0)
        elif k == "select":
            best, bn = -np.inf, "epoch"
            for c in sp[1]:
                fs = S[c][1]
                s_tr = np.mean([NR["rank"][f][tr] for f in fs], 0)
                rr = np.nanmean([rho_tb(s_tr[rtr == r], yfit[tr][rtr == r]) for r in np.unique(rtr)])
                if np.isfinite(rr) and rr > best + 1e-12:
                    best, bn = rr, c
            sc = np.mean([NR["rank"][f][te] for f in S[bn][1]], 0)
            if record is not None:
                record[(hold, name)] = {"chosen": bn}
        out[name] = rho_tb(sc, ytgt[te])
    return out


def pool_rows(pool, ck, has):
    if pool == "eval33":
        return has.copy()
    if pool == "late93":
        return ck.epoch.values >= ck.groupby("run_id").epoch.transform("median").values
    return np.ones(len(ck), bool)


def loro(S, NR, runs, yfit, ytgt, idx, record=None):
    per = {h: fold(S, NR, runs, yfit, ytgt, idx, h, record) for h in RUNS}
    return {n: [per[h][n] for h in RUNS] for n in S}


# ------------------------------------------------------------------ V2 worker
_C: dict = {}


def _split_job(b):
    c = _C
    rng = np.random.default_rng([c["seed"], b])
    perm = rng.permutation(50)
    fa, fb = perm[:25], perm[25:]
    Y = c["Y"]
    res = {}
    for pool, (S, NR, idx) in c["pools"].items():
        res[pool] = loro(S, NR, c["runs"], Y[:, fa].mean(1), Y[:, fb].mean(1), idx)
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7031)
    ap.add_argument("--out", default=str(Z2 / "z2_loro_hybrid_verification.json"))
    a = ap.parse_args()
    t0 = time.time()
    ck, Y, E, V, has = load()
    runs = ck.run_id.values
    yA, yB, y50 = Y[:, :25].mean(1), Y[:, 25:].mean(1), Y.mean(1)
    out = {"name": "z2_loro_hybrid_verification", "label": "EXPLORATORY / post-hoc adversarial verification"}

    # ---------------- V1: point estimates, fixed split + swap, all z2 methods
    z2 = {p: json.loads((Z2 / f).read_text())["pools"][p] for f, p in
          [("z2_loro_hybrid.json", "all93"), ("z2_loro_hybrid.json", "eval33"), ("z2_loro_hybrid_late.json", "late93")]}
    V1, rec_all, POOLS = {}, {}, {}
    for pool in ("all93", "eval33", "late93"):
        rows = pool_rows(pool, ck, has)
        NR = features(ck, E, V, rows)
        idx = {r: np.where(rows & (runs == r))[0] for r in RUNS}
        S = specs(pool, full=True)
        rec = {}
        pt = loro(S, NR, runs, yA, yB, idx, rec)
        sw = loro(S, NR, runs, yB, yA, idx)
        diffs, sdiff = {}, {}
        for n in S:
            if n in z2[pool]["methods"]:
                theirs = [z2[pool]["methods"][n]["per_run"][h]["spearman"] for h in RUNS]
                diffs[n] = float(np.nanmax(np.abs(np.array(pt[n]) - np.array(theirs))))
                sdiff[n] = float(abs(np.mean(sw[n]) - z2[pool]["methods"][n]["swap_fit_HB_eval_HA"]["spearman"]))
        V1[pool] = {"heldout_mean": {n: float(np.mean(v)) for n, v in pt.items()},
                    "heldout_per_run": pt, "swap_mean": {n: float(np.mean(v)) for n, v in sw.items()},
                    "max_abs_diff_vs_z2_per_run_rho": diffs, "abs_diff_vs_z2_swap": sdiff,
                    "n_methods_compared": len(diffs),
                    "n_methods_diff_gt_1e-6": int(sum(v > 1e-6 for v in diffs.values())),
                    "lam_chosen": {f"{h}|{n}": r["lam"] for (h, n), r in rec.items() if "lam" in r},
                    "select_chosen": {h: r["chosen"] for (h, n), r in rec.items() if "chosen" in r}}
        rec_all[pool] = rec
        POOLS[pool] = (specs(pool, full=False), NR, idx)
        # V3: LORO with all 50 episodes for fit (training runs) and target (held-out run)
        p50 = loro(S, NR, runs, y50, y50, idx)
        V1[pool]["loro50_mean"] = {n: float(np.mean(v)) for n, v in p50.items()}
        V1[pool]["loro50_per_run"] = p50
        print(f"[V1/V3] {pool} done {time.time() - t0:.0f}s; max|diff| vs z2 = {max(diffs.values()):.2e}", flush=True)
    out["V1_V3_point"] = V1

    # identical ranking check: eval33 pairlogit_z[all] vs epoch
    ident = {}
    for h in RUNS:
        sc = np.array(rec_all["eval33"][(h, "pairlogit_z[all]")]["score_te"])
        ep = ck.epoch.values[POOLS["eval33"][2][h]]
        ident[h] = bool((np.argsort(sc, kind="stable") == np.argsort(ep, kind="stable")).all())
    out["eval33_pairlogit_z_all_same_order_as_epoch"] = ident

    # ---------------- V2: repeated random splits
    _C.clear()
    _C.update(dict(seed=a.seed, Y=Y, runs=runs, pools=POOLS))
    t1 = time.time()
    with mp.get_context("fork").Pool(a.workers) as P:
        res = list(P.imap(_split_job, range(a.n_splits), chunksize=4))
    print(f"[V2] {a.n_splits} splits in {time.time() - t1:.0f}s", flush=True)
    V2 = {}
    for pool, (S, _, _) in POOLS.items():
        ep_run = np.array([r[pool]["epoch"] for r in res], float)
        ok_run = np.isfinite(ep_run)                       # a run is undefined when its target half is constant
        ep = np.array([np.mean(v[k]) for v, k in zip(ep_run, ok_run)])
        d = {}
        for n in S:
            mr = np.array([r[pool][n] for r in res], float)
            m = np.array([np.mean(v[k]) for v, k in zip(mr, ok_run)])   # mean over runs defined in that split
            dd = m - ep
            orig = V1[pool]["heldout_mean"][n] - V1[pool]["heldout_mean"]["epoch"]
            d[n] = {"mean_rho": float(m.mean()), "mean_delta": float(dd.mean()), "median_delta": float(np.median(dd)),
                    "frac_delta_gt0": float((dd > 1e-12).mean()), "frac_delta_lt0": float((dd < -1e-12).mean()),
                    "delta_q025_q975": [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))],
                    "mean_delta_per_run": dict(zip(RUNS, np.nanmean(mr - ep_run, 0).round(5).tolist())),
                    "frac_splits_beats_epoch_in_all3_runs": float(((mr - ep_run) > 1e-12).all(1).mean()),
                    "original_split_delta": float(orig),
                    "original_split_percentile": float((dd < orig).mean())}
        V2[pool] = {"n_splits_with_undefined_run": int((~ok_run).any(1).sum()),
                    "original_split_epoch_rho_percentile": float((ep < V1[pool]["heldout_mean"]["epoch"]).mean()),
                    "epoch_mean_rho": float(ep.mean()), "epoch_rho_q025_q975": [float(np.quantile(ep, .025)), float(np.quantile(ep, .975))],
                    "methods": d}
    out["V2_random_splits"] = {"n_splits": a.n_splits, "seed": a.seed, "pools": V2}

    # ---------------- V4: free scores vs the full 50-episode target, seed-cluster bootstrap (no refit)
    rng = np.random.default_rng(a.seed + 1)
    V4 = {}
    free_list = ["epoch", "val_action_l1", "RA(epoch,val_action_l1)", "train_loss", "replan_overlap_h16"]
    for pool in ("all93", "eval33", "late93"):
        rows = pool_rows(pool, ck, has)
        NR = POOLS[pool][1]
        idx = POOLS[pool][2]
        fs = {"epoch": ["epoch"], "val_action_l1": ["val_action_l1"], "RA(epoch,val_action_l1)": ["epoch", "val_action_l1"],
              "train_loss": ["train_loss"], "replan_overlap_h16": ["replan_overlap_h16"]}
        if pool == "eval33":
            fs.update({"vlm_nemotron_v2": ["vlm_nemotron_v2"], "RA(epoch,vlm_nemotron_v2)": ["epoch", "vlm_nemotron_v2"]})
        scores = {n: {h: np.mean([NR["rank"][f][idx[h]] for f in v], 0) for h in RUNS} for n, v in fs.items()}

        def allrho(y):
            return {n: np.array([rho_tb(scores[n][h], y[idx[h]]) for h in RUNS]) for n in fs}
        pt = allrho(y50)
        bs = {n: [] for n in fs}
        for _ in range(a.n_boot):
            s = rng.integers(0, 50, 50)
            r_ = allrho(Y[:, s].mean(1))
            for n in fs:
                bs[n].append(np.nanmean(r_[n]))
        e_b = np.array(bs["epoch"])
        V4[pool] = {}
        for n in fs:
            b_ = np.array(bs[n])
            dd = b_ - e_b
            d0 = pt[n].mean() - pt["epoch"].mean()
            V4[pool][n] = {"rho50_mean": float(pt[n].mean()), "rho50_per_run": dict(zip(RUNS, pt[n].round(4).tolist())),
                           "delta_vs_epoch": float(d0),
                           "delta_per_run": dict(zip(RUNS, (pt[n] - pt["epoch"]).round(4).tolist())),
                           "seed_boot_delta_ci": [float(np.quantile(dd, .025)), float(np.quantile(dd, .975))],
                           "seed_boot_p_delta_gt0": float((dd > 1e-12).mean()),
                           "seed_boot_rho_ci": [float(np.quantile(b_, .025)), float(np.quantile(b_, .975))],
                           "frac_boot_delta_below_point": float((dd < d0).mean())}
    out["V4_free_scores_full50"] = {"n_boot": a.n_boot, "pools": V4}

    # ---------------- V5: bootstrap-bias diagnostics from z2 JSONs + moving-block edge effect
    V5 = {}
    for tag, f, plist in [("block", "z2_loro_hybrid.json", ["all93", "eval33"]),
                          ("episode_only", "z2_loro_hybrid_episode_boot.json", ["all93", "eval33"]),
                          ("block_late", "z2_loro_hybrid_late.json", ["late93"])]:
        J = json.loads((Z2 / f).read_text())["pools"]
        for p in plist:
            M = J[p]["methods"]
            rows_ = {}
            outside = 0
            nonref = [n for n in M if not n.startswith(("random", "oracle"))]
            for n in nonref:
                pt_ = M[n]["heldout_HB"]["spearman"]
                ci = M[n]["block_boot"]["spearman"]["ci"]
                dq = M[n]["block_boot"]["spearman"]["delta_quantiles_10_25_50_75_90"]
                d0 = M[n]["run_level"]["spearman"]["delta_vs_epoch"]
                out_ = not (ci[0] <= pt_ <= ci[1])
                outside += out_
                rows_[n] = {"point_rho": pt_, "boot_rho_ci": ci, "point_outside_boot_ci": out_,
                            "point_delta": d0, "boot_delta_median": dq[2], "boot_delta_q90": dq[4],
                            "point_delta_above_boot_q90": bool(d0 > dq[4])}
            V5[f"{tag}|{p}"] = {"n_methods": len(nonref), "n_point_rho_outside_boot_ci": outside,
                                "n_learned_point_delta_above_q90": int(sum(r["point_delta_above_boot_q90"] for n, r in rows_.items()
                                                                           if M[n]["kind"] in ("learn", "unit", "select"))),
                                "n_learned": int(sum(M[n]["kind"] in ("learn", "unit", "select") for n in nonref)),
                                "methods": {n: rows_[n] for n in rows_ if n in (
                                    "epoch", "pairlogit_rank[all]", "ridge_rank[all]", "pairlogit_z[small]", "ridge_rank[small]",
                                    "RA(epoch,val_action_l1)", "val_action_l1", "pairlogit_z[all]", "pairlogit_z[all+vlm]")}}
    edge = {}
    for r, (n, L) in {"can_seed1": (41, 8), "can_seed2": (21, 4), "square_seed1": (31, 4)}.items():
        nb = math.ceil(n / L)
        # expected copies of position i: nb blocks, start uniform on [0, n-L]; the final block is truncated
        cnt = np.zeros(n)
        rr = np.random.default_rng(1)
        for _ in range(20000):
            st = rr.integers(0, n - L + 1, nb)
            sel = np.concatenate([np.arange(s, s + L) for s in st])[:n]
            cnt += np.bincount(sel, minlength=n)
        cnt /= 20000
        edge[r] = {"expected_copies_first": float(cnt[0]), "expected_copies_last": float(cnt[-1]),
                   "expected_copies_middle": float(cnt[n // 2]),
                   "expected_copies_first_quarter_mean": float(cnt[: n // 4].mean()),
                   "expected_copies_last_quarter_mean": float(cnt[-(n // 4):].mean())}
    V5["moving_block_edge_effect"] = edge
    out["V5_bootstrap_bias"] = V5

    # ---------------- V6: claim checks
    M33 = z2["eval33"]["methods"]
    pairs = {}
    for base in ("pairlogit_z", "pairlogit_rank", "ridge_rank", "unitweight"):
        for s in ("all", "small"):
            a_, b_ = M33[f"{base}[{s}+vlm]"]["heldout_HB"]["spearman"], M33[f"{base}[{s}]"]["heldout_HB"]["spearman"]
            pairs[f"{base}[{s}+vlm] vs [{s}]"] = {"with_vlm": a_, "without": b_, "diff": a_ - b_}
    vlm_models = {n: M33[n]["heldout_HB"]["spearman"] for n in M33 if "vlm]" in n}
    M93l = z2["late93"]["methods"]
    late_learn = {n: M93l[n]["heldout_HB"]["spearman"] for n in M93l if M93l[n]["kind"] in ("learn",)}
    late_unit = {n: M93l[n]["heldout_HB"]["spearman"] for n in M93l if M93l[n]["kind"] in ("unit",)}
    power = {}
    for n in ("pairlogit_rank[all]", "ridge_rank[all]", "pairlogit_z[small]", "ridge_rank[small]"):
        dv = np.array(list(z2["all93"]["methods"][n]["run_level"]["spearman"]["delta_per_run"].values()))
        m, sd = dv.mean(), dv.std(ddof=1)
        eff = m / sd
        n50 = n80 = None
        for k in range(3, 400):
            tcrit = stats.t.ppf(0.975, k - 1)
            nc = eff * math.sqrt(k)
            pw = 1 - stats.nct.cdf(tcrit, k - 1, nc) + stats.nct.cdf(-tcrit, k - 1, nc)
            if n50 is None and pw >= 0.5:
                n50 = k
            if n80 is None and pw >= 0.8:
                n80 = k
                break
        power[n] = {"fold_deltas": dv.round(5).tolist(), "mean": float(m), "sd": float(sd), "d_z": float(eff),
                    "runs_for_50pct_power_two_sided_0.05": n50, "runs_for_80pct_power_two_sided_0.05": n80,
                    "one_sample_t_p_two_sided_n3": float(stats.ttest_1samp(dv, 0).pvalue)}
    out["V6_claims"] = {"eval33_vlm_vs_vlmfree_pairs": pairs,
                        "eval33_vlm_model_range": [min(vlm_models.values()), max(vlm_models.values())],
                        "eval33_vlm_models": vlm_models,
                        "late93_learned_range": [min(late_learn.values()), max(late_learn.values())],
                        "late93_unitweight_range": [min(late_unit.values()), max(late_unit.values())],
                        "power": power,
                        "coverage_of_min_max_of_3_for_population_median": 1 - 2 * 0.5 ** 3,
                        "sign_test_3of3_one_sided_p": 0.5 ** 3}
    out["runtime_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {a.out} in {out['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
