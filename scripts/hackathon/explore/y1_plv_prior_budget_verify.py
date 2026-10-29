#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- adversarial verification of y1_plv_prior_budget.

Independent re-implementation (does NOT import y1 or x1):
  * own parser of data/outputs/hackathon_pool/<run>/logs.json.txt (per-episode outcomes)
  * own top-30 % prior subsets from results/hackathon/baseline_metrics.csv (outcome-free columns)
  * own uniform-allocation and top-two-Thompson (TTTS) simulators, perfect judge only
  * own beta-binomial judge-error fits from the final eval JSONLs
  * per-run and leave-one-run-out (LORO) summaries, per-run seed bootstrap of paired deltas
  * threshold / grid-floor sensitivity of the "budget ratio" metric
  * claim checks against results/hackathon/explore/y1_plv_prior_budget*.json

Split: pool = seeds 100000-100024, held-out = 100025-100049 (as y1); random re-splits as sensitivity.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y1_plv_prior_budget_verify.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import betaln, expit  # noqa: E402
from scipy.stats import chi2, rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("VERIFY_OUT", str(ROOT / "results/hackathon/explore/y1_plv_prior_budget_verification.json")))
Y1 = ROOT / "results/hackathon/explore/y1_plv_prior_budget.json"
Y1S = ROOT / "results/hackathon/explore/y1_plv_prior_budget_splits.json"
RERUN = ROOT / "data/outputs/hackathon_explore/y1_plv_prior_budget_verification/rerun_main.json"
RERUN_S = ROOT / "data/outputs/hackathon_explore/y1_plv_prior_budget_verification/rerun_splits.json"
MULTS = (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, 25)
SUMMARY = (0.5, 1, 1.5, 2, 3, 4, 6, 8)
TRUNC = {"can": (60, 100, 150), "square": (90, 150, 240)}
SINGLE = {"val_action_l1": True, "open_loop_smoothness": True, "plv1_smoothness": True,
          "replan_overlap_h16": True, "boundary_jump_h16": True, "val_loss_logged": True, "epoch": False}
COMBOS = {"epoch+replan_overlap_h16": ("epoch", "replan_overlap_h16"),
          "epoch+open_loop_smoothness": ("epoch", "open_loop_smoothness"),
          "epoch+val_action_l1": ("epoch", "val_action_l1"),
          "epoch+boundary_jump_h16": ("epoch", "boundary_jump_h16"),
          "epoch+plv1_smoothness": ("epoch", "plv1_smoothness"),
          "val_action_l1+replan_overlap_h16": ("val_action_l1", "replan_overlap_h16")}
SIM_PRIORS = ["none", "random", "epoch", "replan_overlap_h16", "val_action_l1", "val_loss_logged"]


# ----------------------------------------------------------------------------- data (own parser)
def load():
    df = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv")
    runs = []
    for run_id, g in df.groupby("run_id", sort=True):
        g = g.sort_values("epoch")
        run_dir = ROOT / Path(g["checkpoint_path"].iloc[0]).parents[1]
        rows = {}
        for line in (run_dir / "logs.json.txt").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in r:
                rows.setdefault(int(r["epoch"]), {}).update(r)
        Y = []
        for ep in g["epoch"].astype(int):
            r = rows[ep]
            Y.append([float(r[f"test/sim_max_reward_{s}"]) > 0.5 for s in range(100000, 100050)])
        Y = np.array(Y, float)
        m = {c: g[c].astype(float).values for c in SINGLE if c != "epoch"}
        m["epoch"] = g["epoch"].astype(float).values
        runs.append({"run": run_id, "task": g["task"].iloc[0], "epochs": g["epoch"].astype(int).values, "Y": Y,
                     "metrics": m, "success_csv": g["success"].astype(float).values})
    return runs


def order(metrics, epochs, prior):
    def key(name):
        v = np.asarray(metrics[name], float)
        return v if SINGLE[name] else -v
    if prior in SINGLE:
        k = key(prior)
    else:
        k = np.mean([rankdata(key(a)) for a in COMBOS[prior]], axis=0)
    return sorted(range(len(epochs)), key=lambda i: (k[i], -epochs[i]))


def problems(runs):
    out = []
    for lvl in ("full", "trunc_1", "trunc_2", "trunc_3"):
        for ri, r in enumerate(runs):
            if lvl == "full":
                idx = np.arange(len(r["epochs"]))
            else:
                idx = np.flatnonzero(r["epochs"] <= TRUNC[r["task"]][int(lvl[-1]) - 1])
            out.append({"level": lvl, "ri": ri, "run": r["run"], "idx": idx})
    return out


# ----------------------------------------------------------------------------- simulators (own code)
def sim_uniform(Yobs, arms, budgets, rng, jitter):
    """Yobs [R, n, P] observed outcomes (already in per-rep order); arms [R, k]."""
    R, n, P = Yobs.shape
    k = arms.shape[1]
    cs = np.concatenate([np.zeros((R, n, 1)), np.cumsum(Yobs, 2)], 2)
    cs = np.take_along_axis(cs, arms[:, :, None], 1)
    jit = np.take_along_axis(jitter, arms, 1)
    out = {}
    pri = rng.random((R, k)).argsort(1).argsort(1)
    for B in budgets:
        nn = np.minimum(B // k + (pri < B % k), P)
        s = np.take_along_axis(cs, nn[:, :, None], 2)[:, :, 0]
        out[B] = np.take_along_axis(arms, ((s + 1) / (nn + 2) + jit).argmax(1)[:, None], 1)[:, 0]
    return out


def sim_ttts(Yobs, arms, budgets, rng, jitter, beta=0.5):
    R, n, P = Yobs.shape
    k = arms.shape[1]
    Ya = np.take_along_axis(Yobs, arms[:, :, None], 1)  # [R, k, P]
    jit = np.take_along_axis(jitter, arms, 1)
    nn = np.zeros((R, k), int)
    ss = np.zeros((R, k))
    ar = np.arange(R)
    out = {}
    bs = sorted(set(budgets))
    for t in range(1, bs[-1] + 1):
        avail = nn < P
        th1 = rng.beta(1 + ss, 1 + nn - ss)
        lead = np.where(avail, th1, -np.inf).argmax(1)
        th2 = rng.beta(1 + ss, 1 + nn - ss)
        th2[ar, lead] = -np.inf
        th2 = np.where(avail, th2, -np.inf)
        chal = th2.argmax(1)
        has_chal = np.isfinite(th2[ar, chal])
        a = np.where((rng.random(R) < beta) | ~has_chal, lead, chal)
        ok = avail[ar, a]
        y = Ya[ar, a, np.minimum(nn[ar, a], P - 1)]
        nn[ar[ok], a[ok]] += 1
        ss[ar[ok], a[ok]] += y[ok]
        if t in budgets:
            out[t] = np.take_along_axis(arms, ((ss + 1) / (nn + 2) + jit).argmax(1)[:, None], 1)[:, 0]
    return out


def simulate(r, idx, pool, ho, R, seed, strategies=("uniform", "top_two_thompson"), mults=MULTS):
    seed = seed if isinstance(seed, int) else hash_s(str(seed))
    Y = r["Y"][idx]
    n = len(idx)
    ep = r["epochs"][idx]
    mets = {c: np.asarray(v)[idx] for c, v in r["metrics"].items()}
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((R, len(pool))), 1)
    Yp = Y[:, pool]
    Yobs = np.stack([Yp[:, perm[i]] for i in range(R)])  # [R, n, P], shared order across checkpoints
    H = Y[:, ho].mean(1)
    best = H.max()
    k30 = math.ceil(0.3 * n)
    jitter = rng.random((R, n)) * 1e-9
    budgets = [max(1, int(round(m * n))) for m in mults]
    arms = {"none": np.tile(np.arange(n), (R, 1)), "random": np.argsort(rng.random((R, n)), 1)[:, :k30]}
    for p in SIM_PRIORS[2:]:
        arms[p] = np.tile(np.array(order(mets, ep, p)[:k30]), (R, 1))
    res = {}
    for p, a in arms.items():
        for st in strategies:
            srng = np.random.default_rng([seed, hash_s(p + st)])
            f = sim_uniform if st == "uniform" else sim_ttts
            picks = f(Yobs, a, budgets, srng, jitter)
            res[(p, st)] = np.array([best - H[picks[B]] for B in budgets])  # [M, R]
    return res


def hash_s(s):
    import zlib
    return zlib.crc32(s.encode())


# ----------------------------------------------------------------------------- judge fits (own code)
def judge_fit(stem):
    d = {}
    for line in (ROOT / "data/outputs/hackathon_cosmos" / stem).read_text().splitlines():
        r = json.loads(line)
        if not (r.get("scored") and r.get("parse_status") == "ok"):
            continue
        if not isinstance(r.get("truth_success"), bool) or not isinstance(r.get("cosmos_task_success"), bool):
            continue
        c = d.setdefault(r["checkpoint_id"], [r["task"], 0, 0, 0])  # task, n_fail, fp, n
        c[3] += 1
        if not r["truth_success"]:
            c[1] += 1
            c[2] += int(r["cosmos_task_success"])
    tasks = ["can", "square"]
    data = [(tasks.index(t), nf, fp, 1 - nf / n) for t, nf, fp, n in d.values() if nf > 0]

    def nll(th, quality):
        k = math.exp(th[-1])
        s = 0.0
        for t, nf, fp, se in data:
            mu = expit(th[t] + (th[2] * (se - 0.5) if quality else 0.0))
            mu = min(max(mu, 1e-9), 1 - 1e-9)
            s -= betaln(fp + mu * k, nf - fp + (1 - mu) * k) - betaln(mu * k, (1 - mu) * k)
        return s

    out = {}
    for quality in (False, True):
        npar = 3 if quality else 2
        best_free = min((minimize(lambda th: nll(th, quality), np.r_[[-1.5] * 2, [2.0] * (npar - 2), lk],
                                  method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-11})
                         for lk in (-1.0, 0.0, 1.0, 2.0, 4.0)), key=lambda r: r.fun)
        lk_hi = math.log(1e4)
        binom = minimize(lambda t: nll(np.r_[t, lk_hi], quality), best_free.x[:-1], method="Nelder-Mead",
                         options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-11})
        lr = 2 * (binom.fun - min(best_free.fun, binom.fun))
        out["bbq" if quality else "bb"] = {
            "kappa_mle": float(math.exp(min(best_free.x[-1], lk_hi))), "theta_mle": best_free.x.tolist(),
            "nll": float(min(best_free.fun, binom.fun)), "nll_binomial": float(binom.fun),
            "lr_vs_binomial": float(lr), "p_boundary_mixture": float(0.5 * chi2.sf(lr, 1)) if lr > 0 else 0.5,
            "binomial_theta": binom.x.tolist()}
    # binomial (kappa -> inf) slope test: is the quality slope significant without overdispersion?
    th_b = out["bbq"]["binomial_theta"]
    out["bbq"]["slope_binomial"] = th_b[2]
    out["bb_vs_bbq_binomial_lr"] = float(2 * (out["bb"]["nll_binomial"] - out["bbq"]["nll_binomial"]))
    out["per_checkpoint"] = {k: f"{v[2]}/{v[1]} (n={v[3]})" for k, v in sorted(d.items()) if v[1] > 0}
    out["n_ckpt"] = len(d)
    out["n_fail"] = int(sum(v[1] for v in d.values()))
    out["fp_at_ckpts_with_le2_failures"] = [int(sum(v[2] for v in d.values() if 0 < v[1] <= 2)),
                                            int(sum(v[1] for v in d.values() if 0 < v[1] <= 2))]
    out["fp_at_ckpts_with_ge5_failures"] = [int(sum(v[2] for v in d.values() if v[1] >= 5)),
                                            int(sum(v[1] for v in d.values() if v[1] >= 5))]
    return out


G = {}


def boot_task(t):
    lvl, mode, bi = t
    pc, hc = (G["boots"] if mode == "bootstrap" else G["splits"])[bi]
    out = {}
    for pi, pr in enumerate(G["probs"]):
        if pr["level"] != lvl:
            continue
        s = simulate(G["runs"][pr["ri"]], pr["idx"], pc, hc, G["RB"], [50000 + bi, pi, int(mode == "split")],
                     strategies=("top_two_thompson",), mults=MULTS[:9])
        out[pi] = {}
        for key in G["KEYS"]:
            a_, b_ = key.split("-")
            d_ = s[(a_, "top_two_thompson")].mean(1) - s[(b_, "top_two_thompson")].mean(1)
            out[pi][key] = [float(d_[MULTS.index(1)]), float(d_[G["smi"]].mean())]
    return out


# ----------------------------------------------------------------------------- helpers
def pct(v, q=(2.5, 97.5)):
    return [round(float(x), 4) for x in np.percentile(v, q)]


def row(L, j, p, st):
    return [x for x in L["rows"] if x["judge"] == j and x["prior"] == p and x["strategy"] == st][0]


def pair(L, j, p, st):
    return [x for x in L["paired"] if x["judge"] == j and x["prior"] == p and x["strategy"] == st][0]


def main():
    t0 = time.time()
    V = {"name": "y1_plv_prior_budget_verification", "exploratory": True,
         "label": "EXPLORATORY / post-hoc adversarial verification; independent re-implementation, CPU only"}
    runs = load()
    probs = problems(runs)

    # 1. data alignment ----------------------------------------------------------------------
    V["data"] = {r["run"]: {"n_ckpt": len(r["epochs"]), "max_abs_diff_csv_success_vs_logs":
                            float(np.abs(r["success_csv"] - r["Y"].mean(1)).max())} for r in runs}
    df = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv")
    V["data"]["h10_equals_h16"] = {
        "boundary_jump": bool(np.allclose(df["boundary_jump_h10"], df["boundary_jump_h16"])),
        "replan_overlap": bool(np.allclose(df["replan_overlap_h10"], df["replan_overlap_h16"]))}
    V["data"]["spearman_replan_h16_vs_boundary_h16_within_run"] = {
        r["run"]: float(np.corrcoef(rankdata(r["metrics"]["replan_overlap_h16"]),
                                    rankdata(r["metrics"]["boundary_jump_h16"]))[0, 1]) for r in runs}

    # 2. subset overlap with epoch top-30 % ---------------------------------------------------
    ov = {}
    for lvl in ("full", "trunc_1", "trunc_2", "trunc_3"):
        ov[lvl] = {}
        for p in [*SINGLE, *COMBOS]:
            if p == "epoch":
                continue
            per, num, den = {}, 0, 0
            for pr in [q for q in probs if q["level"] == lvl]:
                r = runs[pr["ri"]]
                idx = pr["idx"]
                mets = {c: np.asarray(v)[idx] for c, v in r["metrics"].items()}
                ep = r["epochs"][idx]
                k30 = math.ceil(0.3 * len(idx))
                a = set(order(mets, ep, p)[:k30])
                b = set(order(mets, ep, "epoch")[:k30])
                per[r["run"]] = f"{len(a & b)}/{k30}"
                num += len(a & b)
                den += k30
            ov[lvl][p] = {"per_run": per, "pooled": round(num / den, 3),
                          "min_per_run": round(min(int(s.split('/')[0]) / int(s.split('/')[1]) for s in per.values()), 3)}
    V["subset_overlap_with_epoch_top30"] = ov

    # 3. held-out plateau + zero-rollout picks --------------------------------------------------
    hz = {}
    for pr in probs:
        r = runs[pr["ri"]]
        idx = pr["idx"]
        H = r["Y"][idx][:, 25:].mean(1)
        Pp = r["Y"][idx][:, :25].mean(1)
        ep = r["epochs"][idx]
        mets = {c: np.asarray(v)[idx] for c, v in r["metrics"].items()}
        z = {"heldout_best": float(H.max()), "n_tied_at_best": int((H == H.max()).sum()), "n": len(idx),
             "n_within_0.04_of_best": int((H >= H.max() - 0.04 - 1e-9).sum()),
             "pool_best_pick_regret": float(H.max() - H[np.flatnonzero(Pp == Pp.max())].mean()),
             "top1_regret": {p: float(H.max() - H[order(mets, ep, p)[0]]) for p in SINGLE}}
        hz.setdefault(pr["level"], {})[r["run"]] = z
    V["heldout_and_zero_rollout"] = hz

    # 4. independent simulation, original split, perfect judge -----------------------------------
    R0 = int(os.environ.get("VERIFY_R0", "400"))
    pool, ho = np.arange(25), np.arange(25, 50)
    sim0 = {}
    for pi, pr in enumerate(probs):
        sim0[pi] = simulate(runs[pr["ri"]], pr["idx"], pool, ho, R0, 1000 + pi)
    print(f"[{time.time() - t0:.0f}s] original-split sims done", flush=True)
    y1 = json.loads(Y1.read_text())
    smi = [MULTS.index(m) for m in SUMMARY]
    comp = {}
    for lvl in ("full", "trunc_1", "trunc_2", "trunc_3"):
        pis = [pi for pi, p in enumerate(probs) if p["level"] == lvl]
        L = y1["levels"][lvl]
        comp[lvl] = {}
        for st in ("uniform", "top_two_thompson"):
            for p in SIM_PRIORS:
                mine = np.mean([sim0[pi][(p, st)].mean(1) for pi in pis], 0)
                se = np.sqrt(np.mean([sim0[pi][(p, st)].var(1) / R0 for pi in pis], 0) / len(pis))
                theirs = np.array(row(L, "perfect", p, st)["mean_regret"])
                comp[lvl][f"{p}|{st}"] = {
                    "mine_1N": round(float(mine[MULTS.index(1)]), 4), "y1_1N": float(theirs[MULTS.index(1)]),
                    "mine_25N": round(float(mine[-1]), 4), "y1_25N": float(theirs[-1]),
                    "mine_avg": round(float(mine[smi].mean()), 4), "y1_avg": round(float(theirs[smi].mean()), 4),
                    "max_abs_diff_over_grid": round(float(np.abs(mine - theirs).max()), 4),
                    "approx_mc_se_1N": round(float(se[MULTS.index(1)]), 4)}
    V["independent_sim_vs_y1_point"] = comp

    # 5. per-run seed bootstrap of paired deltas (TTTS, perfect, full + trunc_1) and LORO -------
    NB, RB = int(os.environ.get("VERIFY_NB", "200")), 20
    rngb = np.random.default_rng(7)
    boots = [(rngb.integers(0, 25, 25), 25 + rngb.integers(0, 25, 25)) for _ in range(NB)]
    rngs = np.random.default_rng(8)
    splits = []
    for _ in range(NB):
        pm = rngs.permutation(50)
        splits.append((np.sort(pm[:25]), np.sort(pm[25:])))
    per_run = {}
    KEYS = ("epoch-none", "epoch-random", "val_action_l1-epoch", "replan_overlap_h16-epoch")
    G.update(runs=runs, probs=probs, boots=boots, splits=splits, RB=RB, smi=smi, KEYS=KEYS)
    tasks = [(lvl, mode, bi) for lvl in ("full", "trunc_1") for mode in ("bootstrap", "split") for bi in range(NB)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(int(os.environ.get("VERIFY_WORKERS", "10"))) as poolw:
        outs = poolw.map(boot_task, tasks, chunksize=4)
    for lvl in ("full", "trunc_1"):
        pis = [pi for pi, p in enumerate(probs) if p["level"] == lvl]
        per_run[lvl] = {}
        for mode in ("bootstrap", "split"):
            acc = {pi: {k: [] for k in KEYS} for pi in pis}
            for (lv_, md_, bi), o_ in zip(tasks, outs):
                if lv_ == lvl and md_ == mode:
                    for pi in pis:
                        for key in KEYS:
                            acc[pi][key].append(o_[pi][key])
            res = {}
            for key in KEYS:
                a_, b_ = key.split("-")
                pt = {}
                for pi in pis:
                    dd = sim0[pi][(a_, "top_two_thompson")].mean(1) - sim0[pi][(b_, "top_two_thompson")].mean(1)
                    pt[probs[pi]["run"]] = [round(float(dd[MULTS.index(1)]), 4), round(float(dd[smi].mean()), 4)]
                arr = {pi: np.array(acc[pi][key]) for pi in pis}  # [NB, 2]
                mean_all = np.mean([arr[pi] for pi in pis], 0)
                res[key] = {
                    "point_per_run_[1N, avg]": pt,
                    "per_run_interval_1N": {probs[pi]["run"]: pct(arr[pi][:, 0]) for pi in pis},
                    "per_run_interval_avg": {probs[pi]["run"]: pct(arr[pi][:, 1]) for pi in pis},
                    "per_run_share_lt0_avg": {probs[pi]["run"]: round(float((arr[pi][:, 1] < 0).mean()), 3) for pi in pis},
                    "per_run_share_gt0_avg": {probs[pi]["run"]: round(float((arr[pi][:, 1] > 0).mean()), 3) for pi in pis},
                    "mean_over_runs_interval_1N": pct(mean_all[:, 0]),
                    "mean_over_runs_interval_avg": pct(mean_all[:, 1]),
                    "mean_over_runs_share_lt0_avg": round(float((mean_all[:, 1] < 0).mean()), 3),
                    "mean_over_runs_median_avg": round(float(np.median(mean_all[:, 1])), 4),
                    "loro_interval_avg": {f"drop_{probs[pd_]['run']}": pct(np.mean([arr[pi][:, 1] for pi in pis if pi != pd_], 0))
                                          for pd_ in pis},
                }
            per_run[lvl][mode] = res
    print(f"[{time.time() - t0:.0f}s] per-run bootstrap done", flush=True)
    V["per_run_paired_TTTS_perfect"] = per_run

    # 6. judge fits -------------------------------------------------------------------------------
    V["judge_fits_independent"] = {"nemotron": judge_fit("eval_nebius_v2.jsonl"), "cosmos": judge_fit("eval_local_v6.jsonl")}

    # 7. claim checks on y1 JSONs + rerun reproducibility ----------------------------------------
    checks = {}
    for tag, path in (("bootstrap", Y1), ("split", Y1S)):
        d = json.loads(path.read_text())
        cc = {}
        for lvl, L in d["levels"].items():
            cells = [x for x in L["paired"] if "beats_both_avg_ci" in x]
            non_vll = [x for x in cells if x["prior"] != "val_loss_logged"]
            cc[lvl] = {
                "n_cells": len(cells),
                "beats_both_avg": sum(x["beats_both_avg_ci"] for x in cells),
                "beats_both_any_mult": sum(bool(x["beats_both_any_mult_ci"]) for x in cells),
                "beats_epoch_alone_avg_or_any_mult": sum(
                    (x["vs_epoch"]["avg_0.5N_8N"]["ci95"][1] < 0) or any(x["vs_epoch"]["by_mult"][m]["ci95"][1] < 0
                                                                          for m in ("0.5", "1", "2", "4", "8")) for x in cells),
                "worse_than_epoch_any": sorted({x["prior"] for x in cells if (x["vs_epoch"]["avg_0.5N_8N"]["ci95"][0] > 0) or any(
                    x["vs_epoch"]["by_mult"][m]["ci95"][0] > 0 for m in ("0.5", "1", "2", "4", "8"))}),
                "non_vll_vs_epoch_avg_point_range": [min(x["vs_epoch"]["avg_0.5N_8N"]["point"] for x in non_vll),
                                                     max(x["vs_epoch"]["avg_0.5N_8N"]["point"] for x in non_vll)],
                "non_vll_vs_epoch_avg_ci_envelope": [min(x["vs_epoch"]["avg_0.5N_8N"]["ci95"][0] for x in non_vll),
                                                     max(x["vs_epoch"]["avg_0.5N_8N"]["ci95"][1] for x in non_vll)],
                "non_vll_vs_epoch_1N_ci_envelope": [min(x["vs_epoch"]["by_mult"]["1"]["ci95"][0] for x in non_vll),
                                                    max(x["vs_epoch"]["by_mult"]["1"]["ci95"][1] for x in non_vll)],
            }
            # epoch vs none: how many judge x strategy cells have a CI excluding 0?
            ep = [x for x in L["paired"] if x["prior"] == "epoch"]
            cc[lvl]["epoch_vs_none_avg_ci_excludes0"] = f"{sum(x['vs_none']['avg_0.5N_8N']['ci95'][1] < 0 for x in ep)}/{len(ep)}"
            cc[lvl]["epoch_vs_none_1N_ci_excludes0"] = f"{sum(x['vs_none']['by_mult']['1']['ci95'][1] < 0 for x in ep)}/{len(ep)}"
            cc[lvl]["epoch_vs_none_perfect_ci_excludes0_by_strategy_avg"] = {
                x["strategy"]: x["vs_none"]["avg_0.5N_8N"] for x in ep if x["judge"] == "perfect"}
            cc[lvl]["epoch_vs_random_avg_ci_excludes0"] = f"{sum(x['vs_random']['avg_0.5N_8N']['ci95'][1] < 0 for x in ep)}/{len(ep)}"
            # budget ratio none/epoch at each threshold (TTTS + uniform, perfect)
            br = {}
            for st in ("top_two_thompson", "uniform"):
                for t in ("0.0", "0.01", "0.02"):
                    q = pair(L, "perfect", "epoch", st)["crossing_paired"][t]["vs_none"]
                    br[f"{st}|thr{t}"] = {"point_none": q["point_ctrl"], "point_epoch": q["point_prior"],
                                          "median_ratio": q["ratio_ctrl_over_prior_median"], "p10_p90": q["ratio_p10_p90"],
                                          "share_epoch_strictly_earlier": q["share_prior_earlier"],
                                          "share_tie_or_later": round(1 - q["share_prior_earlier"], 3)}
            cc[lvl]["budget_ratio_none_over_epoch_perfect"] = br
            # point estimate outside its own percentile CI
            nout = sum(not (x["ci95_lo"][k] <= x["mean_regret"][k] <= x["ci95_hi"][k]) for x in L["rows"] for k in range(len(MULTS)))
            cc[lvl]["rows_x_mults_point_outside_ci"] = f"{nout}/{len(L['rows']) * len(MULTS)}"
            ref = L["reference_perfect_none_uniform_25N"]
            cc[lvl]["reference"] = {"point": ref["point"], "ci95": ref["ci95"], "per_run": ref["per_run_point"]}
        checks[tag] = cc
    V["claim_checks_on_y1_json"] = checks

    # knife-edge check of the full / TTTS / perfect / none crossing at thr 0.01
    L = y1["levels"]["full"]
    ref = L["reference_perfect_none_uniform_25N"]["point"]
    none_t = row(L, "perfect", "none", "top_two_thompson")["mean_regret"]
    V["knife_edge_full_ttts_perfect_none"] = {"reference": ref, "target_thr0.01": ref + 0.01,
                                              "regret_by_mult": dict(zip(map(str, MULTS), none_t)),
                                              "margin_at_1.5N": round(none_t[MULTS.index(1.5)] - (ref + 0.01), 4),
                                              "margin_at_2N": round(none_t[MULTS.index(2)] - (ref + 0.01), 4)}

    # rerun reproducibility
    rr = {}
    for tag, a_, b_ in (("main", Y1, RERUN), ("splits", Y1S, RERUN_S)):
        if not b_.is_file():
            rr[tag] = "rerun file missing"
            continue
        A, B = json.loads(a_.read_text()), json.loads(b_.read_text())
        mx = 0.0
        for lvl in A["levels"]:
            ra = {(x["judge"], x["prior"], x["strategy"]): x for x in A["levels"][lvl]["rows"]}
            for x in B["levels"][lvl]["rows"]:
                y = ra[(x["judge"], x["prior"], x["strategy"])]
                for k in ("mean_regret", "ci95_lo", "ci95_hi"):
                    mx = max(mx, float(np.abs(np.array(x[k]) - np.array(y[k])).max()))
            pa = {(x["judge"], x["prior"], x["strategy"]): x for x in A["levels"][lvl]["paired"]}
            for x in B["levels"][lvl]["paired"]:
                y = pa[(x["judge"], x["prior"], x["strategy"])]
                for c in ("vs_random", "vs_epoch", "vs_none"):
                    if c in x:
                        mx = max(mx, abs(x[c]["avg_0.5N_8N"]["point"] - y[c]["avg_0.5N_8N"]["point"]),
                                 *[abs(u - v) for u, v in zip(x[c]["avg_0.5N_8N"]["ci95"], y[c]["avg_0.5N_8N"]["ci95"])])
        rr[tag] = {"max_abs_diff_rows_and_paired": mx, "judge_fit_identical": A["judge_fits"] == B["judge_fits"]}
    V["rerun_reproducibility"] = rr
    V["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(V, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("wrote", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
